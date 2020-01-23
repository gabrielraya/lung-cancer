"""
Train a CNN on compressed whole-slide images.

    class 1 : luad
    class 0 : lusc

    Dataset:
        531 in LUAD
        506 in LUSC
"""
# Copy data to local instance
cluster: bool = False

import os
if cluster:
    os.system('mkdir tcga_luad/')
    os.system('mkdir tcga_lusc/')
    os.system('cp /mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/results/tcga/featurized/tcga_luad/normal/* ./tcga_luad')
    os.system('cp /mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/results/tcga/featurized/tcga_lusc/normal/* ./tcga_lusc')
    # Import NIC to python path
    import sys
    nic_dir = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/code/neural-image-compression-private'
    sys.path.append(nic_dir +'/source')

import keras
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras import backend as K
from nic.util_fns import cache_file
from nic.train_compressed_wsi import  f1_score_plot
from nic.gradcam_wsi import gradcam_on_features, grad_cam_fn, image_crop_from_wsi, overlay_gradcam_heatmap, overlay_gradcam_heatmap_bicolor
import glob
from os.path import exists, join, basename, dirname
import shutil
from data_processing import read_data, FeaturizedWsiSequence, FeaturizedWsiGenerator
from nic.callbacks import ReduceLROnPlateau, ModelCheckpoint, HistoryCsv, FinishedFlag, PlotHistory, StoreModelSummary, \
    CopyResultsExternally, LearningRateScheduler
from model import build_wsi_classifier


def fit_model(training_generator, validation_generator, output_dir, model, n_epochs, train_step_multiplier, workers,
              patience, custom_objects=None, monitor='val_loss', mode='min', loss_list=['loss', 'val_loss'],
              metric_list=['categorical_accuracy', 'val_categorical_accuracy'], val_step_multiplier=1.0, min_lr=1e-4,
              extra_callbacks=[], cache_dir=None, lr_scheduler_fn=None):

    # Cache output
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    # Ignore if training finished
    if not os.path.exists(os.path.join(output_dir, 'training_finished.txt')):

        # Prepare directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Training model in directory: {d} with content {c}'.format(
            d=output_dir,
            c=os.system("ls " + output_dir)
        ), flush=True)

        # Continue training if model found
        epochs_run = 0
        if os.path.exists(os.path.join(output_dir, 'last_epoch.h5')) and exists(join(output_dir, 'history.csv')):
            print('Resuming training from saved model ...', flush=True)
            model = keras.models.load_model(os.path.join(output_dir, 'last_epoch.h5'), custom_objects=custom_objects)
            df = pd.read_csv(join(output_dir, 'history.csv'), header=0, index_col=0)
            epochs_run = len(df)

            # Copy existing files into cache
            if cache_dir is not None and os.path.exists(cache_dir):
                for path in glob(os.path.join(output_dir, '*')):
                    try:
                        shutil.copyfile(path, join(cache_dir, basename(path)))
                    except Exception as e:
                        print(
                            'Error copying file {f} from external {output_dir} to cache {cache_dir} directory. Exception: {e}'.format(
                                f=path, output_dir=output_dir, cache_dir=cache_dir, e=e
                            ), flush=True)
        else:
            print('Training model from scratch {b1} {b2}...'.format(
                b1=exists(join(output_dir, 'last_epoch.h5')),
                b2=exists(join(output_dir, 'history.csv'))
            ), flush=True)

        if epochs_run < n_epochs:

            if cache_dir is not None and exists(cache_dir):
                external_output_dir = output_dir
                output_dir = cache_dir
            else:
                external_output_dir = None

            # Define callbacks
            callback_list = [
                StoreModelSummary(filepath=join(output_dir, 'model_summary.txt'), verbose=1),
                HistoryCsv(file_path=join(output_dir, 'history.csv'))
            ]

            if len(extra_callbacks) > 0:
                callback_list.extend(extra_callbacks)

            callback_list2 = [
                ModelCheckpoint(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'checkpoint.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=1,
                    save_best_only=True
                ),
                ModelCheckpoint(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'last_epoch.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=1,
                    save_best_only=False
                ),
                PlotHistory(
                    plot_path=join(output_dir, 'history.png'),
                    log_path=join(output_dir, 'history.csv'),
                    loss_list=loss_list,
                    metric_list=metric_list
                ),
                FinishedFlag(
                    file_path=join(output_dir, 'training_finished.txt')
                )
            ]
            callback_list.extend(callback_list2)

            if patience is not None:
                callback_list.append(
                    ReduceLROnPlateau(
                        history_path=join(output_dir, 'history.csv'),
                        monitor=monitor,
                        mode=mode,
                        factor=1.0 / 3,
                        patience=patience,
                        verbose=1,
                        cooldown=2,
                        min_lr=min_lr
                    ) if lr_scheduler_fn is None else LearningRateScheduler(schedule=lr_scheduler_fn, min_lr=min_lr)
                )

            if external_output_dir is not None:
                callback_list.append(
                    CopyResultsExternally(
                        local_dir=output_dir,
                        external_dir=external_output_dir
                    )
                )

            # Train model
            model.fit_generator(
                generator=training_generator,
                steps_per_epoch=int(len(training_generator) * train_step_multiplier),
                epochs=n_epochs,
                verbose=1,
                callbacks=callback_list,
                validation_data=validation_generator,
                validation_steps=int(
                    len(validation_generator) * val_step_multiplier) if validation_generator is not None else None,
                initial_epoch=epochs_run,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=True if workers > 1 else False
            )

            # Finish
            try:
                open(os.path.join(external_output_dir if external_output_dir is not None else output_dir,
                                  'training_finished.txt'), 'a').close()
            except:
                pass


def train_wsi_classifier(data_dir, csv_path, partitions, crop_size, output_dir, cache_dir, n_epochs, batch_size, lr,
                         output_units, code_size, workers, train_step_multiplier=1.0, keep_data_training=1.0,
                         keep_data_validation=1.0, patience=8, custom_objects=None, val_step_multiplier=1.0,
                         occlusion_augmentation=False, elastic_augmentation=False, shuffle_augmentation=None,
                         read_data_fn=None, model=None, binary_target=True, loss_list=['loss', 'val_loss'],
                         metric_list=['categorical_accuracy', 'val_categorical_accuracy'], lr_scheduler_fn=None,
                         use_validation=True, read_data_fn_eval=None, min_lr=1e-4):


    data_dir_luad = data_dir['data_dir_luad']
    data_dir_lusc = data_dir['data_dir_lusc']
    csv_train = csv_path['csv_train']
    csv_val = csv_path['csv_val']
    csv_test = csv_path['csv_test']


    print('Loading training set ...', flush=True)
    training_gen = FeaturizedWsiGenerator(
        data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_train},
        data_fn=read_data,
        batch_size=batch_size,
        augment=True,
        crop_size=crop_size,
        cache_dir=cache_dir,
        balanced=True,
        keep_data=keep_data_training,
        occlusion_augmentation=occlusion_augmentation,
        elastic_augmentation=elastic_augmentation,
        shuffle_augmentation=shuffle_augmentation
    )

    # from nic.train_compressed_wsi import FeaturizedWsiSequence
    keep_data_validation = 1
    # Validation set
    print('Loading validation set ...', flush=True)
    use_validation = True
    validation_gen = FeaturizedWsiSequence(
        data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_val},
        data_fn=read_data,
        batch_size=batch_size,
        crop_size=400,
        cache_dir=cache_dir,
        balanced=True,
        keep_data=keep_data_validation
    ) if use_validation else None

    # Create model
    print('Building model ...', flush=True)
    model = None
    if model is None:
        model = build_wsi_classifier(input_shape=(crop_size, crop_size, code_size), lr=lr, output_units=output_units)

    # Train initial model

    loss_list = ['loss', 'val_loss']
    metric_list = ['categorical_accuracy', 'val_categorical_accuracy']
    custom_objects = None
    lr_scheduler_fn = None
    min_lr = 1e-4

    print('Training model ...', flush=True)
    fit_model(
        training_generator=training_gen,
        validation_generator=validation_gen,
        output_dir=output_dir,
        model=model,
        n_epochs=n_epochs,
        train_step_multiplier=train_step_multiplier,
        val_step_multiplier=val_step_multiplier,
        workers=workers,
        patience=patience,
        monitor='val_loss' if use_validation else 'loss',
        mode='min',
        loss_list=loss_list,
        metric_list=metric_list,
        custom_objects=custom_objects,
        cache_dir=None if cache_dir is None else join(cache_dir, 'models', basename(output_dir)),
        lr_scheduler_fn=lr_scheduler_fn,
        min_lr=min_lr
    )


def eval_model(model_path, data_config, crop_size, output_path, cache_dir, batch_size,
               custom_objects=None, keep_data=1.0):
    # Output dir
    if not exists(dirname(output_path)):
        os.makedirs(dirname(output_path))

    d = dirname(output_path)
    print('Evaluating model in directory: {d} with content {c}'.format(
        d=d,
        c=os.system("ls " + d)
    ), flush=True)

    # Test set
    print('Loading test set ...', flush=True)
    test_gen = FeaturizedWsiSequence(
        data_config,
        data_fn=read_data,
        batch_size=batch_size,
        crop_size=crop_size,
        cache_dir=cache_dir,
        balanced=False,
        keep_data=keep_data,
        return_ids=True
    )

    # Load model
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    # Predictions
    ids = []
    labels = []
    preds = []
    for i in range(len(test_gen)):

        print('Predicting batch {i}/{n} ...'.format(i=i + 1, n=len(test_gen)), flush=True)
        x, y, id = test_gen[i]

        pred = model.predict_on_batch(x)
        if pred.shape[-1] > 2:
            pred = pred.argmax(axis=-1)
        else:
            pred = pred[:, 1]

        ids.extend(id)
        labels.extend(y.argmax(axis=-1))
        preds.extend(pred)

    # Format
    df = pd.DataFrame({'id': ids, 'label': labels, 'pred': preds})
    df = df.sort_values('id')
    df = df.reset_index(drop=True)

    try:
        df.to_csv(output_path)
    except FileNotFoundError as e:
        print('Failed to write file {f}. Exception: {e}'.format(f=output_path, e=e), flush=True)
        d = dirname(output_path)
        if not exists(d):
            os.makedirs(d)
        time.sleep(3)
        df.to_csv(output_path)


def plot_roc(labels, preds, output_path=None, close_fig=True, legend_label=None):
    # ROC
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    lw = 2
    l = 'ROC {tag}(area = {a:0.3f})'.format(tag='' if legend_label is None else legend_label, a=roc_auc)
    plt.plot(fpr, tpr, lw=lw, label=l)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.grid(b=True, which='both')
    plt.legend(loc="lower right")

    if output_path is not None:
        plt.savefig(output_path)
    if close_fig:
        plt.close()

    return roc_auc


def compute_metrics(input_path, output_dir, group_by_slide=False, dropnan=False):
    # Output dir
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Read
    df = pd.read_csv(input_path, header=0, index_col=0)

    # Drop nan
    if dropnan:
        df = df.loc[df.notnull().all(axis=1), :]

    # Group by slide id
    if group_by_slide:
        # needs to be adapted to your id's encoding format
        if 'slide_id' not in df.columns:
            df['slide_id'] = df['id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
        df_group = df.groupby('slide_id').mean()
        labels = df_group['label'].values.astype('int')
        preds = df_group['pred'].values
    else:
        labels = df['label'].values.astype('int')
        preds = df['pred'].values

    # Plot ROC
    roc_auc = plot_roc(labels, preds, join(output_dir, 'roc.png'))
    results = {'roc_auc': [roc_auc]}

    # F1 score
    f1_s, f1_th = f1_score_plot(labels, preds, join(output_dir, 'f1.png'))
    results['f1_score'] = [f1_s]
    results['f1_threshold'] = [f1_th]

    # Store
    pd.DataFrame(results).T.to_csv(join(output_dir, 'metrics.csv'))


# Apply GradCAM analysis to CNN

def gradcam_on_dataset(data_conf, model_path, layer_name, custom_objects=None,
                       cache_dir=None, images_dir=None, vectorized_dir=None, output_dir=None, predict_two_output=True):
    """
    Applies GradCAM to a set of images.

    :param data_dir: path to compressed (featurized) images.
    :param csv_path: list of slides.
    :param partitions: list of partitions to select slides.
    :param model_path: path to trained model.
    :param layer_name: name of convolutional layer used to compute GradCAM.
    :param output_unit: output unit in the final layer of the network to compute GradCAM.
    :param custom_objects: used to load the model.
    :param cache_dir: folder to store compressed images temporarily.
    :return: nothing
    """

    # Featurized directories
    data_dir_luad = data_conf['data_dir_luad']
    data_dir_lusc = data_conf['data_dir_lusc']
    csv_test = data_conf['csv_path']

    # Output dir
    output_dir = join(dirname(model_path), 'gradcam') if output_dir is None else output_dir
    if not exists(output_dir):
        os.makedirs(output_dir)

    print('GradCAM in directory: {d} with content {c}'.format(
        d=output_dir,
        c=os.system("ls " + output_dir)
    ), flush=True)

    # List features
    data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_test}
    image_ids, paths, dm_paths, labels, features_ids = read_data(data_config) #, custom_augmentations=[('none', 0)])

    # Load model and gradient function
    K.set_learning_phase(0)  # required to avoid bug "You must feed a value for placeholder tensor 'batch_normalization_1/keras_learning_phase' with dtype bool"
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    gradient_function_0 = grad_cam_fn(model, 0, layer_name)
    if predict_two_output:
        gradient_function_1 = grad_cam_fn(model, 1, layer_name)
    else:
        gradient_function_1 = None

    # Analyze features
    for i, (image_id, path, dm_path, label, features_id, batch_id) in enumerate(zip(image_ids, paths, dm_paths, labels, features_ids, batch_ids)):

        try:
            print('Computing GradCAM on {filename} ... {i}/{n}'.format(
                    filename=features_id, i=i+1, n=len(image_ids)), flush=True)

            output_npy_path0, output_png_path0 = gradcam_on_features(
                features_path=cache_file(path, cache_dir, overwrite=False),
                distance_map_path=cache_file(dm_path, cache_dir, overwrite=False),
                gradient_function=gradient_function_0,
                output_npy_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.npy'.format(unit=0, preds='{preds:0.3f}')),
                output_png_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.png'.format(unit=0, preds='{preds:0.3f}')),
            )

            if predict_two_output:
                output_npy_path1, output_png_path1 = gradcam_on_features(
                    features_path=cache_file(path, cache_dir, overwrite=False),
                    distance_map_path=cache_file(dm_path, cache_dir, overwrite=False),
                    gradient_function=gradient_function_1,
                    output_npy_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.npy'.format(unit=1, preds='{preds:0.3f}')),
                    output_png_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.png'.format(unit=1, preds='{preds:0.3f}')),
                )

            if (images_dir is not None) and (vectorized_dir is not None):
                image_crop_from_wsi(
                    wsi_path=join(images_dir, batch_id, image_id + '.mrxs'),
                    vectorized_im_shape_path=join(vectorized_dir, image_id + '_im_shape.npy'),
                    distance_map_path=cache_file(dm_path, cache_dir, overwrite=False),
                    output_npy_path=join(output_dir, features_id + '_image.npy'),
                    output_png_path=join(output_dir, features_id + '_image.png'),
                    crop_size=400
                )

                overlay_gradcam_heatmap(
                    gradcam_npy_path=output_npy_path0,
                    image_npy_path=join(output_dir, features_id + '_image.npy'),
                    output_png_path=join(output_dir, features_id + '_{unit}_heatmap.png'.format(unit=0))
                )

                if predict_two_output:
                    overlay_gradcam_heatmap(
                        gradcam_npy_path=output_npy_path1,
                        image_npy_path=join(output_dir, features_id + '_image.npy'),
                        output_png_path=join(output_dir, features_id + '_{unit}_heatmap.png'.format(unit=1))
                    )

                    overlay_gradcam_heatmap_bicolor(
                        gradcam_npy_path1=output_npy_path0,
                        gradcam_npy_path2=output_npy_path1,
                        image_npy_path=join(output_dir, features_id + '_image.npy'),
                        output_png_path=join(output_dir, features_id + '_both_heatmap.png')
                    )
        except Exception as e:
            print('Failed to compute GradCAM on {f}. Exception: {e}'.format(f=path, e=e), flush=True)



if __name__ == '__main__':
    if cluster:
        root_dir = '/home/user'
        data_dir_luad = '/home/user/tcga_luad'
        data_dir_lusc = '/home/user/tcga_lusc'
        csv_path = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/data/tcga/slide_list_tcga.csv'
        csv_train = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/data/tcga/train_slide_list_tcga.csv'
        csv_val = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/data/tcga/validation_slide_list_tcga.csv'
        csv_test = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/data/tcga/test_slide_list_tcga.csv'
        model_dir = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/results/model_1_batch_size_12'  # change this everytime a new model is run
    else:
        csv_path = 'E:/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/data/slide_list_tcga.csv'
        csv_train = 'E:/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/data/train_slide_list_tcga.csv'
        csv_val = 'E:/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/data/validation_slide_list_tcga.csv'
        csv_test = 'E:/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/data/test_slide_list_tcga.csv'
        root_dir = r'E:/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction'
        data_dir_luad = root_dir + r'/results/tcga_luad/featurized'
        data_dir_lusc = root_dir + r'/results/tcga_lusc/featurized'
        model_dir = root_dir + '/results/model'  # change this everytime a new model is run

        # paths = {'csv_path': csv_path, 'csv_train': csv_train, 'csv_val': csv_val, 'csv_test': csv_test}
        # generate_csv_files(paths, test_size=0.2, validation_size = 0.3)

    cache_path = None

    # Training
    multiple_paths = {'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'output_dir': model_dir,
                      'csv_train': csv_train, 'csv_val': csv_val, 'csv_test': csv_test, 'cache_path': cache_path}
    #run_train_model(multiple_paths, epochs=200, size_of_batch=12)
    run_train_model(multiple_paths, epochs=10, size_of_batch=2)
    # Model Evaluation
    #data_config = {'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_test}
    #run_eval(data_config, model_dir, batch_size=12)
    #
    # from nic.gradcam_wsi gradcam_on_dataset
    # # Apply GradCAM analysis to CNN
    # gradcam_on_dataset(
    #     featurized_dir=featurized_dir,
    #     csv_path=csv_path,
    #     model_path=join(result_dir, 'checkpoint.h5'),
    #     partitions=folds[fold_n]['test'],
    #     layer_name='separable_conv2d_1',
    #     output_unit=1,
    #     custom_objects=None,
    #     cache_dir=cache_dir,
    #     images_dir=images_dir,
    #     vectorized_dir=vectorized_dir
    # )
