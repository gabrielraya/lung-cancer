"""
Performs GradCAM analysis in compressed WSIs.
"""

import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available

from matplotlib import pyplot as plt
from os.path import dirname, join, exists
import os
import numpy as np
import keras
from keras import backend as K
from nic.vectorize_wsi import vectorize_wsi, find_tissue_bounding_box
from scipy.ndimage.interpolation import zoom
import multiresolutionimageinterface as mri
from scipy.stats import mode
import math
from PIL import Image
from scipy.ndimage import gaussian_filter

from nic.util_fns import cache_file
from data_processing import read_data


def grad_cam_fn(input_model, output_unit, layer_name):
    """
    Keras function that computes gradients of output node w.r.t. selected layer.

    :param input_model: trained model to apply GradCAM.
    :param layer_name: layer used to compute GradCAM.
    :param output_unit: output unit used to compute GradCAM.
    :return: keras function
    """

    y_c = input_model.output[0, output_unit]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    gradient_function = K.function([input_model.input], [conv_output, grads, y_c])

    return gradient_function


def grad_cam(image, gradient_function):
    """
    Performs GraCAM on given model and image. Adapted from https://github.com/totti0223/gradcamplusplus.

    :param image: input batch.
    :param gradient_function: keras function producing the gradient of the output with respect to a given
    convolutional layer, and the output of this layer.
    :return: GradCAM results.
    """

    # Compute output of selected layer and gradients
    output, grads_val, pred = gradient_function([image])
    output = output[0, :]
    grads_val = grads_val[0, :, :, :]

    # Average gradient feature maps (into weights)
    weights = np.mean(grads_val, axis=(0, 1))

    # Scale output of selected layer by grad weights
    cam = np.dot(output, weights)

    # Zero negative values
    cam = np.maximum(cam, 0)

    # Resize CAM values to desired image size
    cam = zoom(cam, image.shape[1]/cam.shape[0])

    # Normalize max value to 1
    cam = cam / cam.max() if cam.max() != 0 else cam

    return cam, pred


def gradcam_on_features(features_path, distance_map_path, gradient_function, output_npy_path, output_png_path):
    """
    Computes GradCAM on a compressed (featurized) image.

    :param features_path: path to compressed image.
    :param distance_map_path: path to associated distance map.
    :param gradient_function: keras function producing the gradient of the output with respect to a given
    convolutional layer, and the output of this layer.
    :param output_npy_path: destination path (NPY file).
    :param output_png_path: destination path of heatmap image.
    :return: nothing.
    """

    def load_features_crop(features_path, distance_map_path=None, crop_size=400):

        # Import features
        features_orig = np.load(features_path).astype('float32').transpose((1, 2, 0))

        if distance_map_path is not None:
            # Get distance map
            distance_map = np.load(distance_map_path).astype('float32')

            # Sample center
            center = np.argmax(distance_map.flatten())
            center = np.unravel_index(center, distance_map.shape)
            x_center, y_center = center

            # Crop params
            x_size = features_orig.shape[0]
            y_size = features_orig.shape[1]
            x1 = x_center - crop_size // 2
            x2 = x_center + crop_size // 2
            y1 = y_center - crop_size // 2
            y2 = y_center + crop_size // 2

            # Pad
            padx1 = np.abs(np.min([0, x1]))
            padx2 = np.abs(np.min([0, x_size - x2]))
            pady1 = np.abs(np.min([0, y1]))
            pady2 = np.abs(np.min([0, y_size - y2]))
            padding = ((padx1, padx2), (pady1, pady2), (0, 0))
            features = np.pad(features_orig, padding, 'constant')
            x1 += padx1
            x2 += padx1
            y1 += pady1
            y2 += pady1

            # Crop
            features = features[x1:x2, y1:y2, :]

        else:
            features = features_orig

        return features

    # Load features (exact same crop used for testing)
    features = load_features_crop(features_path, distance_map_path, crop_size=400)
    features = np.expand_dims(features, axis=0)

    # Compute gradcam
    grads, preds = grad_cam(
        image=features,
        gradient_function=gradient_function
    )

    # Store
    np.save(output_npy_path.format(preds=preds), grads)

    # Normalize
    if (np.max(grads) - np.min(grads)) != 0:
        grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
    else:
        grads = (grads - np.min(grads))

    # Plot
    plt.imsave(output_png_path.format(preds=preds), grads, vmin=0, vmax=1)
    plt.close()

    return output_npy_path.format(preds=preds), output_png_path.format(preds=preds)


def gradcam_on_dataset(data_dir, csv_path, partitions, model_path, layer_number=1, custom_objects=None,
                       cache_dir=None, images_dir=None, vectorized_dir=None, output_dir=None, predict_two_output=True):
    """
    Applies GradCAM to a set of images.

    :param data_dir: path to compressed (featurized) images.
    :param csv_path: list of slides.
    :param partitions: list of partitions to select slides.
    :param model_path: path to trained model.
    :param layer_number: number of convolutional layer used to compute GradCAM.
    :param output_unit: output unit in the final layer of the network to compute GradCAM.
    :param custom_objects: used to load the model.
    :param cache_dir: folder to store compressed images temporarily.
    :return: nothing
    """

    # Output dir
    output_dir = join(result_dir, 'gradcam') if output_dir is None else output_dir

    if not exists(output_dir):
        os.makedirs(output_dir)

    print('GradCAM in directory: {d} with content {c}'.format(
        d=output_dir,
        c=os.system("ls " + output_dir)
    ), flush=True)

    # List features
    data_config = {'data_dir_luad': data_dir[0], 'data_dir_lusc': data_dir[1], 'csv_path': csv_path}
    image_ids, paths, dm_paths, labels, features_ids = read_data(data_config, custom_augmentations=None)

    # Load model
    K.set_learning_phase(
        0)  # required to avoid bug "You must feed a value for placeholder tensor 'batch_normalization_1/keras_learning_phase' with dtype bool"
    model = keras.models.load_model(model_path, custom_objects=None)

    # get firt layer name
    layer_name = model.layers[1].name

    # Load gradient function
    gradient_function_0 = grad_cam_fn(model, 0, layer_name)

    if predict_two_output:
        gradient_function_1 = grad_cam_fn(model, 1, layer_name)
    else:
        gradient_function_1 = None

    # Analyze features
    for i, (image_id, path, dm_path, label, features_id) in enumerate(
            zip(image_ids, paths, dm_paths, labels, features_ids)):

        try:
            print('Computing GradCAM on {filename} ... {i}/{n}'.format(
                filename=features_id, i=i + 1, n=len(image_ids)), flush=True)

            output_npy_path0, output_png_path0 = gradcam_on_features(
                features_path=cache_file(path, cache_dir, overwrite=False),
                distance_map_path=cache_file(dm_path, cache_dir, overwrite=False),
                gradient_function=gradient_function_0,
                output_npy_path=join(output_dir,
                                     features_id + '_{unit}_{preds}_gradcam.npy'.format(unit=0, preds='{preds:0.3f}')),
                output_png_path=join(output_dir,
                                     features_id + '_{unit}_{preds}_gradcam.png'.format(unit=0, preds='{preds:0.3f}')),
            )

            if predict_two_output:
                output_npy_path1, output_png_path1 = gradcam_on_features(
                    features_path=cache_file(path, cache_dir, overwrite=False),
                    distance_map_path=cache_file(dm_path, cache_dir, overwrite=False),
                    gradient_function=gradient_function_1,
                    output_npy_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.npy'.format(unit=1,
                                                                                                        preds='{preds:0.3f}')),
                    output_png_path=join(output_dir, features_id + '_{unit}_{preds}_gradcam.png'.format(unit=1,
                                                                                                        preds='{preds:0.3f}')),
                )

            if (images_dir is not None) and (vectorized_dir is not None):
                image_crop_from_wsi(
                    wsi_path=join(images_dir, image_id + '.tif'),
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


def image_crop_from_wsi(wsi_path, vectorized_im_shape_path, distance_map_path, output_npy_path, output_png_path,
                        crop_size=400):
    """
    Extract a patch from the whole-slide image corresponding to the exact crop used during evaluation.

    :param wsi_path: path to slide image file.
    :param vectorized_im_shape_path: path to numpy array containing shape of vectorized image.
    :param distance_map_path: path to numpy array containing distance map.
    :param output_npy_path: destination numpy array.
    :param output_png_path: destination PNG path with crop image.
    :param crop_size: size of crop image.
    :return: nothing.
    """

    # Read mask
    image_reader = mri.MultiResolutionImageReader()
    image = image_reader.open(wsi_path)

    # Bounding box with tissue
    level_bb = image.getNumberOfLevels() - 2
    x1, x2, y1, y2 = find_tissue_bounding_box(wsi_path, level_bb)

    # Retrieve patch
    image_tile = image.getUCharPatch(
        x1,
        y1,
        math.ceil((x2 - x1) / (2 ** (level_bb))),
        math.ceil((y2 - y1) / (2 ** (level_bb))),
        level_bb
    )

    # Read feature shape
    features_size = np.load(vectorized_im_shape_path)[::-1]
    image_shape = image_tile.shape
    image_feature_ratio = math.ceil(image_shape[0] / features_size[0])

    # Downsample
    image_tile = image_tile[::image_feature_ratio, ::image_feature_ratio, ...]

    # Get distance map
    distance_map = np.load(distance_map_path).astype('float32')

    # Sample center
    center = np.argmax(distance_map.flatten())
    center = np.unravel_index(center, distance_map.shape)
    x_center, y_center = center

    # Crop params
    x_size = image_tile.shape[0]
    y_size = image_tile.shape[1]
    x1 = x_center - crop_size // 2
    x2 = x_center + crop_size // 2
    y1 = y_center - crop_size // 2
    y2 = y_center + crop_size // 2

    # Pad
    padx1 = np.abs(np.min([0, x1]))
    padx2 = np.abs(np.min([0, x_size - x2]))
    pady1 = np.abs(np.min([0, y1]))
    pady2 = np.abs(np.min([0, y_size - y2]))
    padding = ((padx1, padx2), (pady1, pady2), (0, 0))
    image_tile_crop = np.pad(image_tile, padding, 'constant', constant_values=mode(image_tile.flatten()).mode[0])

    x1 += padx1
    x2 += padx1
    y1 += pady1
    y2 += pady1

    # Crop
    image_tile_crop = image_tile_crop[x1:x2, y1:y2, ...]

    # Store
    np.save(output_npy_path, image_tile_crop)
    Image.fromarray(image_tile_crop).save(output_png_path)


def overlay_gradcam_heatmap(gradcam_npy_path, image_npy_path, output_png_path):
    """
    Draws image crop and GradCAM activity together.

    :param gradcam_npy_path: path to GradCAM activity file.
    :param image_npy_path: path to cropped image.
    :param output_png_path: destination file.
    :return: nothing.
    """

    # Read
    gradcam = np.load(gradcam_npy_path)
    image = np.load(image_npy_path)

    # Format
    from scipy.ndimage import gaussian_filter, uniform_filter
    gradcam = gaussian_filter(gradcam, sigma=3)
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    image = (image - image.min()) / (image.max() - image.min())
    gradcam[gradcam < np.percentile(gradcam, 75)] = np.nan

    # Blend
    plt.figure(frameon=False)
    plt.imshow(image, vmin=0, vmax=1)
    plt.imshow(gradcam, alpha=0.4, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def overlay_gradcam_heatmap_bicolor(gradcam_npy_path1, gradcam_npy_path2, image_npy_path, output_png_path):
    """
    Draws image crop and GradCAM activity together.

    :param gradcam_npy_path: path to GradCAM activity file.
    :param image_npy_path: path to cropped image.
    :param output_png_path: destination file.
    :return: nothing.
    """

    def process_gradcam(input_path):
        gradcam = np.load(input_path)
        gradcam = gaussian_filter(gradcam, sigma=3)
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        gradcam[gradcam < np.percentile(gradcam, 75)] = np.nan
        return gradcam

    # Read
    gradcam1 = process_gradcam(gradcam_npy_path1)
    gradcam2 = process_gradcam(gradcam_npy_path2)
    image = np.load(image_npy_path)
    image = (image - image.min()) / (image.max() - image.min())

    # Blend
    plt.figure(frameon=False)
    plt.imshow(image, vmin=0, vmax=1)
    plt.imshow(gradcam1, alpha=0.4, cmap='Blues_r', vmin=0, vmax=1)
    plt.imshow(gradcam2, alpha=0.4, cmap='Greens_r', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':

    # image_crop_from_wsi(
    #     wsi_path=r"W:\projects\pathology-liver-survival\data\images\Batch4\PR_S01_P000901_C0001_L09_A15.mrxs",
    #     vectorized_im_size_path=r"W:\projects\pathology-liver-survival\results\vectorized\rotterdam1\PR_S01_P000901_C0001_L09_A15_im_shape.npy",
    #     distance_map_path=r"W:\projects\pathology-liver-survival\results\featurized\rotterdam1\bigan\nic\PR_S01_P000901_C0001_L09_A15_0_none_features_distance_map.npy",
    #     output_npy_path=r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_2\gradcam\PR_S01_P000901_C0001_L09_A15_0_none_gradcam_image.npy",
    #     output_png_path=r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_2\gradcam\PR_S01_P000901_C0001_L09_A15_0_none_gradcam_image.png",
    #     crop_size=400
    # )
    #
    # overlay_gradcam_heatmap(
    #     gradcam_npy_path=r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_2\gradcam\PR_S01_P000901_C0001_L09_A15_0_none_gradcam.npy",
    #     image_npy_path=r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_2\gradcam\PR_S01_P000901_C0001_L09_A15_0_none_gradcam_image.npy",
    #     output_png_path=r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_2\gradcam\PR_S01_P000901_C0001_L09_A15_0_none_gradcam_heatmap.png"
    # )

    images_dir = r'W:\projects\pathology-liver-survival\data\images'
    vectorized_dir = r'W:\projects\pathology-liver-survival\results\vectorized\rotterdam1'
    featurized_dir = r'W:\projects\pathology-liver-survival\results\featurized\rotterdam1\bigan\nic'
    csv_path = r"W:\projects\pathology-liver-survival\data\clinical\slide_list_hgpbin.csv"
    folds = [
        {'training': ['partition_0', 'partition_1'], 'validation': ['partition_2'], 'test': ['partition_3']},
        {'training': ['partition_1', 'partition_2'], 'validation': ['partition_3'], 'test': ['partition_0']},
        {'training': ['partition_2', 'partition_3'], 'validation': ['partition_0'], 'test': ['partition_1']},
        {'training': ['partition_3', 'partition_0'], 'validation': ['partition_1'], 'test': ['partition_2']},
    ]
    fold = 0
    model_path = r"W:\projects\pathology-liver-survival\results\models\rotterdam1\bigan\nic\hgp_bin\fold_{i_fold}\checkpoint.h5".format(i_fold=fold)

    gradcam_on_dataset(
        images_dir=images_dir,
        vectorized_dir=vectorized_dir,
        featurized_dir=featurized_dir,
        csv_path=csv_path,
        model_path=model_path,
        partitions=folds[fold]['test'],
        layer_name='separable_conv2d_1',
        custom_objects=None,
        cache_dir=None,
        output_dir=r'C:\Users\david\Downloads\debug_survival\gradcam\bothunits'
    )












