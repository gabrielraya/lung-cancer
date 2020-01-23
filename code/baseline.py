# Import NIC to python path
import sys
import os

nic_dir = '/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/neural-image-compression-private'
sys.path.append(nic_dir + '/source')

# Copy data
print('Copying data to local instance')
os.system('mkdir /home/user/featurized_tcga_luad/')
os.system('mkdir /home/user/featurized_tcga_lusc/')
os.system('cp /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/results/tcga_luad/featurized/no_augmentations/* /home/user/featurized_tcga_luad')
os.system('cp /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/results/tcga_lusc/featurized/no_augmentations/* /home/user/featurized_tcga_lusc')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, shutil
from os.path import join, dirname, exists
import keras
from gradcam_wsi import gradcam_on_dataset
from preprocessing import data_to_csv, create_csv, generate_csv_files
from model_training import train_wsi_classifier, eval_model, compute_metrics
from utils import check_file_exists


def train_model(featurized_dir, csv_path, fold_n, output_dir, cache_dir, batch_size=16, epochs=32,
                images_dir=None, vectorized_dir=None, lr=1e-2, patience=4, delete_folder=False,
                occlusion_augmentation=False, elastic_augmentation=False, shuffle_augmentation=None):
    """
    Trains a CNN using compressed whole-slide images.

    :param featurized_dir: folder containing the compressed (featurized) images.
    :param csv_path: list of slides with labels.
    :param fold_n: fold determining which data partitions to use for training, validation and testing.
    :param output_dir: destination folder to store results.
    :param cache_dir: folder to store compressed images temporarily for fast access.
    :param batch_size: number of samples to train with in one-go.
    :return: nothing.
    """

    # Delete folder and subfolders if exists
    if delete_folder:
        if exists(result_dir):  shutil.rmtree(result_dir)

    # Train CNN
    train_wsi_classifier(
        data_dir=featurized_dir,
        csv_path=csv_path,
        partitions=None,
        crop_size=400,
        output_dir=output_dir,
        output_units=2,
        cache_dir=cache_dir,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        code_size=128,
        workers=1,
        train_step_multiplier=1,
        val_step_multiplier=0.5,
        keep_data_training=1,
        keep_data_validation=1,
        patience=patience,
        occlusion_augmentation=occlusion_augmentation,
        elastic_augmentation=elastic_augmentation,
        shuffle_augmentation=shuffle_augmentation
    )

    # Evaluate CNN

    # Get compressed wsi directories with csv test file
    data_config = featurized_dir
    data_config['csv_path'] = csv_path['csv_test']

    eval_model(
        model_path=join(output_dir, 'checkpoint.h5'),
        data_config=data_config,
        crop_size=400,
        output_path=join(output_dir, 'eval', 'preds.csv'),
        cache_dir=None,
        batch_size=batch_size,
        keep_data=1
    )

    # Metrics
    try:
        compute_metrics(
            input_path=join(output_dir, 'eval', 'preds.csv'),
            output_dir=join(output_dir, 'eval')
        )
    except Exception as e:
        print('Failed to compute metrics. Exception: {e}'.format(e=e), flush=True)


if __name__ == '__main__':
    # project and data directories
    root_dir = r'/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction'
    data_dir = r'/mnt/netcache/pathology/archives/lung'

    # wsi directories
    dir_luad_wsi = os.path.join(data_dir, 'TCGA_LUAD', 'wsi_diagnostic_tif')
    dir_lusc_wsi = os.path.join(data_dir, 'TCGA_LUSC', 'wsi_diagnostic_tif')
    dir_luad_wsi_mask = os.path.join(data_dir, 'TCGA_LUAD', 'tissue_masks_diagnostic')
    dir_lusc_wsi_mask = os.path.join(data_dir, 'TCGA_LUSC', 'tissue_masks_diagnostic')

    # compressed image directories
    vectorized_luad_dir = join(root_dir, 'results', 'tcga_luad', 'vectorized')
    vectorized_lusc_dir = join(root_dir, 'results', 'tcga_lusc', 'vectorized')
    featurized_luad_dir = join(root_dir, 'results', 'tcga_luad', 'featurized', 'no_augmentations')
    featurized_lusc_dir = join(root_dir, 'results', 'tcga_lusc', 'featurized', 'no_augmentations')

    # results directory
    result_dir = join(root_dir, 'results', 'model')  # store the results from trained model
    gradcam_dir = join(result_dir, 'gradcam')  # store gradcam results

    # Set paths
    model_path = './neural-image-compression-private/models/encoders_patches_pathology/encoder_bigan.h5'
    csv_train = os.path.join(root_dir, 'data', 'train_slide_list_tcga.csv')
    csv_val = os.path.join(root_dir, 'data', 'validation_slide_list_tcga.csv')
    csv_test = os.path.join(root_dir, 'data', 'test_slide_list_tcga.csv')
    csv_path_luad_feat = join(root_dir, 'data', 'slide_list_featurized_luad.csv')

    # csv paths
    csv_path_wsi = os.path.join(root_dir, 'data', 'slide_original_list_tcga.csv')
    csv_path_compressed_wsi = os.path.join(root_dir, 'data', 'slide_compressed_list_tcga.csv')

    cache_dir = None  # used to store local copies of files during I/O operations (useful in cluster


    # Train CNN

    # selected_fold = 0

    featurized_dir = {'data_dir_luad': featurized_luad_dir, 'data_dir_lusc': featurized_lusc_dir}
    csv_path = {'csv_train': csv_train, 'csv_val': csv_val, 'csv_test': csv_test}

    # Create csv files
    print('Creating compressed wsi csv file ...')
    create_csv(featurized_luad_dir, featurized_lusc_dir, csv_path_compressed_wsi)

    print('Creating split train/validation/test csv files with no augmentations ...')
    generate_csv_files(csv_path_compressed_wsi, csv_train, csv_val, csv_test, test_size=0.2, validation_size = 0.3)

    # read files to check shapes
    df = pd.read_csv(csv_train);  df2 = pd.read_csv(csv_val);   df3 = pd.read_csv(csv_test)
    print(f'Files were read with shapes: Training: {df.shape}, Validation {df2.shape}, Testing {df3.shape}')
    print(f'Total files: Files were read with shapes: {df.shape[0]+df2.shape[0]+df3.shape[0]}')

    train_model(
        featurized_dir=featurized_dir,
        csv_path=csv_path,
        fold_n=0,
        output_dir=result_dir,
        cache_dir=None,
        batch_size=12,
        epochs=100,
        delete_folder=True,
        occlusion_augmentation=False,
        lr=1e-2,
        patience=4,
        elastic_augmentation=False,
        images_dir=None,  # required for GradCAM
        vectorized_dir=None,  # required for GradCAM
        shuffle_augmentation=None
    )

    print('GradCam will be apply to this dataset!')
    data_to_csv(featurized_luad_dir, csv_path_luad_feat)

    # Apply GradCam on layer 1
    gradcam_on_dataset(
        data_dir=[featurized_luad_dir, featurized_lusc_dir],
        csv_path=csv_path_luad_feat,
        model_path=join(result_dir, 'checkpoint.h5'),
        partitions=0,
        layer_number=1,
        custom_objects=None,
        cache_dir=cache_dir,
        images_dir=dir_luad_wsi,
        vectorized_dir=vectorized_luad_dir,
        output_dir=gradcam_dir,
        predict_two_output = True
    )
