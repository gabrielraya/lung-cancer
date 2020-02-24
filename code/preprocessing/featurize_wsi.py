"""
This file is use to featurize all files given a folder containing vectorized files
Note: Before running this

    import sys
	sys.path.append("neural-image-compression-private\\source")
	print(sys.path) # you will see the path added

# Neural Compression
# vectorize -> get non-background patches
# featurize -> compress a wsi
# at this point we have a compressed image we can use to classify

"""

# Import NIC to python path
import sys
print('Adding Neural Image Compression library to python path')
sys.path.append('/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/neural-image-compression-private/source')

import os
from os.path import join
import keras
from utils import check_file_exists, get_file_list
from nic.featurize_wsi import encode_wsi_npy_simple, encode_augment_wsi


def downsample_encoder_128_to_64(encoder):
    input_layer = keras.layers.Input((128, 128, 3))
    x = keras.layers.AveragePooling2D()(input_layer)
    x = encoder(x)
    encoder_ds = keras.models.Model(inputs=input_layer, outputs=x)
    return encoder_ds


def featurize_images(input_dir, model_path, output_dir, batch_size, downsample_encoder=True):
    """
    Featurizes vectorized of whole-slide images using a trained encoder network.

    :param input_dir: directory containing the vectorized images.
    :param model_path: path to trained encoder network.
    :param output_dir: destination folder to store the compressed images.
    :param batch_size: number of images to process in the GPU in one-go.
    :param downsample_encoder: if true downsample image from 128 to 64
    :return: nothing
    """

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load encoder model
    encoder = keras.models.load_model(model_path, compile=False)

    # Downsample image to fit encoder needed for bigan encoder
    if downsample_encoder:
        encoder = downsample_encoder_128_to_64(encoder)

    image_list = get_file_list(input_dir, ext='_{item}.png')
    total_images = len(image_list)

    for index in range(total_images):
        filename = os.path.splitext(os.path.basename(image_list[index]))[0]
        filename_npy = input_dir + '/' + filename + '.npy' # by convection on NIC it has to be an .npy
        featurized_npy = output_dir + '/' + filename.split('_')[0] + '.npy'
        featurized_png = output_dir + '/' + filename.split('_')[0] + '.png'
        if not check_file_exists(featurized_npy):
            print(f'Processing image {filename}')
            encode_wsi_npy_simple(encoder, filename_npy, batch_size, featurized_npy, featurized_png, output_distance_map=True)
            print(f'Successful vectorized {filename} : {total_images - index - 1} images left')
        else:
            print(f'Already existing file {filename} - {total_images - index - 1} images left')
    print('Finish Processing All images!')


def featurize_images_augmented(input_dir, model_path, output_dir, batch_size, downsample_encoder=True):
    """
    Compresses a set of whole-slide aumented images using a trained encoder network.

    :param input_dir: directory containing the vectorized images.
    :param model_path: path to trained encoder network.
    :param output_dir: destination folder to store the compressed images.
    :param batch_size: number of images to process in the GPU in one-go.
    :param downsample_encoder: if true downsample image from 128 to 64
    :return: nothing
    """

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load encoder model
    encoder = keras.models.load_model(model_path, compile=False)

    # Downsample image to fit encoder needed for bigan encoder
    if downsample_encoder:
        encoder = downsample_encoder_128_to_64(encoder)

    image_list = get_file_list(input_dir, ext='_{item}.png')
    total_images = len(image_list)

    for index in range(total_images):
        filename = os.path.splitext(os.path.basename(image_list[index]))[0]
        filename_npy = input_dir + '/' + filename + '.npy' # by convection on NIC it has to be an .npy
        wsi_pattern = input_dir + '/' + filename.split('_')[0] + '_{item}.npy'
        if check_file_exists(wsi_pattern.format(item='im_shape')):
            print(f'Processing image {filename}')
            encode_augment_wsi(wsi_pattern=filename_npy, encoder =encoder, output_dir=output_dir,
                                batch_size=batch_size, aug_modes=[('none', 0), ('none', 90), ('none', 180),
                                ('none', 270), ('horizontal', 0), ('vertical', 0),('vertical', 90),
                                ('vertical', 270)], overwrite=False)
            print(f'Successful vectorized {filename} : {total_images - index - 1} images left')
        else:
            print('Vectorized file not found: {f}'.format(f=wsi_pattern.format(item='im_shape')), flush=True)
    print('Finish Processing All images!')


if __name__ == '__main__':
    """
    To run the scrip just change the data_dir name
    
    The following will run bigan encoder with augmentations:
        
            python3 featurize_wsi.py 1 1
            
    The following will run 4task encoder with no augmentations:
        
            python3 featurize_wsi.py 0 0       
    """
    # Get parameters
    bigan = int(sys.argv[1])
    augmented = int(sys.argv[2])

    # project and data directories
    root_dir = r'/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction'

    # compressed image directories
    vectorized_luad_dir = join(root_dir, 'results', 'tcga_luad', 'vectorized')
    vectorized_lusc_dir = join(root_dir, 'results', 'tcga_lusc', 'vectorized')

    encoders = ['4task','bigan']

    if bigan:
        model_path = nic_dir + '/models/encoders_patches_pathology/encoder_bigan.h5'
        featurized_luad_dir = join(root_dir, 'results', 'tcga_luad', 'featurized', 'no_augmentations')
        featurized_lusc_dir = join(root_dir, 'results', 'tcga_lusc', 'featurized', 'no_augmentations')
        featurized_luad_dir_aug = join(root_dir, 'results', 'tcga_luad', 'featurized', 'augmented')
        featurized_lusc_dir_aug = join(root_dir, 'results', 'tcga_lusc', 'featurized', 'augmented')

    else:
        model_path = r'/mnt/netcache/pathology/projects/pathology-proacting/neoadjuvant_nki/nic/encoder_zoo/supervsied_enc_2019_4tasks.h5'
        featurized_luad_dir = join(root_dir, 'results', 'tcga_luad', 'featurized', 'no_augmentations_4task')
        featurized_lusc_dir = join(root_dir, 'results', 'tcga_lusc', 'featurized', 'no_augmentations_4task')
        featurized_luad_dir_aug = join(root_dir, 'results', 'tcga_luad', 'featurized', 'augmented_4task')
        featurized_lusc_dir_aug = join(root_dir, 'results', 'tcga_lusc', 'featurized', 'augmented_4task')

    if augmented:
        print(f'Running featurizing with augmentations using {encoders[bigan]} encoder')
        featurize_images_augmented(vectorized_luad_dir, model_path, featurized_luad_dir_aug, batch_size=32)
        featurize_images_augmented(vectorized_lusc_dir, model_path, featurized_lusc_dir_aug, batch_size=32)
    else:
        print(f'Running featurizing with no augmentations using {encoders[bigan]} encoder')
        featurize_images(vectorized_luad_dir, model_path, featurized_luad_dir, batch_size=32)
        featurize_images(vectorized_lusc_dir, model_path, featurized_lusc_dir, batch_size=32)