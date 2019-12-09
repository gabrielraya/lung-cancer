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
nic_dir = '/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/code/neural-image-compression-private'
sys.path.append(nic_dir +'/source')

import os
import keras
from nic.featurize_wsi import encode_wsi_npy_simple, encode_augment_wsi


def get_file_list(path, ext=''):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])


def downsample_encoder_128_to_64(encoder):
    input_layer = keras.layers.Input((128, 128, 3))
    x = keras.layers.AveragePooling2D()(input_layer)
    x = encoder(x)
    encoder_ds = keras.models.Model(inputs=input_layer, outputs=x)
    return encoder_ds


def check_file_exists(filename):
    try:
        f = open(filename, 'r')
        f.close()
        return True
    except IOError:
        return False


def featurize_images(input_dir, model_path, output_dir, batch_size):
    """
    Featurizes vectorized of whole-slide images using a trained encoder network.

    :param input_dir: directory containing the vectorized images.
    :param model_path: path to trained encoder network.
    :param output_dir: destination folder to store the compressed images.
    :param batch_size: number of images to process in the GPU in one-go.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    encoder = keras.models.load_model(model_path, compile=False)
    encoder = downsample_encoder_128_to_64(encoder)  # downsample image, needed for bigan encoder.
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


def featurize_images_augmented(input_dir, model_path, output_dir, batch_size):
    """
    Compresses a set of whole-slide images using a trained encoder network.

    :param input_dir: directory containing the vectorized images.
    :param model_path: path to trained encoder network.
    :param output_dir: destination folder to store the compressed images.
    :param batch_size: number of images to process in the GPU in one-go.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    encoder = keras.models.load_model(model_path, compile=False)
    encoder = downsample_encoder_128_to_64(encoder)  # downsample image, needed for bigan encoder.
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
    """
    model_path = nic_dir + '/models/encoders_patches_pathology/encoder_bigan.h5'
    data_dir = '/mnt/netcache/pathology/archives/lung/TCGA/TCGA_LUAD/'
    input_dir =  data_dir + 'results/vectorized'
    #output_dir = data_dir + 'results/featurized'
    output_dir_aug = data_dir + 'results/featurized_augmented'

    #featurize_images(input_dir, model_path, output_dir, batch_size=32)
    featurize_images_augmented(input_dir, model_path, output_dir_aug, batch_size=32)
