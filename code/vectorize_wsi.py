"""
This file is use to vectorize all files given a folder containing wsi tif files
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
from utils import check_file_exists, get_file_list
from nic.vectorize_wsi import vectorize_wsi
from nic.util_fns import cache_file


def vectorize_images(input_dir, mask_dir, output_dir,  cache_dir, image_level, patch_size):
    """
    Converts a set of whole-slide images into numpy arrays with valid tissue patches for fast processing.

    :param input_dir: folder containing the whole-slide images.
    :param mask_dir: folder containing the whole-slide masks.
    :param output_dir: destination folder to store the vectorized images.
    :param cache_dir: folder to store whole-slide images temporarily for fast access.
    :param image_level: image resolution to read the patches.
    :param patch_size: size of the read patches.
    :return: nothing
    """

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read image file names
    image_paths = get_file_list(input_dir, ext='tif')  # get all the wsi.svs files

    # Read mask file names
    mask_paths = get_file_list(mask_dir)  # get all the mask files

    total_images = len(image_paths)

    for index in range(total_images):
        image_id = (os.path.basename(image_paths[index])).split('.')[0]
        output_pattern = output_dir + '/' + image_id + '_{item}.npy'  # by convection on NIC it has to be an .npy
        vectorized_png = output_dir + '/' + image_id + '_{item}.png'
        if not check_file_exists(vectorized_png):
            print(f'Processing image {image_id}')
            vectorize_wsi(
                image_path=cache_file(image_paths[index], cache_dir, overwrite=False),
                mask_path=mask_paths[index],
                output_pattern=output_pattern,
                image_level=image_level,
                mask_level=image_level,
                patch_size=patch_size,
                stride=patch_size,
                downsample=1,
                select_bounding_box=False
                )
            print(f'Successful vectorized {image_id} : {total_images - index} images left')
        else:
            print(f'Already existing file {image_id} - {total_images - index - 1} images left')
    print('Finish Processing All images!')


if __name__ == '__main__':
    data_dir = '/mnt/netcache/pathology/archives/lung/TCGA_LUSC/'
    image_path =  data_dir + 'wsi_diagnostic_tif'
    mask_path = data_dir + 'tissue_masks_diagnostic'
    output_dir = data_dir + 'results/vectorized'
    cache_dir = data_dir

    vectorize_images(image_path, mask_path, output_dir, cache_dir, image_level=1, patch_size=128)
