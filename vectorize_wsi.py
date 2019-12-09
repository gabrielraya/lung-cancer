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
from nic.vectorize_wsi import vectorize_wsi
from nic.util_fns import cache_file


def check_file_exists(filename):
    try:
        f = open(filename, 'r')
        f.close()
        return True
    except IOError:
        return False


def get_file_list(path, ext=''):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])


def vectorize_images(image_paths, mask_paths, output_path,  cache_dir, image_level, patch_size):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    total_images = len(image_paths)
    for index in range(total_images):
        image_id = (os.path.basename(image_paths[index])).split('.')[0]
        output_pattern = output_path + '/' + image_id + '_{item}.npy'  # by convection on NIC it has to be an .npy
        vectorized_png = output_path + '/' + image_id + '_{item}.png'
        if not check_file_exists(vectorized_png):
            print(f'Processing image {image_id}')
            vectorize_wsi(cache_file(image_paths[index], cache_dir, overwrite=False), mask_paths[index], output_pattern, image_level=image_level,
                          mask_level=image_level, patch_size=patch_size, stride=patch_size, downsample=1,
                          select_bounding_box=False)

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

    image_paths = get_file_list(image_path, ext='tif')  # get all the wsi.svs files
    mask_paths = get_file_list(mask_path, ext='')  # get all the mask files
    vectorize_images(image_paths, mask_paths, output_dir, cache_dir, image_level=1, patch_size=128)
