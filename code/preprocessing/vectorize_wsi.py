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
print('Adding Neural Image Compression library to python path')
sys.path.append('/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/neural-image-compression-private/source')

import os
from utils import check_file_exists, get_file_list
from nic.vectorize_wsi import vectorize_wsi
from nic.util_fns import cache_file
from argparse import ArgumentParser

#%%
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
    """
    Example: 
    python3 /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/preprocessing/vectorize_wsi.py 
    --wsiFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUSC/wsi_diagnostic_tif" --maskFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUSC/tissue_masks_diagnostic"
    --outputDir="/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/results/tcga_lusc/vectorized"
    """
    ## Define Arguments
    parser = ArgumentParser(description='Vectorize whole slide images')
    parser.add_argument("--wsiFolder", help="WSI path to be vectorized", dest='wsiFolder')
    parser.add_argument("--maskFolder", help="WSI masks path", dest='maskFolder')
    parser.add_argument("--outputDir", help="output path", dest='outputDir')
    parser.add_argument("--cacheDir", help="Directory to temporary store the images, for fast processing", dest='cacheDir')
    parser.add_argument("--imageLevel", help="magnification level to read the image patches ..", type=float, dest='imageLevel')
    parser.add_argument("--patchSize", help="patch size", dest='patchSize')

    ## Parse Arguments
    args = parser.parse_args()
    image_path = args.wsiFolder
    mask_path = args.maskFolder
    output_dir = args.outputDir

    if args.cacheDir is None:
        print("No Cache Directory found")
        
    if args.imageLevel is None:
        print("No Image level found")
        args.imageLevel = 1

    if args.patchSize is None:
        print("No Patch Size found")
        args.patchSize = 128

    cache_dir = args.cacheDir
    image_level = args.imageLevel
    patch_size = args.patchSize

    print(f'\nVectorizing Whole Slide Images\nInput Directory: {image_path} \nMask directory:{mask_path} \nOutput Directory: {output_dir}\n'
    f'Image Level :{image_level}\nPatch Size : {patch_size}\n\n')
    vectorize_images(image_path, mask_path, output_dir, cache_dir, image_level, patch_size)
