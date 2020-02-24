import multiresolutionimageinterface as mri
import matplotlib.pyplot as plt
from os.path import join
from sklearn.utils import shuffle
import numpy as np


def get_pixel_lenghts(image_path):
    """
    Reads a complete WSI and returns a the width and height

    returns: image numpy array size
    x_dim : width
    y_dim : height
    """
    # Open wsi image
    mr_image = mri.MultiResolutionImageReader().open(image_path)
    # Get Dimensions
    x_dim, y_dim = mr_image.getDimensions()
    return x_dim, y_dim


def readImage(image_path, level=4):
    """
    Reads a complete WSI and returns a numpy array
    In a WSI, at each higher resolution level the size is decreased by half.
    For this reason the dimensions should be decreased by 2^level for a given image resolution level

    returns: image numpy array
    """
    # Open wsi image
    mr_image = mri.MultiResolutionImageReader().open(image_path)
    # Get Dimensions
    x_dim, y_dim = mr_image.getDimensions()
    # Scale image to be display
    img = mr_image.getUCharPatch(startX=0, startY=0, width=x_dim // 2 ** level, height=y_dim // 2 ** level, level=level)

    return img, x_dim, y_dim


def plot_wsi(filename_path, title, t):
    """ Plots a single tif file with corresponding mask at level 4
    """

    # single wsi tif file
    fig, ax = plt.subplots(1, figsize=t)
    image_path = filename_path + '.tif'
    img, N, M = readImage(image_path, level=4)

    plt.imshow(img)
    plt.title(f'{title} WSI: ({M}, {N}, 3)')
    plt.axis('off')
    plt.show()


def plot_wsi_labels(preview, data_dir, cancer_type, start):
    fig, axs = plt.subplots(1,4,figsize=(18,20))
    axs = axs.ravel()
    for i in range(4):
        filename = preview.__getitem__(i+start)
        image_path= join(data_dir, filename +'.tif')
        img, N, M = readImage(image_path, level=4)
        axs[i].imshow(img)
        axs[i].set_title(cancer_type)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)


def plot_type(data, dir_wsi, cancer_type = 1, title = 'LUAD samples' ):
    """

    :param data: data frame containing all the slide_ids
    :param dir_wsi:
    :param cancer_type:
    :param title:
    :return:
    """
    columns = 4
    fig, ax = plt.subplots(1,columns, figsize=(20,20))
    shuffled_data = shuffle(data)

    for i, filename in enumerate(shuffled_data[shuffled_data['label'] == cancer_type]['slide_id'][:columns]):
        image_path= join(dir_wsi, filename +'.tif')
        img, N, M = readImage(image_path, level=4)
        ax[i].imshow(img)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_ticks([])
        ax[0].set_ylabel(title, size='large')


def plot_masks(dir_wsi, dir_wsi_mask, filename):
    """ Plot tif file with corresponding mask at level 4 and 0 correspondingly
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    ax0, ax1, = axes.flatten()

    # single wsi tif file
    image_path = join(dir_wsi, filename + '.tif')
    img, N, M = readImage(image_path, level=4)

    # read the compressed version
    mask_path = join(dir_wsi_mask, filename + '_tissue.tif')
    img_mask, N2, M2 = readImage(mask_path, level=0)

    ax0.imshow(img)
    ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax0.set(title=f'Wsi tif shape: ({M}, {N}, 3)')
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)

    ax1.imshow(np.squeeze(img_mask), cmap='gray')  ## remove single dimension entry
    ax1.set(title=f'Wsi mask shape: ({M2}, {N2})')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.show()