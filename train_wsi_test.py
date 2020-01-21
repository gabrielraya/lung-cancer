import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#from nic.gradcam_wsi import gradcam_on_dataset
#from Project.train_compressed_wsi import FeaturizedWsiGenerator, FeaturizedWsiSequence
#from digitalpathology.image.io import imagereader
import scipy
from nic.util_fns import cache_file
import glob
from os.path import exists, join, basename
import shutil
from nic.callbacks import ReduceLROnPlateau, ModelCheckpoint, HistoryCsv, FinishedFlag, PlotHistory, StoreModelSummary, CopyResultsExternally, LearningRateScheduler





class_0 = [load_image(f'dir0/{filename}') for filename in os.listdir('dir0')]
class_1 = ...
all_examples = [(x, 0) for x in class_0] + [(x, 1) for x in class_1]
#%%

def get_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')
    return image_ids, labels


def read_data(data_config, custom_augmentations=None):
    """
    Data reader
    Outputs:
        image_ids_all, features_path, distance_map_path, labels_all, features_ids_all
    """

    # Get params
    data_dir = data_config['data_dir']
    csv_path = data_config['csv_path']

    if custom_augmentations is None:
        pass
        #augmentations = [('none', 0), ('none', 90), ('none', 180), ('none', 270), ('horizontal', 0), ('vertical', 0), ('vertical', 90), ('vertical', 270)]
    else:
        augmentations = custom_augmentations

    # Read image file names
    df = pd.read_csv(csv_path)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')

    # Get paths
    image_ids_all = []
    features_path = []
    distance_map_path = []
    labels_all = []
    features_ids_all = []
    #batch_ids_all = []
    for i, image_id in enumerate(image_ids):
        if custom_augmentations is None:
            l = labels[i]
            if (l == 0):
                f_path = os.path.join(data_dir+'/tcga_lusc/normal', '{image_id}.npy'.format(image_id=image_id))
                dm_path = os.path.join(data_dir+'/tcga_lusc/normal', '{image_id}_distance_map.npy'.format(image_id=image_id))
                feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
            else:
                f_path = os.path.join(data_dir+'/tcga_luad/normal','{image_id}.npy'.format(image_id=image_id))
                dm_path = os.path.join(data_dir+'/tcga_luad/normal', '{image_id}_distance_map.npy'.format(image_id=image_id))
                feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
            image_ids_all.append(image_id)
            features_path.append(f_path)
            distance_map_path.append(dm_path)
            labels_all.append(l)
            features_ids_all.append(feature_id)
        else:
            for flip, rot in augmentations:
                l = labels[i]
                if (l == 0):
                    f_path = os.path.join(data_dir+'/tcga_lusc/augmented', '{image_id}_{rot}_{flip}_features.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
                    dm_path = os.path.join(data_dir+'/tcga_lusc/augmented', '{image_id}_{rot}_{flip}_features_distance_map.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
                    feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
                else:
                    f_path = os.path.join(data_dir+'/tcga_luad/augmented','{image_id}_{rot}_{flip}_features.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
                    dm_path = os.path.join(data_dir+'/tcga_luad/augmented', '{image_id}_{rot}_{flip}_features_distance_map.npy'.format(image_id=image_id, rot=int(rot),flip=str(flip)))
                    feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
                image_ids_all.append(image_id)
                features_path.append(f_path)
                distance_map_path.append(dm_path)
                labels_all.append(l)
                features_ids_all.append(feature_id)

    # Format
    # labels_all = np.eye(2)[labels_all].astype('uint8')

    # Shuffle
    idx = np.random.choice(len(image_ids_all), len(image_ids_all), replace=False)
    image_ids_all = [image_ids_all[i] for i in idx]
    features_path = [features_path[i] for i in idx]
    distance_map_path = [distance_map_path[i] for i in idx]
    features_ids_all = [features_ids_all[i] for i in idx]
    #batch_ids_all = [batch_ids_all[i] for i in idx]
    labels_all = np.array([labels_all[i] for i in idx]).astype('uint8')

    return image_ids_all, features_path, distance_map_path, labels_all, features_ids_all


if __name__ == '__main__':

    """
    TCGA DATASET
    531 in LUAD
    506 in LUSC
    """

    """ So far I am running this code not in the cluster """

    root_dir=  r'E:\pathology-weakly-supervised-lung-cancer-growth-pattern-prediction'
    data_dir =  root_dir+'/data/'
    csv_path =  root_dir+'/data/slide_list_tcga.csv'
    output_dir =   root_dir+'/results/'

    slides, labels = get_labels_from_csv(csv_path)

    data_config={'data_dir': data_dir, 'csv_path': csv_path}
    image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config, custom_augmentations=None)


#%%