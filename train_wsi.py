"""
Train a CNN on compressed whole-slide images.
"""

import os
#import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#from nic.gradcam_wsi import gradcam_on_dataset
#from Project.train_compressed_wsi import FeaturizedWsiGenerator, FeaturizedWsiSequence
#from digitalpathology.image.io import imagereader

"""
531 in LUAD
506 in LUSC
"""

def create_csv(data_dir, csv_path):
    """
    Output: csv file with slide names and corresponding labels, to be use for preprocessing
    """
    data_dir1 = data_dir + '/tcga_luad/normal'
    data_dir0 = data_dir + '/tcga_lusc/normal'

    image_files1 = sorted(
        [(os.path.basename(file)).split('.')[0] for file in os.listdir(data_dir1) if file.endswith('.png')])
    image_files0 = sorted(
        [(os.path.basename(file)).split('.')[0] for file in os.listdir(data_dir0) if file.endswith('.png')])
    labels1 = np.ones(len(image_files1), dtype=np.int8)
    labels0 = np.zeros(len(image_files0), dtype=np.int8)

    df1 = pd.DataFrame(list(zip(image_files1, labels1)), columns=['slide_id', 'label'])
    df0 = pd.DataFrame(list(zip(image_files0, labels0)), columns=['slide_id', 'label'])

    # conacatenate dataframes
    data = pd.concat([df1, df0], ignore_index=True, )
    export_csv = data.to_csv(csv_path, index=None, header=True)
    print('Csv file sucessfully exported!')

def get_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')

    return image_ids, labels



def read_data(data_config, custom_augmentations=None):

    # Get params
    data_dir = data_config['data_dir']
    csv_path = data_config['csv_path']

    if custom_augmentations is None:
        augmentations = [('none', 0), ('none', 90), ('none', 180), ('none', 270), ('horizontal', 0), ('vertical', 0), ('vertical', 90), ('vertical', 270)]
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
        for flip, rot in augmentations:
            f_path = os.path.join(data_dir, '{image_id}_{rot}_{flip}_features.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
            dm_path = os.path.join(data_dir, '{image_id}_{rot}_{flip}_features_distance_map.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
            l = labels[i]
            feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]

            image_ids_all.append(image_id)
            features_path.append(f_path)
            distance_map_path.append(dm_path)
            labels_all.append(l)
            features_ids_all.append(feature_id)
            #batch_ids_all.append(batch_ids[i])

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



#%%


#%%

root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
featurized_dir =  root_dir+'/results/tcga/featurized'
csv_path =  root_dir+'/data/tcga/slide_list_tcga.csv'
output_dir =   root_dir+'/results/tcga/model'


data_config={'data_dir': featurized_dir, 'csv_path': csv_path}

image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config, custom_augmentations=None)


#%%
if __name__ == '__main__':

""" So far I am running this code not in the cluster """

    root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
    featurized_dir =  root_dir+'/results/tcga/featurized'
    csv_path =  root_dir+'/data/tcga/slide_list_tcga.csv'
    output_dir =   root_dir+'/results/tcga/model'

    slides, labels = get_labels_from_csv(csv_path)

    image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config, custom_augmentations=None)
