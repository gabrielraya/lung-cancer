"""
Train a CNN on compressed whole-slide images.

    class 1 : luad
    class 0 : lusc
"""

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
"""
531 in LUAD
506 in LUSC
"""

def create_csv(data_dir, csv_path):
    """
    Output: csv file with slide names and corresponding labels, to be use for preprocessing
        labels 1 correspond to LUAD
        labesl 0 correspond to LUSC
    """
    data_dir1 = data_dir + '/tcga_luad/augmented'
    data_dir0 = data_dir + '/tcga_lusc/augmented'

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
    data.to_csv(csv_path, index=None, header=True)
    print('Csv file sucessfully exported!')

def get_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')

    return image_ids, labels


#%%


root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
data_dir_luad =  root_dir + r'\results\tcga\featurized\tcga_luad\normal'
data_dir_lusc =  root_dir + r'\results\tcga\featurized\tcga_lusc\normal'
csv_path      =  root_dir+'/data/tcga/slide_list_tcga.csv'
output_dir    =   root_dir+'/results/tcga/model'
root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
csv_path      =  root_dir+'/data/tcga/slide_list_tcga.csv'
csv_train     =  root_dir+'/data/tcga/train_slide_list_tcga.csv'
csv_val       =  root_dir+'/data/tcga/validation_slide_list_tcga.csv'
csv_test      =  root_dir+'/data/tcga/test_slide_list_tcga.csv'


from sklearn.model_selection import train_test_split

def generate_csv_files(paths, split=0.2):
    """
    Generates csv files given a csv file
    """
    csv_dir= paths['original']
    csv_train_path = paths['split1']
    csv_test_path = paths['split2']
    df = pd.read_csv(csv_dir)
    X = df['slide_id']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)
    df = pd.DataFrame(pd.concat([X_train, y_train], axis=1))
    df.to_csv(csv_train_path, index=None, header=True)
    df = pd.DataFrame(pd.concat([X_test, y_test], axis=1))
    df.to_csv(csv_test_path, index=None, header=True)
    print('Csvs file sucessfully exported!')

def split_train_tes_val(paths, test_size=0.2, validation_size = 0.3):
    """
    Split data set into training, validation and test sets
    """
    csv_path = paths['csv_path']
    csv_train = paths['csv_train']
    csv_valid = paths['csv_val']
    csv_test = paths['csv_test']
    csv_paths = {'original': csv_path, 'split1': csv_train, 'split2': csv_test}
    generate_csv_files(csv_paths, split=test_size)
    csv_paths = {'original': csv_train, 'split1': csv_train, 'split2':csv_val}
    generate_csv_files(csv_paths, split=validation_size)

paths = {'csv_path': csv_path, 'csv_train': csv_train, 'csv_val': csv_val, 'csv_test':csv_test}
split_train_tes_val(paths, test_size=0.2, validation_size = 0.3)
#%%
df = pd.read_csv(csv_train)
df2 = pd.read_csv(csv_val)
df3 = pd.read_csv(csv_train)
n = df.shape

df = pd.read_csv(csv_train)
df.shape


#%%
#data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path}

#image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config)


def read_data(data_conf):
    """
    Data reader
    Inputs:
        data_conf : dictionary with the data paths (featurize paths and csv file)
    Outputs:
        image_ids_all, features_path, distance_map_path, labels_all, features_ids_all
    """

    # Get params
    data_dir_class0 = data_conf['data_dir_lusc']
    data_dir_class1 = data_conf['data_dir_luad']
    csv_dir = data_conf['csv_path']

    # Read image file names
    df = pd.read_csv(csv_dir)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')

    # Get paths
    image_ids_all=[]; features_path=[]; distance_map_path=[]; labels_all=[]; features_ids_all=[]

    for i, image_id in enumerate(image_ids):
            label = labels[i]
            if label == 0:
                f_path = os.path.join(data_dir_class0, '{image_id}.npy'.format(image_id=image_id))
                dm_path = os.path.join(data_dir_class0, '{image_id}_distance_map.npy'.format(image_id=image_id))
                feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
            else:
                f_path = os.path.join(data_dir_class1, '{image_id}.npy'.format(image_id=image_id))
                dm_path = os.path.join(data_dir_class1, '{image_id}_distance_map.npy'.format(image_id=image_id))
                feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
            image_ids_all.append(image_id)
            features_path.append(f_path)
            distance_map_path.append(dm_path)
            labels_all.append(label)
            features_ids_all.append(feature_id)

    # Shuffle
    idx = np.random.choice(len(image_ids_all), len(image_ids_all), replace=False)
    image_ids_all = [image_ids_all[i] for i in idx]
    features_path = [features_path[i] for i in idx]
    distance_map_path = [distance_map_path[i] for i in idx]
    features_ids_all = [features_ids_all[i] for i in idx]
    labels_all = np.array([labels_all[i] for i in idx]).astype('uint8')

    return image_ids_all, features_path, distance_map_path, labels_all, features_ids_all


def crop_features(features, distance_map, crop_size, deterministic=False, crop_id=None, n_crops=None):

    # Sample center
    if deterministic:
        if (crop_id is not None) and (n_crops is not None):
            distance_map_idxs = np.where(distance_map.flatten() != 0)[0]
            center = distance_map_idxs[int(len(distance_map_idxs) * (crop_id / n_crops))]
        else:
            center = np.argmax(distance_map.flatten())
        center = np.unravel_index(center, distance_map.shape)
        x_center, y_center = center
    else:
        center = np.random.choice(len(distance_map.flatten()), 1, replace=True, p=distance_map.flatten())
        center = np.unravel_index(center, distance_map.shape)
        x_center, y_center = (center[0][0], center[1][0])

    # Crop params
    x_size = features.shape[0]
    y_size = features.shape[1]
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
    features = np.pad(features, padding, 'constant')
    x1 += padx1
    x2 += padx1
    y1 += pady1
    y2 += pady1

    # Crop
    features = features[x1:x2, y1:y2, :]

    return features


class FeaturizedWsiGenerator(object):

    def __init__(self, data_config, data_fn, batch_size, augment, crop_size, cache_dir=None, balanced=True,
                 keep_data=1.0, occlusion_augmentation=False, elastic_augmentation=False, shuffle_augmentation=None):
        """
            data_config : data configuration dictionary
            data_fn : read data function
        """
        # Params
        self.batch_size = batch_size
        self.data_config = data_config
        self.augment = augment
        self.crop_size = crop_size
        self.cache_dir = cache_dir
        self.balanced = balanced
        self.keep_data = keep_data
        self.occlusion_augmentation = occlusion_augmentation
        self.elastic_augmentation = elastic_augmentation
        self.shuffle_augmentation = shuffle_augmentation

        # Cache dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Read paths : image file names, featurize npy paths, distance map paths, ...
        self.image_ids, self.paths, self.dm_paths, self.labels, self.feature_ids = data_fn(data_config)

        # Keep data (assume they are shuffled)
        n = int(len(self.image_ids) * keep_data)
        self.image_ids = self.image_ids[:n]
        self.paths = self.paths[:n]
        self.dm_paths = self.dm_paths[:n]
        self.labels = self.labels[:n]
        self.feature_ids = self.feature_ids[:n]

        # Indexes for positive and negative samples
        if self.balanced:
            self.pos_idx = np.where(self.labels == 1)[0]
            self.neg_idx = np.where(self.labels == 0)[0]
        else:
            self.pos_idx = np.arange(len(self.paths))
            self.neg_idx = np.arange(len(self.paths))

        # Other
        self.n_samples = len(self.paths)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

        # Print
        print('FeaturizedWsiGenerator data config: ' + str(data_config), flush=True)
        print('FeaturizedWsiGenerator using {n1} samples and {n2} batches, distributed in {n3} positive and {n4} negative samples.'.format(
                n1=self.n_samples, n2=self.n_batches, n3=len(self.pos_idx), n4=len(self.neg_idx)
        ), flush=True)

        # Elastic
        if self.elastic_augmentation:
            self.n_maps = 50
            self.deformation_maps = self.create_deformation_maps()
        else:
            self.n_maps = None
            self.deformation_maps = None

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def create_deformation_maps(self):

        alpha_interval = (10000, 50000)  # (300, 1200) # TODO test higher values!
        sigma_interval = (20.0, 20.0)
        image_shape = (self.crop_size, self.crop_size, 1)
        deformation_maps = []

        for _ in range(self.n_maps):
            alpha = np.random.uniform(low=alpha_interval[0], high=alpha_interval[1], size=None)
            sigma = np.random.uniform(low=sigma_interval[0], high=sigma_interval[1], size=None)

            dx = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma, mode='constant', cval=0) * alpha
            dy = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma, mode='constant', cval=0) * alpha
            z, x, y = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), np.arange(image_shape[2]), indexing='ij')
            indices = (np.reshape(z, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)))

            deformation_maps.append(indices)

        return deformation_maps

    def augment_batch(self, x, y):
        """
        Randomly applies 90-degree rotation and horizontal-vertical flipping (same augmentation for the entire batch).

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        # Flip
        x = np.flip(x, np.random.randint(2) + 1)

        # Rot
        x = np.rot90(x, np.random.randint(4), axes=(1, 2))

        # Elastic
        if self.elastic_augmentation:

            # Per sample
            for i in range(len(x)):
                if np.random.rand() > 0.25:

                    indices = self.deformation_maps[np.random.randint(0, self.n_maps)]

                    # Per channel
                    for j in range(x.shape[-1]):
                        x[i, :, :, j] = scipy.ndimage.interpolation.map_coordinates(
                                input=x[i, :, :, j:j+1], coordinates=indices, order=0,
                                mode='reflect').reshape(x[i, :, :, j].shape)

        # Shuffle crop augmentation
        if self.shuffle_augmentation is not None:

            # from featurize_wsi import plot_feature_map

            # Per sample
            labels = np.argmax(y, axis=-1)
            x_source = np.copy(x)
            for i in range(len(x)):
                # plot_feature_map(np.copy(x[i].transpose((2, 0, 1))), r'W:\projects\pathology-liver-survival\debug\shuffle_augmentation\{i}_before.png'.format(i=i)) # todo

                # Find target
                idxs = np.random.choice(np.where(labels == labels[i])[0], self.shuffle_augmentation)
                for idx in idxs:

                    # Crop coordinates
                    x1 = int(np.random.uniform(0, self.crop_size))
                    y1 = int(np.random.uniform(0, self.crop_size))

                    # Paste
                    x[i, x1:, y1:, :] = x_source[idx, x1:, y1:, :]

                    # Rotate
                    x[i, ...] = np.rot90(x[i, ...], np.random.randint(4), axes=(0, 1))

                # plot_feature_map(np.copy(x[i].transpose((2, 0, 1))), r'W:\projects\pathology-liver-survival\debug\shuffle_augmentation\{i}_after.png'.format(i=i))

        # Occlusion
        if self.occlusion_augmentation:

            # Per sample
            for i in range(len(x)):
                if np.random.rand() > 0.25:
                    x1 = int(np.random.uniform(0, self.crop_size // 2))
                    y1 = int(np.random.uniform(0, self.crop_size // 2))
                    x2 = int(np.random.uniform(self.crop_size // 2, self.crop_size))
                    y2 = int(np.random.uniform(self.crop_size // 2, self.crop_size))
                    x[i, x1:x1+x2, y1:y1+y2, :] = 0

        return x

    def assemble_batch(self, idxs):
        """
        Creates a training batch from featurized WSIs on disk. It samples a crop taking distance to background into
        account (crops centered on lots of tissue are more likely). It pads the crops if needed. It copies the files
        to cache if needed.

        Args:
            idxs: file indexes to process.

        Returns: tuple of batch and labels.

        """

        x = []
        y = []
        for idx in idxs:

            try:

                # Get features
                if self.cache_dir:
                    self.paths[idx] = cache_file(self.paths[idx], self.cache_dir, overwrite=False)
                features = np.load(self.paths[idx]).astype('float32').transpose((1, 2, 0))

                # Get distance map
                if self.dm_paths is not None:
                    self.dm_paths[idx] = cache_file(self.dm_paths[idx], self.cache_dir, overwrite=False)
                    distance_map = np.load(self.dm_paths[idx])

                    # Crop
                    features = crop_features(features, distance_map, crop_size=self.crop_size, deterministic=False)

                # Append
                x.append(features)

                # Label
                y.append(self.labels[idx])

                # print('File {f} with label {l}'.format(f=basename(self.paths[idx]), l=self.labels[idx]), flush=True)

            except Exception as e:
                print(
                    'FeaturizedWsiGenerator failed to assemble batch with idx {idx}, skipping sample. Exception: {e}'.format(
                        idx=idx, e=e), flush=True)

        # Fill
        if len(x) < len(idxs):
            print('Filling batch to match batch size...', flush=True)
            fill_idxs = np.random.choice(len(x), len(idxs) - len(x), replace=False).astype('uint8')
            for fill_idx in fill_idxs:
                x.append(x[fill_idx])
                y.append(y[fill_idx])

        # Concat
        x = np.stack(x, axis=0)
        if self.binary_target:
            y = np.eye(2)[np.array(y, dtype='int')]
        else:
            y = np.array(y)

        return x, y

    def next(self):
        """
        Builds the training batch.

        Returns: tuple of batch and labels.
        """

        # Get samples
        idxs_pos = np.random.choice(self.pos_idx, self.batch_size // 2, replace=True)
        idxs_neg = np.random.choice(self.neg_idx, self.batch_size // 2, replace=True)

        # Merge
        idxs = np.concatenate([idxs_pos, idxs_neg])

        # Randomize
        r = np.random.choice(len(idxs), len(idxs), replace=False)
        idxs = idxs[r]

        # Build batch
        x, y = self.assemble_batch(idxs)

        # Augment
        if self.augment:
            x = self.augment_batch(x, y)

        return x, y


root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
data_dir_luad =  root_dir + r'\results\tcga\featurized\tcga_luad\normal'
data_dir_lusc =  root_dir + r'\results\tcga\featurized\tcga_lusc\normal'
csv_path      =  root_dir+'/data/tcga/slide_list_tcga.csv'
output_dir    =   root_dir+'/results/tcga/model'

data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path}

image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config)


print('Loading training set ...', flush=True)
training_gen = FeaturizedWsiGenerator(
    data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path},
    data_fn=read_data,
    batch_size=32,
    augment=False,
    crop_size=400,
    cache_dir=None,
    balanced=True,
    keep_data=1,
    occlusion_augmentation=None,
    elastic_augmentation=None,
    shuffle_augmentation=None
)

class FeaturizedWsiSequence(keras.utils.Sequence):
    """
    Class to randomly provide batches of featurized WSIs loaded from numpy arrays on disk.
    """

    def __init__(self, data_config, data_fn, batch_size, crop_size, balanced, cache_dir=None, keep_data=1.0,
                 return_ids=False, binary_target=True, n_crops=None):

        # Params
        self.batch_size = batch_size
        self.data_config = data_config
        self.crop_size = crop_size
        self.cache_dir = cache_dir
        self.keep_data = keep_data
        self.balanced = balanced
        self.return_ids = return_ids
        self.binary_target = binary_target
        self.n_crops = n_crops

        # Cache dir
        if self.cache_dir and not exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Read paths
        self.image_ids, self.paths, self.dm_paths, self.labels, self.feature_ids = data_fn(data_config)

        # Keep data (assume they are shuffled)
        n = int(np.ceil(len(self.image_ids) * keep_data))
        self.image_ids = self.image_ids[:n]
        self.paths = self.paths[:n]
        self.dm_paths = self.dm_paths[:n] if self.dm_paths is not None else None
        self.labels = self.labels[:n]
        self.feature_ids = self.feature_ids[:n]

        # N crops
        if self.n_crops is not None:

            # Extend set
            def extend(l, n):
                nl = []
                for i in l:
                    for j in range(n):
                        nl.append(i)
                return nl

            self.crop_ids = np.concatenate([np.arange(n_crops) for _ in self.image_ids])
            self.image_ids = extend(self.image_ids, n_crops)
            self.paths = extend(self.paths, n_crops)
            self.dm_paths = extend(self.dm_paths, n_crops)
            self.labels = extend(self.labels, n_crops)
            self.feature_ids = extend(self.feature_ids, n_crops)
        else:
            self.crop_ids = None

        # Indexes for positive and negative samples
        if self.balanced and self.binary_target and self.n_crops is None:
            self.pos_idx = np.where(self.labels == 1)[0]
            self.neg_idx = np.where(self.labels == 0)[0]
        else:
            self.pos_idx = np.arange(len(self.paths))
            self.neg_idx = np.arange(len(self.paths))

        # Other
        if self.balanced and self.binary_target and self.n_crops is None:
            self.n_batches = int(np.ceil(np.max([len(self.pos_idx), len(self.neg_idx)]) * 2 / batch_size))
        else:
            self.n_batches = int(np.ceil(len(self.labels) / batch_size))

        # Print
        print('FeaturizedWsiSequence data config: ' + str(data_config), flush=True)
        print('FeaturizedWsiSequence using {n1} samples and {n2} batches, distributed in {n3} positive and {n4} negative samples.'.format(
                n1=len(self.image_ids), n2=self.n_batches, n3=len(self.pos_idx), n4=len(self.neg_idx)
        ), flush=True)

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def assemble_batch(self, idxs):
        """
        Creates a batch from featurized WSIs on disk. It samples a crop taking the center with the maximum distance to
        background. It pads the crops if needed. It copies the files to cache if needed.

        Args:
            idxs: file indexes to process.

        Returns: tuple of batch and labels.

        """

        x = []
        y = []
        ids = []
        for idx in idxs:

            try:

                # Get features
                if self.cache_dir:
                    self.paths[idx] = cache_file(self.paths[idx], self.cache_dir, overwrite=False)
                features = np.load(self.paths[idx]).astype('float32').transpose((1, 2, 0))

                # Get distance map
                if self.dm_paths is not None:
                    self.dm_paths[idx] = cache_file(self.dm_paths[idx], self.cache_dir, overwrite=False)
                    distance_map = np.load(self.dm_paths[idx])

                    # Crop
                    if self.n_crops is not None:
                        features = crop_features(features, distance_map, crop_size=self.crop_size, deterministic=True, crop_id=self.crop_ids[idx], n_crops=self.n_crops)
                    else:
                        features = crop_features(features, distance_map, crop_size=self.crop_size, deterministic=True)

                # Get ids
                ids.append(self.feature_ids[idx])

                # Append
                x.append(features)

                # Label
                y.append(self.labels[idx])

            except Exception as e:
                print('FeaturizedWsiSequence failed to assemble batch with idx {idx}, skipping sample. Exception: {e}'.format(idx=idx, e=e), flush=True)

        # Fill
        if len(x) < len(idxs):
            print('Filling batch to match batch size...', flush=True)
            fill_idxs = np.zeros(len(idxs) - len(x), dtype='uint8')  # fill with first sample
            for fill_idx in fill_idxs:
                x.append(x[fill_idx])
                y.append(y[fill_idx])
                ids.append(ids[fill_idx])

        # Concat
        x = np.stack(x, axis=0)
        y = np.array(y).astype('float')
        y_na = np.copy(y)
        y[np.isnan(y)] = 0
        if self.binary_target:
            y = np.eye(2)[np.array(y, dtype='int')]
        y[np.isnan(y_na), ...] = np.nan

        return x, y, ids

    def get_idxs(self, idx):

        if self.balanced and self.binary_target and self.n_crops is None:
            # Get positive samples
            idx_batch_pos = idx * self.batch_size // 2
            idxs_pos = np.mod(np.arange(idx_batch_pos, idx_batch_pos + self.batch_size // 2), len(self.pos_idx))
            idxs_pos = self.pos_idx[idxs_pos]

            # Get negative samples
            idx_batch_neg = idx * self.batch_size // 2
            idxs_neg = np.mod(np.arange(idx_batch_neg, idx_batch_neg + self.batch_size // 2), len(self.neg_idx))
            idxs_neg = self.neg_idx[idxs_neg]

            idxs = np.concatenate([idxs_pos, idxs_neg])

        else:
            # Get samples
            idx_batch = idx * self.batch_size
            if idx_batch + self.batch_size >= len(self.labels):
                idxs = np.arange(idx_batch, len(self.labels))
            else:
                idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        return idxs

    def __getitem__(self, idx):
        """
        Builds the batch (balanced if needed).

        Returns: tuple of batch and labels.
        """

        # Find idxs
        idxs = self.get_idxs(idx)

        # Build batch
        x, y, ids = self.assemble_batch(idxs)

        if self.return_ids:
            return x, y, ids
        else:
            return x, y


def build_wsi_classifier(input_shape, lr, output_units):
    """
    Builds a neural network that performs classification on featurized WSIs.

    Args:
        input_shape: shape of features with channels last, for example (400, 400, 128).
        lr (float): learning rate.

    Returns: compiled Keras model.

    """

    def conv_op(x, stride, dropout=0.2):

        # Conv
        l2_reg = keras.regularizers.l2(1e-5)
        x = keras.layers.SeparableConv2D(
            filters=128, kernel_size=3, strides=stride, padding='valid', depth_multiplier=1,
            activation='linear', depthwise_regularizer=l2_reg, pointwise_regularizer=l2_reg,
            bias_regularizer=l2_reg, kernel_initializer='he_uniform'
        )(x)

        # Batch norm
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        # Activation
        x = keras.layers.LeakyReLU()(x)

        # Dropout
        if dropout is not None:
            x = keras.layers.SpatialDropout2D(dropout)(x)

        return x

    def dense_op(x, n_units, bn, activation, l2_factor):

        # Regularization
        if l2_factor is not None:
            l2_reg = keras.regularizers.l2(l2_factor)
        else:
            l2_reg = None

        # Op
        x = keras.layers.Dense(units=n_units, activation='linear', kernel_regularizer=l2_reg,
                         bias_regularizer=l2_reg, kernel_initializer='he_uniform')(x)

        # Batch norm
        if bn:
            x = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        # Activation
        if activation == 'lrelu':
            x = keras.layers.LeakyReLU()(x)
        else:
            x = keras.layers.Activation(activation)(x)

        return x

    # Define classifier
    input_x = keras.layers.Input(input_shape)
    x = conv_op(input_x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=1)
    x = conv_op(x, stride=1)
    x = keras.layers.Flatten()(x)
    x = dense_op(x, n_units=128, bn=True, activation='lrelu', l2_factor=1e-5)
    x = dense_op(x, n_units=output_units, bn=False, activation='softmax', l2_factor=None)

    # Compile
    model = keras.models.Model(inputs=input_x, outputs=x)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy]
    )

    # print('Classifier model:', flush=True)
    # model.summary()

    return model


root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
data_dir_luad =  root_dir + r'\results\tcga\featurized\tcga_luad\normal'
data_dir_lusc =  root_dir + r'\results\tcga\featurized\tcga_lusc\normal'
csv_path      =  root_dir+'/data/tcga/slide_list_tcga.csv'
output_dir    =   root_dir+'/results/tcga/model'

# Training set
batch_size =32
cache_dir = None
keep_data_training=1
occlusion_augmentation=False; elastic_augmentation=False; shuffle_augmentation=None;

data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path}

image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config)




print('Loading training set ...', flush=True)
training_gen = FeaturizedWsiGenerator(
    data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path},
    data_fn=read_data,
    batch_size=batch_size,
    augment=True,
    crop_size=400,
    cache_dir=cache_dir,
    balanced=True,
    keep_data=keep_data_training,
    occlusion_augmentation=occlusion_augmentation,
    elastic_augmentation=elastic_augmentation,
    shuffle_augmentation=shuffle_augmentation
)

#%%
crop_size=400
code_size=128
lr=1e-2
output_units=1


keep_data_validation=0.60
#from nic.train_compressed_wsi import FeaturizedWsiSequence

# Validation set
print('Loading validation set ...', flush=True)
use_validation = True
validation_gen = FeaturizedWsiSequence(
    data_config={'data_dir_luad': data_dir_luad, 'data_dir_lusc': data_dir_lusc, 'csv_path': csv_path},
    data_fn=read_data,
    batch_size=batch_size,
    crop_size=400,
    cache_dir=cache_dir,
    balanced=True,
    keep_data=keep_data_validation
)if use_validation else None

#%%
# Create model
print('Building model ...', flush=True)
model=None
if model is None:
    model = build_wsi_classifier(input_shape=(crop_size, crop_size, code_size), lr=lr, output_units=output_units)

#%%
model.summary()


#%%
crop_size=400
code_size=128
lr=1e-2
output_units=1
n_epochs=2
batch_size=32
lr=1e-2
code_size=128
workers=1
train_step_multiplier=1
val_step_multiplier=0.5
keep_data_training=1
keep_data_validation=1
patience=4
occlusion_augmentation=occlusion_augmentation
elastic_augmentation=elastic_augmentation


def fit_model(training_generator, validation_generator, output_dir, model, n_epochs, train_step_multiplier, workers,
              patience, custom_objects=None, monitor='val_loss', mode='min', loss_list=['loss', 'val_loss'],
              metric_list=['categorical_accuracy', 'val_categorical_accuracy'], val_step_multiplier=1.0, min_lr=1e-4,
              extra_callbacks=[], cache_dir=None, lr_scheduler_fn=None):

    # Cache output
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    # Ignore if training finished
    if not os.path.exists(os.path.join(output_dir, 'training_finished.txt')):

        # Prepare directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Training model in directory: {d} with content {c}'.format(
            d=output_dir,
            c=os.system("ls " + output_dir)
        ), flush=True)

        # Continue training if model found
        epochs_run = 0
        if os.path.exists(os.path.join(output_dir, 'last_epoch.h5')) and exists(join(output_dir, 'history.csv')):
            print('Resuming training from saved model ...', flush=True)
            model = keras.models.load_model(os.path.join(output_dir, 'last_epoch.h5'), custom_objects=custom_objects)
            df = pd.read_csv(join(output_dir, 'history.csv'), header=0, index_col=0)
            epochs_run = len(df)

            # Copy existing files into cache
            if cache_dir is not None and os.path.exists(cache_dir):
                for path in glob(os.path.join(output_dir, '*')):
                    try:
                        shutil.copyfile(path, join(cache_dir, basename(path)))
                    except Exception as e:
                        print('Error copying file {f} from external {output_dir} to cache {cache_dir} directory. Exception: {e}'.format(
                            f=path, output_dir=output_dir, cache_dir=cache_dir, e=e
                        ), flush=True)
        else:
            print('Training model from scratch {b1} {b2}...'.format(
                b1=exists(join(output_dir, 'last_epoch.h5')),
                b2=exists(join(output_dir, 'history.csv'))
            ), flush=True)

        if epochs_run < n_epochs:

            if cache_dir is not None and exists(cache_dir):
                external_output_dir = output_dir
                output_dir = cache_dir
            else:
                external_output_dir = None

            # Define callbacks
            callback_list = [
                StoreModelSummary(filepath=join(output_dir, 'model_summary.txt'), verbose=1),
                HistoryCsv(file_path=join(output_dir, 'history.csv'))
            ]

            if len(extra_callbacks) > 0:
                callback_list.extend(extra_callbacks)

            callback_list2 = [
                ModelCheckpoint(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'checkpoint.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=1,
                    save_best_only=True
                ),
                ModelCheckpoint(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'last_epoch.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=1,
                    save_best_only=False
                ),
                PlotHistory(
                    plot_path=join(output_dir, 'history.png'),
                    log_path=join(output_dir, 'history.csv'),
                    loss_list=loss_list,
                    metric_list=metric_list
                ),
                FinishedFlag(
                    file_path=join(output_dir, 'training_finished.txt')
                )
            ]
            callback_list.extend(callback_list2)

            if patience is not None:
                callback_list.append(
                    ReduceLROnPlateau(
                        history_path=join(output_dir, 'history.csv'),
                        monitor=monitor,
                        mode=mode,
                        factor=1.0/3,
                        patience=patience,
                        verbose=1,
                        cooldown=2,
                        min_lr=min_lr
                    ) if lr_scheduler_fn is None else LearningRateScheduler(schedule=lr_scheduler_fn, min_lr=min_lr)
                )

            if external_output_dir is not None:

                callback_list.append(
                    CopyResultsExternally(
                        local_dir=output_dir,
                        external_dir=external_output_dir
                    )
                )

            # Train model
            model.fit_generator(
                generator=training_generator,
                steps_per_epoch=int(len(training_generator) * train_step_multiplier),
                epochs=n_epochs,
                verbose=1,
                callbacks=callback_list,
                validation_data=validation_generator,
                validation_steps=int(len(validation_generator) * val_step_multiplier) if validation_generator is not None else None,
                initial_epoch=epochs_run,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=True if workers > 1 else False
            )

            # Finish
            try:
                open(os.path.join(external_output_dir if external_output_dir is not None else output_dir, 'training_finished.txt'), 'a').close()
            except:
                pass
#%%

# Train initial model

loss_list=['loss', 'val_loss']
metric_list=['categorical_accuracy', 'val_categorical_accuracy']
custom_objects=None
lr_scheduler_fn =None
min_lr=1e-4


print('Training model ...', flush=True)
fit_model(
    training_generator=training_gen,
    validation_generator=validation_gen,
    output_dir=output_dir,
    model=model,
    n_epochs=n_epochs,
    train_step_multiplier=train_step_multiplier,
    val_step_multiplier=val_step_multiplier,
    workers=workers,
    patience=patience,
    monitor='val_loss' if use_validation else 'loss',
    mode='min',
    loss_list=loss_list,
    metric_list=metric_list,
    custom_objects=custom_objects,
    cache_dir=None if cache_dir is None else join(cache_dir, 'models', basename(output_dir)),
    lr_scheduler_fn=lr_scheduler_fn,
    min_lr=min_lr
)
#%%
if __name__ == '__main__':

""" So far I am running this code not in the cluster """

    root_dir=  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction'
    featurized_dir =  root_dir+'/results/tcga/featurized'
    csv_path =  root_dir+'/data/tcga/slide_list_tcga.csv'
    output_dir =   root_dir+'/results/tcga/model'

    slides, labels = get_labels_from_csv(csv_path)

    image_ids_all, features_path, distance_map_path, labels_all, features_ids_all = read_data(data_config, custom_augmentations=None)
