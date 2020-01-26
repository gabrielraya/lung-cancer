import os
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import keras
import scipy

from nic.util_fns import cache_file


def read_data(data_config, custom_augmentations=None):
    """
    Data reader and shuffle
    Inputs:
        data_conf : dictionary with the data paths (featurize paths and csv file)
        augmentations = [('none', 0), ('none', 90), ('none', 180), ('none', 270), ('horizontal', 0), ('vertical', 0), ('vertical', 90), ('vertical', 270)]
    Outputs:
        image_ids_all, features_path, distance_map_path, labels_all, features_ids_all
    """

    # Set augmentations : Comment this line to run baseline without augmentations
    custom_augmentations = [('none', 0), ('none', 90), ('none', 180), ('none', 270), ('horizontal', 0), ('vertical', 0), ('vertical', 90), ('vertical', 270)]

    # Get params
    data_dir_class1 = data_config['data_dir_luad']
    data_dir_class0 = data_config['data_dir_lusc']
    csv_dir = data_config['csv_path']

    # Read image file names
    df = pd.read_csv(csv_dir)
    df = shuffle(df)
    image_ids = list(df['slide_id'].values)
    labels = df['label'].values.astype('uint8')

    set_dir = [data_dir_class0, data_dir_class1]

    # Get paths
    image_ids_all = []
    features_path = []
    distance_map_path = []
    labels_all = []
    features_ids_all = []

    if custom_augmentations is None:
        for i, image_id in enumerate(image_ids):
            label = labels[i]
            # if label 0 then dir data_dir_class0 if 1 data_dir_class1
            f_path = os.path.join(set_dir[label], '{image_id}.npy'.format(image_id=image_id))
            dm_path = os.path.join(set_dir[label], '{image_id}_distance_map.npy'.format(image_id=image_id))
            feature_id = os.path.splitext(os.path.basename(f_path))[0][:-9]
            image_ids_all.append(image_id)
            features_path.append(f_path)
            distance_map_path.append(dm_path)
            labels_all.append(label)
            features_ids_all.append(feature_id)
    else:
        for i, image_id in enumerate(image_ids):
            for flip, rot in custom_augmentations:
                label = labels[i]
                f_path = os.path.join(set_dir[label],'{image_id}_{rot}_{flip}_features.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
                dm_path = os.path.join(set_dir[label], '{image_id}_{rot}_{flip}_features_distance_map.npy'.format(image_id=image_id, rot=int(rot), flip=str(flip)))
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
                 keep_data=1.0, occlusion_augmentation=False, elastic_augmentation=False, shuffle_augmentation=None,
                 binary_target=True):

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
        self.binary_target = binary_target

        # Cache dir
        if self.cache_dir and not exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Read paths
        self.image_ids, self.paths, self.dm_paths, self.labels, self.feature_ids = data_fn(data_config)

        # Keep data (assume they are shuffled)
        n = int(len(self.image_ids) * keep_data)
        self.image_ids = self.image_ids[:n]
        self.paths = self.paths[:n]
        self.dm_paths = self.dm_paths[:n] if self.dm_paths is not None else None
        self.labels = self.labels[:n]
        self.feature_ids = self.feature_ids[:n]

        # Indexes for positive and negative samples
        if self.balanced and self.binary_target:
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
        print(
            'FeaturizedWsiGenerator using {n1} samples and {n2} batches, distributed in {n3} positive and {n4} negative samples.'.format(
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

            dx = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma,
                                                       mode='constant', cval=0) * alpha
            dy = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma,
                                                       mode='constant', cval=0) * alpha
            z, x, y = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), np.arange(image_shape[2]),
                                  indexing='ij')
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
                            input=x[i, :, :, j:j + 1], coordinates=indices, order=0,
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
                    x[i, x1:x1 + x2, y1:y1 + y2, :] = 0

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
        print(
            'FeaturizedWsiSequence using {n1} samples and {n2} batches, distributed in {n3} positive and {n4} negative samples.'.format(
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
                        features = crop_features(features, distance_map, crop_size=self.crop_size, deterministic=True,
                                                 crop_id=self.crop_ids[idx], n_crops=self.n_crops)
                    else:
                        features = crop_features(features, distance_map, crop_size=self.crop_size, deterministic=True)

                # Get ids
                ids.append(self.feature_ids[idx])

                # Append
                x.append(features)

                # Label
                y.append(self.labels[idx])

            except Exception as e:
                print(
                    'FeaturizedWsiSequence failed to assemble batch with idx {idx}, skipping sample. Exception: {e}'.format(
                        idx=idx, e=e), flush=True)

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
