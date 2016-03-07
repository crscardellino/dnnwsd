# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np

from sklearn.cross_validation import train_test_split


class DataSet(object):
    def __init__(self, data, target):
        assert data.shape[0] == target.shape[0]
        
        self._data = data
        self._target = target
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def data_count(self):
        return self._data.shape[0]

    @property
    def labels_count(self):
        return np.unique(self._target).shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def vector_length(self):
        return self._data.shape[1]

    @property
    def one_hot_labels(self):
        num_labels = self.data_count
        num_classes = self.labels_count
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + self._target.ravel()] = 1
        return labels_one_hot

    def next_batch(self, batch_size, one_hot=True):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self.data_count:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.data_count)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._target = self._target[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.data_count

        end = self._index_in_epoch

        if one_hot:
            return self._data[start:end], self.one_hot_labels[start:end]
        else:
            return self._data[start:end], self._target[start:end]


class SemiDataSet(object):
    def __init__(self, annotated_data, annotated_target, unannotated_data, unannotated_target):
        self.annotated_ds = DataSet(annotated_data, annotated_target)
        self.unannotated_ds = DataSet(unannotated_data, unannotated_target)
        # unannotated_target are the sentences ids

    @property
    def data_count(self):
        return self.annotated_ds.data_count + self.unannotated_ds.data_count

    def next_batch(self, batch_size, one_hot=True):
        annotated_data, target = self.annotated_ds.next_batch(min(batch_size, self.annotated_ds.data_count))
        unannotated_data, _ = self.unannotated_ds.next_batch(batch_size, one_hot=False)

        if hasattr(annotated_data, 'todense'):
            annotated_data = annotated_data.todense()
        if hasattr(unannotated_data, 'todense'):
            unannotated_data = unannotated_data.todense()

        data = np.vstack([annotated_data, unannotated_data])
        return data, target


class DataSets(object):
    def __init__(self, dataset_path, train_ratio=0.8, test_ratio=0.2, validation_ratio=0):
        assert train_ratio + test_ratio + validation_ratio == 1, "Train, Test and Validation ratio don't sum 1"

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        self._lemma = dataset['lemma']
        self._lemma_index = dataset['index']
        self._raw_unannotated_sentences = dataset['unannotated_sentences']

        annotated_data = dataset['annotated_dataset']['data']
        annotated_target = dataset['annotated_dataset']['target']
        unannotated_data = dataset['unannotated_dataset']['data']
        unannotated_target_sentences = dataset['unannotated_dataset']['sentences']

        if annotated_data.max() > 1 or unannotated_data.max() > 1:  # Normalize dataset
            max_value = max(annotated_data.max(), unannotated_data.max())

            annotated_data = np.multiply(annotated_data, 1.0 / max_value)
            unannotated_data = np.multiply(unannotated_data, 1.0 / max_value)

        tr_index, te_index, va_index = self.__split_data__(annotated_target, train_ratio, test_ratio, validation_ratio)

        self.train_ds = SemiDataSet(annotated_data[tr_index], annotated_target[tr_index],
                                    unannotated_data, unannotated_target_sentences)

        self.test_ds = DataSet(annotated_data[te_index], annotated_target[te_index])

        if va_index.shape[0] > 0:
            self.validation_ds = DataSet(annotated_data[va_index], annotated_target[va_index])
        else:
            self.validation_ds = None

    def __getitem__(self, item):
        return self._raw_unannotated_sentences[item]

    @property
    def lemma(self):
        return self._lemma

    @property
    def lemma_index(self):
        return self._lemma_index

    @staticmethod
    def __split_data__(target, train_ratio, test_ratio, validation_ratio):
        tr_set = set()
        te_set = set()
        va_set = set()

        init_tr_index = []
        init_te_index = []
        init_va_index = []

        permuted_indices = np.random.permutation(target.shape[0])

        # We make sure every split has at least one example of each class
        for target_index in permuted_indices:
            if target[target_index] not in tr_set:
                init_tr_index.append(target_index)
                tr_set.add(target[target_index])
            elif target[target_index] not in te_set:
                init_te_index.append(target_index)
                te_set.add(target[target_index])
            elif target[target_index] not in va_set and validation_ratio > 0:
                init_va_index.append(target_index)
                va_set.add(target[target_index])

        filtered_indices = permuted_indices[~np.in1d(
                permuted_indices, np.array(init_tr_index + init_te_index + init_va_index)
        )]

        # We randomly split the remaining examples
        tr_index, te_index = train_test_split(filtered_indices, train_size=train_ratio)
        split_index = np.ceil(te_index.shape[0] / 2).astype(np.int32)
        te_index, va_index = te_index[:split_index], te_index[split_index:]

        return (np.hstack([init_tr_index, tr_index]).astype(np.int32),
                np.hstack([init_te_index, te_index]).astype(np.int32),
                np.hstack([init_va_index, va_index]).astype(np.int32)
                )

