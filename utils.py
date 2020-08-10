# -*- coding: utf-8 -*-
# Author: Li Disen
# E-mail: lidisen@126.com
# Affiliation: Northeastern University (China, Shenyang)
# Tools: TensorFlow version 1.12.1


import numpy as np

def split_idx(number, train_ratio, test_ratio):
    '''
    Split the idx to training ,val, and test mask

        number : the number of samples
        train_ratio : the train set ratio of all setting
        test_ratio : the test set ratio of all setting

        return : the idx of train , test and val
    '''
    number = int(number)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(number)
    train_set_size = int(number * train_ratio)
    test_set_size = int(number * test_ratio)

    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:train_set_size + test_set_size]
    val_indices = shuffled_indices[train_set_size + test_set_size:]

    return train_indices, val_indices, test_indices