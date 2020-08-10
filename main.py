# -*- coding: utf-8 -*-
# Author: Li Disen
# E-mail: lidisen@126.com
# Affiliation: Northeastern University (China, Shenyang)
# Tools: TensorFlow version 1.12.1

import model, utils
import numpy as np

""" Prepare data and label """

# Get the data and labels
np.random.seed(123)
X1 = np.random.normal(loc=0.0, scale=1.0, size=(100, 224, 224, 3))
X2 = np.zeros(shape=(100, 224, 224, 3))
X = np.concatenate((X1, X2), axis=0)

Y1 = np.ones(shape=(100,))
Y2 = np.zeros(shape=(100,))
Y = np.concatenate((Y1, Y2), axis=0)

# Split the data
num_data = X.shape[0]
train_idx, test_idx, val_idx = utils.split_idx(num_data, 0.8, 0.1)
train_data = X[train_idx, :, :, :]
val_data = X[val_idx, :, :, :]
test_data = X[test_idx, :, :, :]

train_labels = Y[train_idx]
val_labels = Y[val_idx]
test_labels = Y[test_idx]

""" Neural networks"""
params = {}
params['dir_name'] = 'project_name/'
params['num_epochs'] = 20
params['batch_size'] = 10
params['decay_steps'] = train_data.shape[0] / params['batch_size']
params['eval_frequency'] = int(len(train_idx) / params['batch_size'])
params['regularization'] = 5e-4
params['dropout'] = 0.5
params['learning_rate'] = 0.01  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
params['decay_rate'] = 0.95
params['momentum'] = 0.9
params['height'] = 224
params['width'] = 224
params['channel'] = 3

""" Run the model """
# Run the model
model = model.Model(**params)
accuracy, loss, t_step = model.fit(train_data, train_labels, val_data, val_labels)

# Evaluate the test data
print("Test accuracy is:")
res = model.evaluate(test_data, test_labels)
print(res[0])