import pytest
from tests import _PATH_DATA
import torch
import os


def test_data():
    
    processed_dir = os.path.join(_PATH_DATA, 'processed')
    train_tensor_file = 'train_tensor.pt'
    train_labels_file = 'train_labels.pt'
    test_tensor_file = 'test_tensor.pt'
    test_labels_file = 'test_labels.pt'

    train_data = torch.load(os.path.join(processed_dir, train_tensor_file))
    train_labels = torch.load(os.path.join(processed_dir, train_labels_file))
    test_data = torch.load(os.path.join(processed_dir, test_tensor_file))
    test_labels = torch.load(os.path.join(processed_dir, test_labels_file))
    N_train = 25000
    N_test = 5000

    # check correct length of training and test data
    assert len(train_data) == N_train
    assert len(test_data) == N_test

    #assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    for p in train_data:
        assert p.shape == (1,28,28)
    for p in test_data:
        assert p.shape == (1,28,28)
    
    #assert that all labels are represented
    assert len(train_labels) == len(train_data)
    assert len(test_labels) == len(test_data)

