import pytest
import numpy as np

from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.datasets import fetch_movielens


def _fake_data(n):
    users = np.random.choice(10000, (n, 1))
    items = np.random.choice(10000, (n, 1))
    weight = np.random.rand(n,1)
    return np.concatenate((users, items, weight), axis=1)

@pytest.mark.parametrize("test_percentage", [0.2, 0.5, 0.7])
def test_random_train_test_split(test_percentage):

    data = _fake_data(1000)

    # Use Dataset to prep your interactions and weights.
    dataset = Dataset()

    dataset.fit(users=np.unique(data[:, 0]), items=np.unique(data[:, 1]))
    interactions, sample_weight = dataset.build_interactions((i[0], i[1], i[2]) for i in data)

    train, test = random_train_test_split(interactions, test_percentage=test_percentage)

    assert test.nnz / float(interactions.nnz) == test_percentage
