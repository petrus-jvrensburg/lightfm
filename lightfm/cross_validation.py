# coding=utf-8
"""
Dataset splitting functions.
"""

import numpy as np
import scipy.sparse as sp


def random_train_test_split(interactions, sample_weight=None, test_percentage=0.2, random_state=None):
    """
    Randomly split interactions between training and testing.

    This function takes an interaction set and splits it into
    two disjoint sets, a training set and a test set. If a sample_weight set
    is provided, it will be split in the same way as the interaction set.
    Note that no effort is made to make sure that all items and users with
    interactions in the test set also have interactions in the
    training set; this may lead to a partial cold-start problem
    in the test set.

    Parameters
    ----------

    interactions: a scipy sparse matrix containing interactions
        The interactions to split.
    sample_weight: a scipy sparse matrix expressing weights of individual interactions, optional
        The sample weights corresponding to the interactions that are to be split.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: int or numpy.random.RandomState, optional
        Random seed used to initialize the numpy.random.RandomState number generator.
        Accepts an instance of numpy.random.RandomState for backwards compatibility.

    Returns
    -------

    (train, test): (scipy.sparse.COOMatrix,
                    scipy.sparse.COOMatrix)
         A tuple of (train data, test data)
    """

    if not sp.issparse(interactions):
        raise ValueError("Interactions must be a scipy.sparse matrix.")

    if sample_weight is not None and not sp.issparse(sample_weight):
        raise ValueError("Sample weight must be a scipy.sparse matrix.")

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)

    interactions = interactions.tocoo()

    shape = interactions.shape
    uids, iids, data = (interactions.row, interactions.col, interactions.data)

    # shuffle matrices
    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)
    uids, iids, data = uids[shuffle_indices], iids[shuffle_indices], data[shuffle_indices]

    if sample_weight is not None:
        weights_uids, weights_iids, weights_data = (
            sample_weight.row, sample_weight.col, sample_weight.data
        )
        weights_uids, weights_iids, weights_data = (
            weights_uids[shuffle_indices], weights_iids[shuffle_indices], weights_data[shuffle_indices]
        )

    cutoff = int((1.0 - test_percentage) * len(uids))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = sp.coo_matrix(
        (data[train_idx], (uids[train_idx], iids[train_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )
    test = sp.coo_matrix(
        (data[test_idx], (uids[test_idx], iids[test_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )

    if sample_weight is not None:
        train_weights = sp.coo_matrix(
            (weights_data[train_idx], (weights_uids[train_idx], weights_iids[train_idx])),
            shape=shape,
            dtype=sample_weight.dtype,
        )
        test_weights = sp.coo_matrix(
            (weights_data[test_idx], (weights_uids[test_idx], weights_iids[testn_idx])),
            shape=shape,
            dtype=sample_weight.dtype,
        )
        return train, test, train_weights, test_weights

    return train, test
