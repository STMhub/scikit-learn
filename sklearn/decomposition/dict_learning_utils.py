import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model.coordescendant import L2_CONSTRAINT, coordescendant


def _general_update_dict(dictionary, B, A, precomputed=True,
                         solver=coordescendant, penalty_model=L2_CONSTRAINT,
                         reg=1., l2_reg=0., max_iter=1, positive=False,
                         emulate_sklearn=False, random=False,
                         random_state=None, reduction_ratio=1.):
    """Applies BCD dictionary-update for online dictionary-learning"""
    reg = float(reg)
    l2_reg = float(l2_reg)
    rng = check_random_state(random_state)
    W = np.ascontiguousarray(dictionary.T)
    X_or_Gram, Y_or_Cov = map(np.asfortranarray,
                              (A.conjugate().T, B.conjugate().T))
    if reduction_ratio < 1.:
        n_targets = W.shape[1]
        mask = rng.rand(n_targets) > reduction_ratio
        W_ = W[:, mask]
        Y_or_Cov_ = Y_or_Cov[:, mask]
    else:
        W_ = W
        Y_or_Cov_ = Y_or_Cov
    dictionary = solver(
        W_, reg, l2_reg, X_or_Gram, Y_or_Cov_, max_iter=max_iter,
        precomputed=precomputed, penalty_model=penalty_model,
        positive=positive, emulate_sklearn_dl=emulate_sklearn, random=random,
        rng=rng)[0]
    if reduction_ratio < 1.:
        W[:, mask] = dictionary
    else:
        W = dictionary
    return dictionary.T
