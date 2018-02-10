import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model.coordescendant import L2_CONSTRAINT, coordescendant


def _general_update_dict(dictionary, B, A, precomputed=True,
                         solver=coordescendant, penalty_model=L2_CONSTRAINT,
                         reg=1., l2_reg=0., max_iter=1, positive=False,
                         emulate_sklearn=False, random=False, verbose=0,
                         random_state=None, **n_jobs):
    """Applies BCD dictionary-update for online dictionary-learning"""
    rng = check_random_state(random_state)
    reg = float(reg)
    l2_reg = float(l2_reg)
    dictionary = np.ascontiguousarray(dictionary.T)
    X_or_Gram, Y_or_Cov = map(np.asfortranarray,
                              (A.conjugate().T, B.conjugate().T))
    return solver(
        dictionary, reg, l2_reg, X_or_Gram, Y_or_Cov, max_iter=max_iter,
        precomputed=precomputed, penalty_model=penalty_model,
        positive=positive, emulate_sklearn_dl=emulate_sklearn,
        random=random, rng=rng)[0].T
    return dictionary
