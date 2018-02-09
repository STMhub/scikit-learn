import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state, resample
from sklearn.linear_model.coordescendant import L2_CONSTRAINT, coordescendant


def _general_update_dict(dictionary, B, A, precomputed=True,
                         solver=coordescendant, penalty_model=L2_CONSTRAINT,
                         reg=1., l2_reg=0., max_iter=1, positive=False,
                         emulate_sklearn=False, random=False, verbose=0,
                         random_state=None, n_blocks=1, block_size=1., lr=1.,
                         n_jobs=1):
    """Applies BCD dictionary-update for online dictionary-learning"""
    rng = check_random_state(random_state)
    n_components = len(dictionary[0])
    block_size = max(int(np.ceil(n_components * block_size)), 1)
    if block_size < n_components:
        # split components into blocks
        n_jobs = min(n_jobs, block_size)
        blocks = [resample(range(n_components), n_samples=block_size,
                           random_state=rng, replace=False)
                  for _ in range(n_blocks)]
        blocks = np.asarray(blocks)

        # run coordinate-descent on each block of components
        with Parallel(n_jobs=n_jobs) as parallel:
            dicos = parallel(delayed(_general_update_dict)(
                dictionary[:, block], B[:, block] if precomputed else B,
                A[block][:, block] if precomputed else A[:, block], reg=reg,
                l2_reg=l2_reg, precomputed=precomputed, solver=solver,
                penalty_model=penalty_model, max_iter=max_iter,
                positive=positive, emulate_sklearn=emulate_sklearn,
                random_state=random_state, random=random) for block in blocks)

            # gather results
            old_dictionary = dictionary.copy()
            dictionary = np.zeros_like(dictionary)
            for block, dico in zip(blocks, dicos):
                dictionary[:, block] += dico
            for k, cnt in zip(*np.unique(blocks, return_counts=True)):
                # cnt is the number of blocks which containing atom number k
                if cnt > 1:
                    dictionary[:, k] /= cnt

                    # don't forget the good-old-times just yet
                    if lr < 1.:
                        dictionary[:, k] *= lr
                        dictionary[:, k] += (1. - lr) * old_dictionary[:, k]
                else:
                    # atom k was never picked, restore old value
                    dictionary[:, k] = old_dictionary[:, k]

        return dictionary
    else:
        reg = float(reg)
        l2_reg = float(l2_reg)
        W = np.ascontiguousarray(dictionary.T)
        X_or_Gram, Y_or_Cov = map(np.asfortranarray,
                                  (A.conjugate().T, B.conjugate().T))
        dictionary = solver(
            W, reg, l2_reg, X_or_Gram, Y_or_Cov, max_iter=max_iter,
            precomputed=precomputed, penalty_model=penalty_model,
            positive=positive, emulate_sklearn_dl=emulate_sklearn,
            random=random, rng=rng)[0]
        return dictionary.T
