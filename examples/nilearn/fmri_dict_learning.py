from functools import partial
import numpy as np
from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import MiniBatchDictionaryLearning
from nilearn._utils.compat import _basestring
from nilearn.decomposition import (DictLearning as
                                   fMRIMiniBatchDictionaryLearning)
from nilearn.decomposition.dict_learning import BaseDecomposition
from nilearn.decomposition.base import mask_and_reduce
from sklearn.linear_model.coordescendant import L11_PENALTY
from sklearn.decomposition.dict_learning_utils import _general_update_dict
import os
import sys
sys.path.append(os.path.join(os.environ["HOME"],
                             "CODE/FORKED/elvis_dohmatob/rsfmri2tfmri"))
from datasets import load_imgs


class ProximalfMRIMiniBatchDictionaryLearning(fMRIMiniBatchDictionaryLearning):
    def __init__(self, n_components=20, random_state=None, mask=None,
                 smoothing_fwhm=None, standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None, target_affine=None,
                 target_shape=None, mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0, n_jobs=1,
                 dict_penalty_model=L11_PENALTY, positive=True, batch_size=20,
                 n_epochs=1, reduction_ratio=1., dict_init=None, alpha=1.,
                 dict_alpha=1., fit_algorithm="lars", rescale_atoms=False,
                 callback=None, learning_curve_nticks=10, verbose=0):
        fMRIMiniBatchDictionaryLearning.__init__(
            self, n_components=n_components, random_state=random_state,
            mask=mask, smoothing_fwhm=smoothing_fwhm, standardize=standardize,
            detrend=detrend, low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args, memory=memory,
            memory_level=memory_level, n_jobs=n_jobs, batch_size=batch_size,
            reduction_ratio=reduction_ratio, dict_init=dict_init,
            alpha=alpha, n_epochs=n_epochs, verbose=verbose)
        self.dict_penalty_model = dict_penalty_model
        self.dict_alpha = dict_alpha
        self.positive = positive
        self.fit_algorithm = fit_algorithm
        self.rescale_atoms = rescale_atoms
        self.callback = callback
        self.learning_curve_nticks = learning_curve_nticks

    def _load_data(self, data, confounds=None):
        if isinstance(data[0], _basestring):
            if data[0].endswith(".npy"):
                return load_imgs(data)
            else:
                return mask_and_reduce(
                    self.masker_, data, confounds,  n_jobs=self.n_jobs,
                    reduction_ratio=self.reduction_ratio,
                    n_components=self.n_components,
                    random_state=self.random_state,
                    memory_level=max(0, self.memory_level - 1))
        else:
            return np.asarray(data)

    def fit(self, imgs, y=None, confounds=None):
        BaseDecomposition.fit(self, imgs)
        return self._raw_fit(imgs)

    def _log(self, msg):
        if self.verbose:
            print('[fMRIProximalMiniBatchDictionaryLearning] %s' % msg)

    def _raw_fit(self, data, confounds=None):
        """Helper function that direcly process unmasked data

        Parameters
        ----------
        data: ndarray,
            Shape (n_samples, n_features)
        """
        # misc
        updater = partial(_general_update_dict, reg=self.dict_alpha,
                          penalty_model=self.dict_penalty_model,
                          positive=self.positive)
        if not hasattr(self, "sodl_"):  # warm-start maybe
            self.sodl_ = MiniBatchDictionaryLearning(
                n_components=self.n_components, random_state=self.random_state,
                verbose=self.verbose, updater=updater, ensure_nonzero=True,
                fit_algorithm=self.fit_algorithm, dict_init=self.dict_init,
                n_iter=1)
        n_samples = len(data)
        batch_size = min(n_samples, self.batch_size)
        batches = list(gen_batches(len(data), batch_size))
        n_iter = len(batches)
        cb_freq = max(1, n_iter // self.learning_curve_nticks)

        # loop over data and incrementally fit the model
        for epoch in range(self.n_epochs):
            cnt = 0
            for batch in batches:
                # update model on incoming mini batch of data
                cnt += 1
                self._log('Epoch %02i/%02i, batch %02i/%02i' % (
                    epoch + 1, self.n_epochs, cnt, n_iter))
                data_batch = self._load_data(data[batch])
                self.sodl_.partial_fit(data_batch)

                # invoke user-supplied callback
                if self.callback is not None:
                    if (cnt % cb_freq == 0 or
                        cnt == n_iter - 1):
                        self.callback(locals())

        # final sip
        self.components_ = self.sodl_.components_
        if self.rescale_atoms:
            # Unit-variance scaling
            S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
            S[S == 0] = 1
            self.components_ /= S[:, None]

            # Flip signs in each composant so that positive part is l1 larger
            # than negative part. Empirically this yield more positive looking
            # maps than with setting the max to be positive.
            for component in self.components_:
                if np.sum(component > 0) < np.sum(component < 0):
                    component *= -1
        return self

    def transform(self, imgs, components=None, batch_size=None):
        if components is None:
            check_is_fitted(self, "sodl_")
            sodl = self.sodl_
        else:
            sodl = clone(self.sodl_)
            sodl.components_ = components
        if isinstance(imgs, _basestring):
            imgs = [imgs]
        n_imgs = len(imgs)
        n_components = len(sodl.components_)
        codes = np.ndarray((n_imgs, n_components),
                           dtype=sodl.components_.dtype)
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = min(n_imgs, batch_size)
        for batch in gen_batches(len(imgs), batch_size):
            data_batch = self._load_data(imgs[batch])
            codes[batch] = sodl.transform(data_batch)
        return codes
