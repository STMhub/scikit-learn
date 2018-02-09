from functools import partial
import numpy as np
from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.utils import gen_batches, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import MiniBatchDictionaryLearning
from nilearn._utils.compat import _basestring
from nilearn.decomposition import (DictLearning as
                                   fMRIMiniBatchDictionaryLearning)
from nilearn.masking import apply_mask
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
                 dict_penalty_model=L11_PENALTY, fit_algorithm=None,
                 transform_algorithm=None, positive=True, batch_size=20,
                 n_epochs=1, reduction_ratio=1., reduction_factor=1,
                 dict_init=None, alpha=1., dict_alpha=1., rescale_atoms=False,
                 callback=None, learning_curve_nticks=5, backend="sklearn",
                 block_size=1., n_blocks=1, verbose=0):
        fMRIMiniBatchDictionaryLearning.__init__(
            self, n_components=n_components, random_state=random_state,
            mask=mask, smoothing_fwhm=smoothing_fwhm, standardize=standardize,
            detrend=detrend, low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args, memory=memory,
            memory_level=memory_level, n_jobs=n_jobs, batch_size=batch_size,
            reduction_ratio=reduction_ratio, dict_init=dict_init,
            alpha=alpha, n_epochs=n_epochs, verbose=verbose)
        self.backend = backend
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.reduction_factor = reduction_factor
        self.dict_penalty_model = dict_penalty_model
        self.dict_alpha = dict_alpha
        self.positive = positive
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.rescale_atoms = rescale_atoms
        self.callback = callback
        self.learning_curve_nticks = learning_curve_nticks

    def _load_data(self, data, confounds=None, ensure_2d=False):
        if isinstance(data[0], np.ndarray):
            return data
        if isinstance(data, _basestring) and data.endswith(".npy"):
            return load_imgs([data])[0]
        elif isinstance(data[0], _basestring) and data[0].endswith(".npy"):
            return load_imgs(data)
        else:
            return mask_and_reduce(
                self.masker_, data, confounds,  n_jobs=self.n_jobs,
                reduction_ratio=self.reduction_ratio,
                n_components=self.n_components,
                random_state=self.random_state,
                memory_level=max(0, self.memory_level - 1))

    def _prefit(self, imgs=None):
        if imgs is not None:
            BaseDecomposition.fit(self, imgs)
        dico_extra_params = {}
        updater = partial(_general_update_dict, reg=self.dict_alpha,
                          penalty_model=self.dict_penalty_model,
                          positive=self.positive, block_size=self.block_size,
                          n_blocks=self.n_blocks)
        for param in ["transform_algorithm"]:
            if hasattr(self, param) and getattr(self, param) is not None:
                dico_extra_params[param] = getattr(self, param)
        if not hasattr(self, "dico_"):  # warm-start maybe
            if self.backend == "sklearn":
                self.dico_ = MiniBatchDictionaryLearning(
                    n_components=self.n_components, updater=updater,
                    random_state=self.random_state, verbose=self.verbose,
                    ensure_nonzero=True, fit_algorithm=self.fit_algorithm,
                    dict_init=self.dict_init, n_jobs=self.n_jobs, n_iter=1,
                    **dico_extra_params)
            elif self.backend == "modl":
                from modl.decomposition.dict_fact import DictFact
                self.dico_ = DictFact(
                    n_components=self.n_components, code_alpha=1.,
                    random_state=self.random_state, verbose=0,
                    n_epochs=self.n_epochs, code_l1_ratio=0.,
                    dict_init=self.dict_init)
                self.dico_.n_iter_ = 0
            else:
                raise NotImplementedError(self.backend)

    def from_niigz(self, components_img, imgs=None):
        self._prefit(imgs=imgs)
        if imgs is None:
            self.dico_.components_ = apply_mask(components_img, self.mask)
        else:
            self.dico_.components_ = self.masker_.transform(components_img)
        self.components_ = self.dico_.components_
        return self

    def fit(self, imgs, y=None, confounds=None):
        self._prefit(imgs)
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
        n_records = len(data)
        self._prefit()

        # loop over data and incrementally fit the model
        rng = check_random_state(self.random_state)
        cb_freq = None
        for epoch in range(self.n_epochs):
            cnt = 0
            rng.shuffle(data)  # this is important for fast convergence
            for s, record_data in enumerate(data):
                if isinstance(record_data, _basestring):
                    record_data = self._load_data(record_data)
                if self.reduction_factor is not None:
                    record_data = record_data[::self.reduction_factor]
                batch_size = min(self.batch_size, len(record_data))
                batches = list(gen_batches(len(record_data), batch_size))
                n_iter = len(batches)
                if cb_freq is None:
                    cb_freq = max(
                        n_iter * n_records // self.learning_curve_nticks, 1)
                rng.shuffle(batches)
                for b, batch in enumerate(batches):
                    # update model on incoming mini batch of data
                    cnt += 1
                    self._log(
                        'Epoch %02i/%02i record %02i/%02i batch %02i/%02i' % (
                            epoch + 1, self.n_epochs, s + 1, n_records, b + 1,
                            n_iter))
                    data_batch = self._load_data(record_data[batch])[:, :1000]
                    self.dico_.partial_fit(data_batch)

                    # invoke user-supplied callback
                    if self.callback is not None:
                        if cnt % cb_freq == 0 or epoch == 0 and cnt < 3:
                            self.callback(locals())

        # just in case we missed some data
        if self.callback is not None:
            self.callback(locals())

        # final sip
        self.components_ = self.dico_.components_
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
            check_is_fitted(self, "dico_")
            dico = self.dico_
        else:
            dico = clone(self.dico_)
            dico.components_ = components
        if isinstance(imgs, _basestring):
            imgs = self._load_data(imgs)
        n_imgs = len(imgs)
        n_components = len(dico.components_)
        codes = np.ndarray((n_imgs, n_components),
                           dtype=dico.components_.dtype)
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = min(n_imgs, batch_size)
        for batch in gen_batches(len(imgs), batch_size):
            data_batch = self._load_data(imgs[batch])
            codes[batch] = dico.transform(data_batch)
        return codes
