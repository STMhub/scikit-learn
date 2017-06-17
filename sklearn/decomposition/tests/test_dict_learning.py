import numpy as np
import itertools

from sklearn.exceptions import ConvergenceWarning

from sklearn.utils import check_array

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import TempMemmap

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.decomposition.dict_learning import (dict_learning_online,
                                                 dict_learning)
from sklearn.decomposition import sparse_encode

from sklearn.decomposition.dict_learning import _update_dict

rng_global = np.random.RandomState(0)
n_samples, n_features = 10, 8
X = rng_global.randn(n_samples, n_features)


def test_sparse_encode_shapes_omp():
    rng = np.random.RandomState(0)
    algorithms = ['omp', 'lasso_lars', 'lasso_cd', 'lars', 'threshold']
    for n_components, n_samples in itertools.product([1, 5], [1, 9]):
        X_ = rng.randn(n_samples, n_features)
        dictionary = rng.randn(n_components, n_features)
        for algorithm, n_jobs in itertools.product(algorithms, [1, 3]):
            code = sparse_encode(X_, dictionary, algorithm=algorithm,
                                 n_jobs=n_jobs)
            assert_equal(code.shape, (n_samples, n_components))


def test_dict_learning_shapes():
    n_components = 5
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert_equal(dico.components_.shape, (n_components, n_features))

    n_components = 1
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert_equal(dico.components_.shape, (n_components, n_features))
    assert_equal(dico.transform(X).shape, (X.shape[0], n_components))


def test_dict_learning_overcomplete():
    n_components = 12
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert_true(dico.components_.shape == (n_components, n_features))


def test_dict_learning_reconstruction():
    n_components = 12
    dico = DictionaryLearning(n_components, transform_algorithm='omp',
                              transform_alpha=0.001, random_state=0)
    code = dico.fit(X).transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X)

    dico.set_params(transform_algorithm='lasso_lars')
    code = dico.transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)

    # used to test lars here too, but there's no guarantee the number of
    # nonzero atoms is right.


def test_dict_learning_reconstruction_parallel():
    # regression test that parallel reconstruction works with n_jobs=-1
    n_components = 12
    dico = DictionaryLearning(n_components, transform_algorithm='omp',
                              transform_alpha=0.001, random_state=0, n_jobs=-1)
    code = dico.fit(X).transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X)

    dico.set_params(transform_algorithm='lasso_lars')
    code = dico.transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)


def test_dict_learning_lassocd_readonly_data():
    n_components = 12
    with TempMemmap(X) as X_read_only:
        dico = DictionaryLearning(n_components, transform_algorithm='lasso_cd',
                                  transform_alpha=0.001, random_state=0,
                                  n_jobs=-1)
        with ignore_warnings(category=ConvergenceWarning):
            code = dico.fit(X_read_only).transform(X_read_only)
        assert_array_almost_equal(np.dot(code, dico.components_), X_read_only,
                                  decimal=2)


def test_dict_learning_nonzero_coefs():
    n_components = 4
    dico = DictionaryLearning(n_components, transform_algorithm='lars',
                              transform_n_nonzero_coefs=3, random_state=0)
    code = dico.fit(X).transform(X[np.newaxis, 1])
    assert_true(len(np.flatnonzero(code)) == 3)

    dico.set_params(transform_algorithm='omp')
    code = dico.transform(X[np.newaxis, 1])
    assert_equal(len(np.flatnonzero(code)), 3)


def test_dict_learning_unknown_fit_algorithm():
    n_components = 5
    dico = DictionaryLearning(n_components, fit_algorithm='<unknown>')
    assert_raises(ValueError, dico.fit, X)


def test_dict_learning_split():
    n_components = 5
    dico = DictionaryLearning(n_components, transform_algorithm='threshold',
                              random_state=0)
    code = dico.fit(X).transform(X)
    dico.split_sign = True
    split_code = dico.transform(X)

    assert_array_equal(split_code[:, :n_components] -
                       split_code[:, n_components:], code)


def test_dict_learning_online_shapes():
    rng = np.random.RandomState(0)
    n_components = 8
    code, dictionary = dict_learning_online(X, n_components=n_components,
                                            alpha=1, random_state=rng)
    assert_equal(code.shape, (n_samples, n_components))
    assert_equal(dictionary.shape, (n_components, n_features))
    assert_equal(np.dot(code, dictionary).shape, X.shape)


def test_dict_learning_online_verbosity():
    n_components = 5
    # test verbosity
    from sklearn.externals.six.moves import cStringIO as StringIO
    import sys

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        dico = MiniBatchDictionaryLearning(n_components, n_iter=20, verbose=1,
                                           random_state=0)
        dico.fit(X)
        dico = MiniBatchDictionaryLearning(n_components, n_iter=20, verbose=2,
                                           random_state=0)
        dico.fit(X)
        dict_learning_online(X, n_components=n_components, alpha=1, verbose=1,
                             random_state=0)
        dict_learning_online(X, n_components=n_components, alpha=1, verbose=2,
                             random_state=0)
    finally:
        sys.stdout = old_stdout

    assert_true(dico.components_.shape == (n_components, n_features))


def test_dict_learning_online_estimator_shapes():
    n_components = 5
    dico = MiniBatchDictionaryLearning(n_components, n_iter=20, random_state=0)
    dico.fit(X)
    assert_true(dico.components_.shape == (n_components, n_features))


def test_dict_learning_online_overcomplete():
    n_components = 12
    dico = MiniBatchDictionaryLearning(n_components, n_iter=20,
                                       random_state=0).fit(X)
    assert_true(dico.components_.shape == (n_components, n_features))


def test_dict_learning_online_initialization():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    dico = MiniBatchDictionaryLearning(n_components, n_iter=0,
                                       dict_init=V, random_state=0).fit(X)
    assert_array_equal(dico.components_, V)


def test_dict_learning_online_partial_fit():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    dict1 = MiniBatchDictionaryLearning(n_components, n_iter=10 * len(X),
                                        batch_size=1,
                                        alpha=1, shuffle=False, dict_init=V,
                                        random_state=0).fit(X)
    dict2 = MiniBatchDictionaryLearning(n_components, alpha=1,
                                        n_iter=1, dict_init=V,
                                        random_state=0)
    for i in range(10):
        for sample in X:
            dict2.partial_fit(sample[np.newaxis, :])

    assert_true(not np.all(sparse_encode(X, dict1.components_, alpha=1) ==
                           0))
    assert_array_almost_equal(dict1.components_, dict2.components_,
                              decimal=2)


def test_sparse_encode_shapes():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    for algo in ('lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'):
        code = sparse_encode(X, V, algorithm=algo)
        assert_equal(code.shape, (n_samples, n_components))


def test_sparse_encode_input():
    n_components = 100
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    Xf = check_array(X, order='F')
    for algo in ('lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'):
        a = sparse_encode(X, V, algorithm=algo)
        b = sparse_encode(Xf, V, algorithm=algo)
        assert_array_almost_equal(a, b)


def test_sparse_encode_error():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    code = sparse_encode(X, V, alpha=0.001)
    assert_true(not np.all(code == 0))
    assert_less(np.sqrt(np.sum((np.dot(code, V) - X) ** 2)), 0.1)


def test_sparse_encode_error_default_sparsity():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 64)
    D = rng.randn(2, 64)
    code = ignore_warnings(sparse_encode)(X, D, algorithm='omp',
                                          n_nonzero_coefs=None)
    assert_equal(code.shape, (100, 2))


def test_unknown_method():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    assert_raises(ValueError, sparse_encode, X, V, algorithm="<unknown>")


def test_sparse_coder_estimator():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    code = SparseCoder(dictionary=V, transform_algorithm='lasso_lars',
                       transform_alpha=0.001).transform(X)
    assert_true(not np.all(code == 0))
    assert_less(np.sqrt(np.sum((np.dot(code, V) - X) ** 2)), 0.1)


def test__update_dict_implements_ref_paper():
    # The problem to minimize is .5 * tr(VtVA) - tr(VtB) subject
    # to ||V[:, k]|| <= 1 for all 0 <= k <=n_components,
    # where V = dictionary, etc. See eqn 9 of ref paper
    # http://www.di.ens.fr/sierra/pdfs/icml09.pdf
    #
    # If n_components = n_features = 1, then this should reduce to a
    # scalar quadratic problem: minimize .5 AV**2 - BV subject to |V| <= 1
    # which has unique solution V = B (assuming |B| <= 1)
    A = np.array([[1.]])
    B = np.array([[.5]])
    V = rng_global.randn(1, 1)
    _update_dict(V, B, A)
    assert_array_equal(V, B)


def test_dict_learning_output_lengths():
    rng = np.random.RandomState(0)
    n_components = 8
    alpha = 1.
    for return_n_iter in [True, False]:
        out = dict_learning(X, n_components, alpha, random_state=rng,
                            return_n_iter=return_n_iter)
        assert_equal(len(out), 4 if return_n_iter else 3)
    # _, _, errors, n_iter = dict_learning(
    #     X, n_components, alpha, random_state=rng, return_n_iter=True)
    # assert_equal(len(errors), n_iter)


def test_dict_learning_online_output_lengths():
    rng = np.random.RandomState(0)
    n_components = 8
    alpha = 1.
    for return_code, return_inner_stats, return_n_iter in itertools.product(
            [False, True], [False, True], [False, True]):
        out = dict_learning_online(X, n_components, alpha, random_state=rng,
                                   return_n_iter=return_n_iter,
                                   return_code=return_code,
                                   return_inner_stats=return_inner_stats)
        l = 1
        if return_inner_stats:
            l += 1
        if return_code:
            if not return_inner_stats:
                # dict_learning_online wierdly makes the options
                # return_inner_stats  and return_code mutually exclusive
                l += 1
        if return_n_iter:
            l += 1
        if l == 1:
            assert_true(isinstance(out, np.ndarray))
            assert_equal(len(out), n_components)
        else:
            assert_equal(len(out), l)
