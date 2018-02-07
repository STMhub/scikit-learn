import os
import glob
import time
from functools import partial
import numpy as np
from sklearn.externals.joblib import delayed, Parallel
from sklearn.model_selection import train_test_split
# from sklearn.metrics import explained_variance_score
from nilearn.image.image import check_niimg
from nilearn.masking import unmask
from nilearn_prox_operators import ProximalOperator
from fmri_dict_learning import ProximalfMRIMiniBatchDictionaryLearning
import sys
sys.path.append(os.path.join(os.environ["HOME"],
                             "CODE/FORKED/elvis_dohmatob/rsfmri2tfmri"))
from config import hcp_distro, data_dir, root
from feature_extraction import _enet_coder
from utils import score_multioutput
from datasets import load_hcp_rest, load_imgs, fetch_hcp_task

random_state = 42
n_components = 40
bcd_n_iter = 1
n_epochs = 1
dict_alpha = 100.
dataset = os.environ.get("DATASET", "IBC zmaps")
n_jobs = os.environ.get("N_JOBS", None)
if n_jobs is None:
    if "parietal" in os.environ["HOME"]:
        n_jobs = 20
    else:
        n_jobs = 1
else:
    n_jobs = int(n_jobs)


def get_data(dataset):
    batch_size = 30
    train_size = .75
    if dataset == "HCP zmaps":
        mask_img = os.path.join(data_dir, hcp_distro, "mask_img.nii.gz")
        zmaps = fetch_hcp_task(os.path.join(data_dir, hcp_distro))
        X = zmaps[zmaps.contrast_name == "STORY-MATH"].groupby( 
           "subject_id")["zmap"].apply(sum)
    elif dataset == "HCP rest":
        rs_filenames, _, mask_img = load_hcp_rest(
            data_dir=os.path.join(data_dir, hcp_distro), raw=True,
            test_size=0.)
        rs_filenames = np.concatenate(rs_filenames)
        X = [xs for Xs in rs_filenames for xs in np.load(Xs, mmap_mode='r')[::6]]
        batch_size = 200
        if train_size is None:
            train_size = .1
    elif dataset == "IBC zmaps":
        zmap_file_pattern = os.path.join(root,
                                         "storage/store/data/ibc",
                                         "derivatives/sub-*/ses-*",
                                         "res_stats_hcp_*_ffx",
                                         "stat_maps/*.nii.gz")
        X = sorted(glob.glob(zmap_file_pattern))
        mask_img = os.path.join(root, "storage/store/data/ibc/derivatives/group",
                                "mask.nii.gz")
    elif dataset == "IBC bold":
        mask_img = os.path.join(root, "storage/store/data/ibc/derivatives/group",
                                "mask.nii.gz")
        X = glob.glob(os.path.join(root, "storage/store/data/ibc/derivatives",
                                   "sub-*/ses-05/func/wrdcsub-*_bold.npy"))
        X = [img for Xs in load_imgs(X) for img in Xs]
    else:
        raise NotImplementedError(dataset)
    mask_img = check_niimg(mask_img)
    X_train, X_test = train_test_split(X, train_size=train_size)
    return X_train, X_test, mask_img, batch_size


X_train, X_test, mask_img, batch_size = get_data(dataset)
artefacts = dict(components=[], codes=[], r2=[], pearsonr=[])


class Artefacts(object):
    def __init__(self, penalty):
        self.penalty = penalty
        self.stuff_ = dict(components=[], codes=[], r2=[], pearsonr=[])

    def callback(self, env):
        model = env["self"]
        components = model.dico_.components_.copy()
        codes = model.transform(X_test)
        self.stuff_["components"].append(components)
        self.stuff_["codes"].append(codes)

    def compute_scores(self):
        X_true = np.asarray(model._load_data(X_test))
        for codes, components in zip(self.stuff_["codes"],
                                     self.stuff_["components"]):
            X_pred = np.dot(codes, components)
            for scorer in ["r2", "pearsonr"]:
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(score_multioutput)(
                        x_true, x_pred, scorer=scorer)
                    for x_true, x_pred in zip(X_true, X_pred))
                self.stuff_[scorer].append(scores)
            # scores = Parallel(n_jobs=n_jobs)(
            #     delayed(explained_variance_score)(x_true, x_pred)
            #     for x_true, x_pred in zip(X_true, X_pred))
            # self.stuff_["ev"].append(scores)


coder = partial(_enet_coder, l1_ratio=0., alpha=1.)
sodl_prox = ProximalOperator(which="graph-net", affine=mask_img.affine,
                             mask=mask_img.get_data().astype(bool), radius=10.,
                             l1_ratio=.1)
social_prox = ProximalOperator(which="graph-net", affine=mask_img.affine,
                               fwhm=2, mask=mask_img.get_data().astype(bool),
                               kernel="gaussian", radius=10.)
all_artefacts = []
for dict_alpha, prox, tag in zip([100., 1.], [-1, sodl_prox],
                                 ["L1", "Graph-Net"]):
    artefacts = Artefacts(tag)
    model = ProximalfMRIMiniBatchDictionaryLearning(
        n_components=n_components, random_state=random_state,
        fit_algorithm=coder, dict_penalty_model=prox, mask=mask_img,
        n_epochs=n_epochs, callback=artefacts.callback, n_jobs=n_jobs,
        batch_size=batch_size, dict_alpha=dict_alpha, verbose=1)
    t0 = time.time()
    model.fit(X_train)
    artefacts.stuff_["duration"] = time.time() - t0
    all_artefacts.append(artefacts)


def create_results_df(recompute_scores=True):
    import pandas as pd
    df = []
    for artefacts in all_artefacts:
        if recompute_scores:
            artefacts.compute_scores()
        elif "ev" not in artefacts.stuff_:
            raise RuntimeError
        n_heartbeat = len(artefacts.stuff_["r2"])
        for ns in range(n_heartbeat):
            n_samples = int(
                np.ceil((ns + 1) * len(X_train) / float(n_heartbeat)))
            for tid in range(len(X_test)):
                res = {"model": artefacts.penalty, "# samples": n_samples,
                       "tid": tid}
                for scorer in ["r2", "pearsonr"]:
                    score = artefacts.stuff_[scorer][ns][tid]
                    res[scorer] = score
                df.append(res)
    return pd.DataFrame(df)


def plot_dico(model, tag="sodl", close_figs=False):
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map
    dico_out_file = "%s_%s.nii.gz" % (tag, dataset.replace(" ", "_"))
    unmask(model.components_, mask_img).to_filename(dico_out_file)
    print(dico_out_file)
    for c in range(0, n_components):
        if c == 3:
            continue
        plot_stat_map(unmask(model.components_[c], mask_img),
                      display_mode="ortho", colorbar=False, black_bg=True)
        out_file = "%s_%s_comp%02i.png" % (tag, dataset.replace(" ", "_"), c)
        plt.savefig(out_file, dpi=200, bbox_inches="tight", pad_inches=0)
        os.system("mogrify -trim %s" % out_file)
        print(out_file)
        if close_figs:
            plt.close("all")
    if not close_figs:
        plt.show()


