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
batch_size = 50
bcd_n_iter = 1
n_epochs = 2
dict_alpha = 100.
dataset = os.environ.get("DATASET", "HCP zmaps")
n_jobs = os.environ.get("N_JOBS", None)
penalties = ["L1", "social", "Graph-Net"]
if n_jobs is None:
    if "parietal" in os.environ["HOME"]:
        n_jobs = 20
    else:
        n_jobs = 1
else:
    n_jobs = int(n_jobs)
if dataset == "IBC bold":
    n_components = 100
    batch_size = 100
    penalties = ["L1", "social"]
if dataset == "HCP bold":
    batch_size = 200
    penalties = ["L1", "social"]


def get_data(dataset):
    train_size = .75
    misc = {}
    if dataset == "HCP zmaps":
        mask_img = os.path.join(data_dir, hcp_distro, "mask_img.nii.gz")
        zmaps = fetch_hcp_task(os.path.join(data_dir, hcp_distro))
        X = zmaps.groupby("subject_id")["zmap"].apply(list).tolist()
    elif dataset == "HCP rest":
        rs_filenames, _, mask_img = load_hcp_rest(
            data_dir=os.path.join(data_dir, hcp_distro), raw=True,
            test_size=0.)
        rs_filenames = np.concatenate(rs_filenames)
        X = [xs for Xs in rs_filenames for xs in np.load(Xs, mmap_mode='r')[::6]]
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
        X = load_imgs(X)
        misc["trs"] = list(map(len, X))
        X = [img for Xs in X for img in Xs]
    else:
        raise NotImplementedError(dataset)
    mask_img = check_niimg(mask_img)
    X_train, X_test = train_test_split(X, train_size=train_size)
    return X_train, X_test, mask_img, misc


X_train, X_test, mask_img, misc = get_data(dataset)
if "zmaps" in dataset:
    X_train = [img for Xs in X_train for img in Xs]
    n_files = len(X_train)
    tmp = []
    for i in range(n_files // batch_size):
        tmp.append(X_train[i * batch_size: (i + 1) * batch_size])
    tmp.append(X_train[(i + 1) * batch_size:])
    X_train = tmp


class Artefacts(object):
    def __init__(self, penalty):
        self.penalty = penalty
        self.durations_ = []
        self.scores_ = dict(r2=[], pearsonr=[])

    def start_clock(self):
        self.t0_ = time.time()

    def callback(self, env):
        self.durations_.append(time.time() - self.t0_)
        model = env["self"]
        components = model.dico_.components_.copy()
        record_codes = []
        for Xs in X_test:
            record_codes.append(model.transform(Xs))
        self.compute_scores(record_codes, components)
        # self.components_.append(components)
        # self.codes_.append(codes)

    def compute_scores(self, record_codes, components):
        record_scores = dict(r2=[])
        for Xs, codes in zip(X_test, record_codes):
            Xs = self.model_._load_data(Xs)
            Xs_pred = np.dot(codes, components)
            for scorer in ["r2"]:
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(score_multioutput)(
                        x_true, x_pred, scorer=scorer)
                    for x_true, x_pred in zip(Xs, Xs_pred))
                record_scores[scorer].append(scores)
        for scorer in record_scores:
            self.scores_[scorer].append(record_scores[scorer])


coder = partial(_enet_coder, l1_ratio=0., alpha=1.)
graphnet_prox = ProximalOperator(which="graph-net", affine=mask_img.affine,
                                 mask=mask_img.get_data().astype(bool),
                                 radius=10., l1_ratio=.1)
social_prox = ProximalOperator(which="social", affine=mask_img.affine,
                               fwhm=2, mask=mask_img.get_data().astype(bool),
                               kernel="gaussian", radius=10.)
all_artefacts = []
for penalty in penalties:
    if penalty == "social":
        prox = social_prox
        dict_alpha = 100.
    elif penalty == "L1":
        prox = -1
        dict_alpha = 100.
    elif penalty == "Graph-Net":
        prox = graphnet_prox
    else:
        raise NotImplementedError(penalty)

    artefacts = Artefacts(penalty)
    model = ProximalfMRIMiniBatchDictionaryLearning(
        n_components=n_components, random_state=random_state,
        fit_algorithm=coder, dict_penalty_model=prox, mask=mask_img,
        n_epochs=n_epochs, callback=artefacts.callback, n_jobs=n_jobs,
        batch_size=batch_size, dict_alpha=dict_alpha, verbose=1)
    artefacts.model_ = model
    # model.from_niigz("sodl_ibc_bold_100.nii.gz")
    artefacts.start_clock()
    model.fit(X_train)
    all_artefacts.append(artefacts)


def create_results_df():
    import pandas as pd
    scores_df = []
    time_df = []
    for artefacts in all_artefacts:
        n_heartbeat = len(artefacts.scores_["r2"])
        for ns in range(n_heartbeat):
            n_samples = int(
                np.ceil((ns + 1) * len(X_train) / float(n_heartbeat)))
            time_df.append(dict(model=artefacts.penalty,
                                n_samples=n_samples,
                                duration=artefacts.durations_[ns]))
            for tid in range(len(X_test)):
                res = {"model": artefacts.penalty, "n_samples": n_samples,
                       "tid": tid}
                for scorer in ["r2", "pearsonr"]:
                    score = artefacts.scores_[scorer][ns][tid]
                    res[scorer] = score
                scores_df.append(res)
    return pd.DataFrame(scores_df), pd.DataFrame(time_df)


def plot_results(scores_df, timing_df=None, output_dir="unknown_dataset",
                 dataset=None, figsize=(3, 2.5)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    if output_dir is not None:
        dataset = dataset.replace(" ", "_")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def rename_model(model):
        if model == "L1":
            return "L1 constraint"
        elif model == "social":
            return "social sparsity"
        else:
            return model

    n_samples = scores_df["n_samples"].unique().tolist()
    n_samples = n_samples[:3] + n_samples[4:][::-5]
    scores_df = scores_df[scores_df.n_samples.isin(n_samples)]
    scores_df["penalty model"] = scores_df["model"].apply(rename_model)
    hue_order = ["L1 constraint", "social sparsity", "Graph-Net"]
    _, ax = plt.subplots(figsize=figsize)
    sns.pointplot(data=scores_df, x="n_samples", y="r2", hue="penalty model",
                  hue_order=hue_order, ax=ax)
    plt.xlabel("# samples (3D MRI images)")
    plt.ylabel("$R^2$ score")
    plt.tight_layout()
    if output_dir is not None:
        out_file = os.path.join(output_dir, "lc_scores_%s.png" % dataset)
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        print(out_file)
    if timing_df is not None:
        timing_df = timing_df[timing_df.n_samples.isin(n_samples)]
        timing_df["penalty model"] = timing_df["model"].apply(rename_model)
        # timing_df["duration_minutes"] = timing_df["duration"] / 60.
        _, ax = plt.subplots(figsize=figsize)
        sns.pointplot(data=timing_df, x="n_samples", y="duration",
                      hue="penalty model", hue_order=hue_order, ax=ax)
        ax.legend_.remove()
        plt.ylabel("time (seconds)")
        plt.xlabel("# samples (3D MRI images)")
        plt.tight_layout()
        if output_dir is not None:
            out_file = os.path.join(output_dir, "lc_timing_%s.png" % dataset)
            plt.savefig(out_file, dpi=200, bbox_inches="tight")
            print(out_file)


def plot_dico(components_img, tag="sodl", close_figs=False,
              output_dir="."):
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map
    from nilearn.image.image import iter_img
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dico_out_file = os.path.join(
        output_dir, "%s_%s.nii.gz" % (tag, dataset.replace(" ", "_")))
    print(dico_out_file)
    for c, c_img in enumerate(iter_img(components_img)):
        if c == 3:
            continue
        plot_stat_map(c_img, display_mode="ortho", colorbar=False,
                      black_bg=True)
        out_file = os.path.join(
            output_dir, "%s_%s_comp%02i.png" % (
                tag, dataset.replace(" ", "_"), c))
        plt.savefig(out_file, dpi=200, bbox_inches="tight", pad_inches=0)
        os.system("mogrify -trim %s" % out_file)
        print(out_file)
        if close_figs:
            plt.close("all")
    if not close_figs:
        plt.show()


