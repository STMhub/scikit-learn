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
# from utils import score_multioutput
from datasets import load_hcp_rest, load_imgs, fetch_hcp_task


def bundle_up(X, batch_size):
    X = [img for Xs in X for img in Xs]
    n_files = len(X)
    batch_size = min(n_files, batch_size)
    tmp = []
    for i in range(n_files // batch_size):
        tmp.append(X[i * batch_size: (i + 1) * batch_size])
    remainder = X[(i + 1) * batch_size:]
    if len(remainder):
        tmp.append(remainder)
    return tmp

output_dir = os.path.abspath(os.environ.get("OUTPUT_DIR", "."))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
benchmark = False
random_state = 42
n_components = 40
batch_size = 50
bcd_n_iter = 1
n_epochs = 1
dict_alpha = 100.
dataset = os.environ.get("DATASET", "HCP rest")
n_jobs = os.environ.get("N_JOBS", None)
penalties = ["L1", "social", "Graph-Net"]
n_jobs = 1
if n_jobs is None:
    if "parietal" in os.environ["HOME"]:
        n_jobs = 20
    else:
        n_jobs = 1
else:
    n_jobs = int(n_jobs)
reduction_factor = None
if dataset == "IBC bold":
    batch_size = 200
    n_components = 100
    penalties = ["L1"]
if dataset == "HCP rest":
    n_components = 100
    batch_size = 200
    # reduction_factor = 6
    penalties = ["L1"]


def get_data(dataset):
    if benchmark:
        train_size = .75
    else:
        train_size = None
    misc = {}
    if dataset == "HCP zmaps":
        if benchmark:
            train_size = .95
        mask_img = os.path.join(data_dir, hcp_distro, "mask_img.nii.gz")
        zmaps = fetch_hcp_task(os.path.join(data_dir, hcp_distro))
        X = zmaps.groupby(
            "subject_id")["zmap"].apply(list).tolist()
    elif dataset == "HCP rest":
        rs_filenames, _, mask_img = load_hcp_rest(
            data_dir=os.path.join(data_dir, hcp_distro), raw=True,
            test_size=0)
        # X = np.concatenate(rs_filenames)
        rs_filenames = rs_filenames[:100]
        X = [files[0] for files in rs_filenames]
        X = [Xs for Xs in X if os.path.exists(Xs)]
        if benchmark:
            train_size = .8
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
                                   "sub-*/ses-*/func/wrdcsub-*_bold.npy"))
        X = load_imgs(X)
    else:
        raise NotImplementedError(dataset)
    mask_img = check_niimg(mask_img)
    if train_size is None:
        X_train, X_test = X, []
    else:
        X_train, X_test = train_test_split(X, train_size=train_size)
    return X_train, X_test, mask_img, misc


X_train, X_test, mask_img, misc = get_data(dataset)
n_subjects = len(X_train)
if "zmaps" in dataset:
    X_train = bundle_up(X_train, batch_size)


class Artifacts(object):
    def __init__(self, penalty):
        self.penalty = penalty
        self.durations_ = []
        self.scores_ = dict(r2=[], pearsonr=[])

    def start_clock(self):
        self.t0_ = time.time()

    def callback(self, env):
        self.durations_.append(time.time() - self.t0_)
        now = time.time()
        print("\n[Artifacts] Computing test scores")
        model = env["self"]
        components = model.dico_.components_
        codes = []
        X_test_bundle = X_test
        if "zmaps" in dataset:
            X_test_bundle = bundle_up(X_test, 10 * batch_size)
        for Xs in X_test_bundle:
            codes.append(model.transform(Xs))
        self.compute_scores(X_test_bundle, codes, components, model)
        lost = time.time() - now
        self.t0_ += lost
        self.durations_.append(time.time() - self.t0_)
        # self.components_.append(components)
        # self.codes_.append(codes)

    def compute_scores(self, record_data, record_codes, components,
                       model):
        record_scores = dict(r2=[])
        for Xs, codes in zip(record_data, record_codes):
            Xs = model._load_data(Xs)
            Xs_pred = np.dot(codes, components)
            for scorer in ["r2"]:
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(score_multioutput)(
                        x_true, x_pred, scorer=scorer)
                    for x_true, x_pred in zip(Xs, Xs_pred))
                record_scores[scorer] += scores
        for scorer in record_scores:
            self.scores_[scorer].append(record_scores[scorer])

if __name__ == "__main__":
    coder = partial(_enet_coder, l1_ratio=0., alpha=1.)
    graphnet_prox = ProximalOperator(which="graph-net", affine=mask_img.affine,
                                     mask=mask_img.get_data().astype(bool),
                                     radius=10., l1_ratio=.1, max_iter=1,
                                     verbose=0)
    social_prox = ProximalOperator(which="social", affine=mask_img.affine,
                                   fwhm=2, mask=mask_img.get_data().astype(bool),
                                   kernel="gaussian")
    all_artifacts = []
    for penalty in penalties:
        if penalty == "social":
            prox = social_prox
            dict_alpha = 1.
        elif penalty == "L1":
            prox = -1
            dict_alpha = 100.
        elif penalty == "Graph-Net":
            prox = graphnet_prox
        else:
            raise NotImplementedError(penalty)

        artifacts = Artifacts(penalty)
        model = ProximalfMRIMiniBatchDictionaryLearning(
            n_components=n_components, random_state=random_state,
            fit_algorithm=coder, dict_penalty_model=prox, mask=mask_img,
            n_epochs=n_epochs,  # callback=artifacts.callback,
            n_jobs=n_jobs, reduction_factor=reduction_factor,
            batch_size=batch_size, dict_alpha=dict_alpha, verbose=1,
            backend="sklearn", block_size=.1, n_blocks=3)
        artifacts.model_ = model
        # model.from_niigz("sodl_ibc_bold_100.nii.gz")
        artifacts.start_clock()
        model.fit(X_train)
        all_artifacts.append(artifacts)
        components_img = unmask(model.components_, mask_img)
        dico_out_file = os.path.join(
            output_dir, "%s_%s.nii.gz" % (penalty, dataset.replace(" ", "_")))
        components_img.to_filename(dico_out_file)
        print(dico_out_file)


def create_results_df():
    import pandas as pd
    scores_df = []
    time_df = []
    for artifacts in all_artifacts:
        n_heartbeat = len(artifacts.scores_["r2"])
        if dataset == "HCP zmaps":
            n_subjects = 362
        else:
            n_subjects = len(X_train)
        for ns in range(n_heartbeat):
            n_samples = int(
                np.ceil((ns + 1) * n_subjects / float(n_heartbeat)))
            time_df.append(dict(model=artifacts.penalty,
                                n_samples=n_samples,
                                duration=artifacts.durations_[ns]))
            for tid in range(len(X_test)):
                res = {"model": artifacts.penalty, "n_samples": n_samples,
                       "tid": tid}
                for scorer in ["r2"]:
                    score = artifacts.scores_[scorer][ns][0][tid][0]
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
    n_samples = n_samples[:3] + n_samples[4:][::-4]
    scores_df = scores_df[scores_df.n_samples.isin(n_samples)]
    scores_df["penalty model"] = scores_df["model"].apply(rename_model)
    hue_order = ["L1 constraint", "social sparsity", "Graph-Net"]
    _, ax = plt.subplots(figsize=figsize)
    sns.pointplot(data=scores_df, x="n_samples", y="r2", hue="penalty model",
                  hue_order=hue_order, ax=ax)
    plt.xlabel("# subjects")
    plt.ylabel("$R^2$ score")
    plt.tight_layout()
    if output_dir is not None:
        out_file = os.path.join(output_dir, "lc_scores_%s.png" % dataset)
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        print(out_file)
    if timing_df is not None:
        timing_df = timing_df[timing_df.n_samples.isin(n_samples)]
        timing_df["penalty model"] = timing_df["model"].apply(rename_model)
        timing_df["duration_minutes"] = timing_df["duration"] / 60.
        _, ax = plt.subplots(figsize=figsize)
        sns.pointplot(data=timing_df, x="n_samples", y="duration_minutes",
                      hue="penalty model", hue_order=hue_order, ax=ax)
        ax.legend_.remove()
        plt.ylabel("time (minutes)")
        plt.xlabel("# subjects")
        plt.tight_layout()
        if output_dir is not None:
            out_file = os.path.join(output_dir, "lc_timing_%s.png" % dataset)
            plt.savefig(out_file, dpi=200, bbox_inches="tight")
            print(out_file)


def plot_dico(model, tag="L1", display_mode="yz", close_figs=False):
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map
    from nilearn.image.image import iter_img

    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if hasattr(model, "components_"):
        components_img = unmask(model.components_, mask_img)
    else:
        components_img = check_niimg(model)
    for c, c_img in enumerate(iter_img(components_img)):
        plot_stat_map(c_img, colorbar=False, display_mode=display_mode,
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

