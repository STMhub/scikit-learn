import os
import glob
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
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
from datasets import load_hcp_rest, load_imgs

random_state = 42
n_components = 40
bcd_n_iter = 1
n_epochs = 2
dict_alpha = 1.
batch_size = 30
train_size = .75
dataset = "HCP rest"
if "parietal" in os.environ["HOME"]:
    n_jobs = 20
else:
    n_jobs = 1


if dataset == "HCP zmaps":
    from datasets import fetch_hcp_task
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

batch_size = min(batch_size, len(X))
mask_img = check_niimg(mask_img)
n_voxels = mask_img.get_data().sum()

X_train, X_test = train_test_split(X, train_size=train_size)

artefacts = dict(components=[], codes=[], r2=[], pearsonr=[])


def callback(env):
    model = env["self"]
    components = model.sodl_.components_.copy()
    codes = model.transform(X_test)
    artefacts["components"].append(components)
    artefacts["codes"].append(codes)
    X_true = model._load_data(X_test)
    X_pred = np.dot(codes, components)
    for scorer in ["r2", "pearsonr"]:
        scores = score_multioutput(X_true.T, X_pred.T, scorer=scorer)
        artefacts[scorer].append(scores)


prox = ProximalOperator(which="graph-net", affine=mask_img.affine, fwhm=2,
                        mask=mask_img.get_data().astype(bool), l1_ratio=.1,
                        kernel="gaussian", radius=10.)
model = ProximalfMRIMiniBatchDictionaryLearning(
    n_components=n_components, random_state=random_state,
    fit_algorithm=partial(_enet_coder, l1_ratio=0., alpha=1.),
    dict_penalty_model=-1, mask=mask_img, n_epochs=n_epochs, callback=callback,
    batch_size=batch_size, dict_alpha=dict_alpha, n_jobs=n_jobs, verbose=1)
model.fit(X)

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map


def plot_dico(model):
    for c in range(0, n_components):
        if c == 3:
            continue
        plot_stat_map(unmask(model.components_[c], mask_img), display_mode="ortho",
                      colorbar=False, black_bg=True)
        out_file = "%s_comp%02i.png" % (dataset.replace(" ", "_"), c)
        plt.savefig(out_file, dpi=200, bbox_inches="tight", pad_inches=0)
        os.system("mogrify -trim %s" % out_file)
        print(out_file)

