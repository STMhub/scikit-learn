import glob
import os
root = os.environ.get("ROOT", "/")
for zmap in sorted(glob.glob(root, "/store/data/ibc/derivatives/sub-*/ses-*/*/stat_maps/*.nii.gz")):
    dst = zmap.replace("drago/storage", "drago/storage/tompouce/dohmatob/drago/storage")
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    os.system("ln -s %s %s" % (zmap, dst))
    print("%s ==> %s" % (zmap, dst))
