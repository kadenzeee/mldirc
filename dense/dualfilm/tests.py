import numpy as np


f = np.load("500K22TO140dualfilm.npz")

times = f["TIMES"]
hists = f["HISTS"]
angles = f["ANGLES"]
labels = f["LABELS"]


print(sum(hists[0:10]))