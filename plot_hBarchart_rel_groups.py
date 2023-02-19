import matplotlib.pyplot as plt
import numpy as np

from data import dataset_attributes
from utils import get_weights

# Script to generate Figure 3
# Takes about 6min

# Set font for plot to Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"

# Parameters for WJTT and JTT
alpha = 3
rho = 0.5
JTT_upweight = 6

# Load dataset to get group labels
dataset = "CivilComments"
dataset_attrs = dataset_attributes[dataset]
full_dataset = dataset_attrs["class"]()
train_data, val_data, test_data = full_dataset.split()
groups = [train_data[k][2] for k in range(len(train_data))]

# Load error set
error_set = "./error_sets/" + dataset + "/err_set.csv"
probs = list(np.genfromtxt(error_set, delimiter=","))

# Get weights
wjtt_w = get_weights(probs, alpha, rho, "DRO2")
jtt_w = get_weights(probs, JTT_upweight, 0, "JTT")

# Initialize lists for group counts
n_g = dataset_attrs["n_groups"]  # Number of groups in dataset
groups_org = [0 for k in range(n_g)]  # Original dataset
groups_jtt = [0 for k in range(n_g)]  # JTT dataset
groups_wjtt = [0 for k in range(n_g)]  # WJTT dataset

for k in range(len(groups)):
    g = groups[k]  # get group label
    # Add weight to group counts
    groups_org[g] += 1
    groups_jtt[g] += jtt_w[k]
    groups_wjtt[g] += wjtt_w[k]

# Lists with relative group sizes
rel_groups_org = [g / sum(groups_org) for g in groups_org]
rel_groups_jtt = [g / sum(groups_jtt) for g in groups_jtt]
rel_groups_wjtt = [g / sum(groups_wjtt) for g in groups_wjtt]

# Get group names
group_names = [dataset_attrs[f"group{k}"] for k in range(dataset_attrs["n_groups"])]

# x-axis range on plot
xlim = [0, 0.6]

# Make plot
fig, axs = plt.subplots(3, 1)
axs[0].barh(
    group_names, rel_groups_org, color="royalblue", edgecolor="black", linewidth=1.2
)
axs[0].set_xlim(xlim)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["bottom"].set_visible(False)
axs[0].get_xaxis().set_visible(False)
axs[0].title.set_text("Original training set")
axs[0].axvline(0.25, color="red", linestyle="dashed")

axs[1].barh(
    group_names, rel_groups_jtt, color="orange", edgecolor="black", linewidth=1.2
)
axs[1].set_xlim(xlim)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["bottom"].set_visible(False)
axs[1].get_xaxis().set_visible(False)
axs[1].title.set_text("JTT")
axs[1].axvline(0.25, color="red", linestyle="dashed")

axs[2].barh(
    group_names, rel_groups_wjtt, color="orange", edgecolor="black", linewidth=1.2
)
axs[2].set_xlim(xlim)
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].title.set_text("WJTT")
axs[2].axvline(0.25, color="red", linestyle="dashed")

plt.subplots_adjust(hspace=0.8)
plt.tight_layout()
plt.savefig("hBarchart_" + dataset + "_rel_groups.png", dpi=300)
plt.savefig("hBarchart_" + dataset + "_rel_groups.eps", dpi=300)
