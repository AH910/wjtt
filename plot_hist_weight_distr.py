import matplotlib.pyplot as plt
import numpy as np

from utils import get_weights

# Set font for plot to Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"

# Weighting parameters
alpha = 3
rho = 0.5
JTT_upweight = 6

# Choose dataset
dataset = "MultiNLI"

# Load error set
error_set = "./error_sets/" + dataset + "/err_set.csv"
probs = list(np.genfromtxt(error_set, delimiter=","))
# For MultiNLI error_set is 2 arrays (probabilities is the first one)
if dataset == "MultiNLI":
    misclassified = probs[1]
    probs = probs[0]

# Get weights for JTT and WJTT
wjtt_weights = get_weights(probs, alpha, rho, "DRO2")
if dataset == "MultiNLI":
    jtt_weights = [int(alpha + 1) if m == 1 else 1 for m in misclassified]
else:
    jtt_weights = get_weights(probs, JTT_upweight, 0, "JTT")

# Define arrays with bins and weight count
max_weight = max([max(jtt_weights), max(wjtt_weights)])
wjtt_w = range(1, max_weight + 1)  # bins = [1,2,3,...,max_weight]
wjtt_freq = [wjtt_weights.count(w) for w in wjtt_w]  # frequency of weight
jtt_w = range(1, max_weight + 1)
jtt_freq = [jtt_weights.count(w) for w in wjtt_w]


# Make plot
plt.bar(
    jtt_w,
    jtt_freq,
    bottom=wjtt_freq,
    label="JTT",
    color="orange",
    edgecolor="black",
    linewidth=1.2,
)
plt.bar(
    wjtt_w, wjtt_freq, label="WJTT", color="royalblue", edgecolor="black", linewidth=1.2
)

plt.ylabel("Frequency")
plt.xlabel("Weight")
# X ticks
if dataset == "waterbird":
    x_ticks = [1, 10, 20, 101]
elif dataset == "celebA":
    x_ticks = [1, 10, 20, 30, 40, 51]
elif dataset == "MultiNLI":
    x_ticks = [1, 5, 10, 15, 20]
elif dataset == "CivilComments":
    x_ticks = [1, 5, 10, 15, 20, 25, 30, 35]
plt.xticks(x_ticks)
plt.yscale("log")
plt.legend()
plt.savefig("histogram_weights_" + dataset + ".png", dpi=300)
# plt.savefig("histogram_weights_" + dataset + ".eps", dpi=300)
