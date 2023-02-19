import matplotlib.pyplot as plt
import numpy as np

from utils import get_weights

# Script to generate Figure 1

# Set font for plot Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"

# Parameters of JTT and WJTT
alpha = 8
rho = 1
JTT_upweight = 50

# probs = [0.001,0.002,...,1]
probs = np.linspace(0.001, 1, 1000)

# Get weights
dro_weights = get_weights(probs, alpha, rho, "DRO2")
jtt_weights = get_weights(probs, JTT_upweight, 0, "JTT")

# Make plot
plt.plot(probs, dro_weights, label="DRO")
plt.plot(probs, jtt_weights, label="JTT")
plt.xlabel("Probability of true label")
plt.ylabel("Weight")
plt.legend()
plt.savefig(f"JTT_ups{JTT_upweight}_vs_DRO_a{alpha}_r{rho}.png", dpi=300)
plt.savefig(f"JTT_upw{JTT_upweight}_vs_DRO_a{alpha}_r{rho}.eps", dpi=300)
