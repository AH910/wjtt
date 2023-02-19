import matplotlib.pyplot as plt
import numpy as np

n_epochs = np.linspace(0.1, 4, 40)

val_wga_jtt = [
    0.20,
    18.34,
    45.92,
    59.29,
    63.97,
    67.38,
    70.40,
    70.13,
    70.51,
    72.86,
    71.90,
    71.96,
    73.73,
    71.72,
    74.79,
    76.17,
    74.62,
    75.57,
    74.14,
    75.62,
    75.55,
    78.93,
    77.96,
    76.12,
    77.72,
    76.53,
    78.11,
    78.06,
    78.32,
    78.30,
    80.07,
    80.29,
    78.39,
    81.12,
    79.77,
    83.36,
    80.69,
    81.36,
    81.52,
    81.51,
]

test_wga_jtt = [
    0.28,
    22.52,
    51.58,
    65.18,
    69.81,
    72.89,
    75.67,
    75.43,
    75.80,
    77.58,
    76.80,
    77.02,
    78.47,
    76.83,
    79.27,
    80.36,
    79.02,
    79.84,
    78.86,
    79.94,
    79.74,
    82.44,
    81.85,
    80.34,
    81.53,
    80.67,
    82.14,
    81.88,
    82.14,
    82.08,
    81.67,
    83.89,
    82.34,
    80.56,
    83.43,
    81.11,
    83.33,
    81.67,
    82.22,
    80.00,
]

val_avgacc_wjtt = [
    77.27,
    85.16,
    86.84,
    89.72,
    90.12,
    89.64,
    90.41,
    91.39,
    91.10,
    92.01,
    91.71,
    91.99,
    92.56,
    92.89,
    92.02,
    92.25,
    92.19,
    92.54,
    92.43,
    92.59,
    92.63,
    93.08,
    92.39,
    93.43,
    93.03,
    93.22,
    93.54,
    93.15,
    93.91,
    92.99,
    93.97,
    94.29,
    93.94,
    94.06,
    94.21,
    94.50,
    94.36,
    94.19,
    94.10,
    94.38,
]

test_wga_wjtt = [
    75.33,
    76.11,
    75.56,
    67.78,
    69.44,
    74.44,
    74.44,
    72.78,
    76.11,
    72.22,
    76.67,
    76.11,
    72.78,
    72.78,
    75.56,
    74.44,
    75.56,
    75.00,
    74.44,
    75.56,
    74.44,
    72.22,
    76.11,
    71.67,
    73.89,
    72.22,
    69.44,
    72.78,
    67.78,
    73.33,
    66.11,
    59.44,
    64.44,
    62.78,
    58.33,
    56.67,
    56.11,
    57.22,
    57.78,
    51.11,
]

early_stop_jtt = 3.6
early_stop_wjtt = 3.6

fig, axs = plt.subplots(2, 1)
axs[0].plot(n_epochs, val_wga_jtt, label="Val. worst-group acc.", color="orange")
axs[0].plot(n_epochs, test_wga_jtt, label="Test worst-group acc.", color="royalblue")
axs[0].set_ylim([0, 100])
axs[0].title.set_text("JTT")
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].set_ylabel("Accuracy")
axs[0].axvline(early_stop_jtt, color="red", linestyle="dashed")
axs[0].legend(loc="lower center")

axs[1].plot(n_epochs, val_avgacc_wjtt, label="Val. average acc.", color="orange")
axs[1].plot(n_epochs, test_wga_wjtt, label="Test worst-group acc.", color="royalblue")
axs[1].set_ylim([0, 100])
axs[1].title.set_text("WJTT")
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Number of epochs (second model)")
axs[1].axvline(early_stop_wjtt, color="red", linestyle="dashed")
axs[1].legend(loc="lower center")

plt.subplots_adjust(hspace=0.8)
plt.tight_layout()
plt.savefig("wga", dpi=300)
