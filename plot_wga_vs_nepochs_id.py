import matplotlib.pyplot as plt

# Script to generate Figure 2 and 5

# Set font for plot Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"

# Dataset (waterbird or celebA)
dataset = "waterbird"

if dataset == "celebA":
    wga_WJTT = [67.8, 58.9, 56.7, 50.6, 42.2]
    wga_JTT = [83.2, 83.0, 81.1, 77.2, 80]

    wga_gDRO = [88.9] * 5
    wga_ERM = [47.2] * 5

    n_epochs = [0.5, 0.75, 1, 1.25, 1.5]


if dataset == "waterbird":
    wga_WJTT = [82.2, 83.8, 83.0, 78.8, 78.2, 76.5]
    wga_JTT = [66.7, 88.3, 87.7, 84.7, 86.3, 84.1]

    wga_gDRO = [91.4] * 6
    wga_ERM = [72.6] * 6

    n_epochs = [20, 40, 60, 80, 100, 120]

plt.plot(n_epochs, wga_gDRO, "--", label="gDRO", color="green")
plt.plot(n_epochs, wga_JTT, marker="o", label="JTT", color="orange")
plt.plot(n_epochs, wga_WJTT, marker="o", label="WJTT", color="royalblue")
plt.plot(n_epochs, wga_ERM, "--", label="ERM", color="red")

plt.xlabel("Number of epochs (identification model)")
plt.ylabel("Worst-group accuracy")
plt.legend()
if dataset == "waterbird":
    plt.ylim((50, 100))
elif dataset == "celebA":
    plt.ylim((35, 100))
    plt.xticks(n_epochs)


plt.savefig("wga_vs_num_ep_idmodel_" + dataset + ".png", dpi=300)
plt.savefig("wga_vs_num_ep_idmodel_" + dataset + ".eps", dpi=300)
