import numpy as np
import lib
import matplotlib.pyplot as plt

root = "./data/prostate_reader/"
group = "_5class"
datasets = ["assisted", "unassisted"]
colors = ["red", "green"]
datasets = [np.transpose(lib.data_reader(root + dataset + group + ".npy")) for dataset in datasets]

step = 5
num_graphs = 4 + 1
interest = 28
bounds = (interest - step * (num_graphs // 2), interest + step * (num_graphs // 2))
choices = np.arange(bounds[0], bounds[1], step)
num_choices = len(choices)
scale = 2.5
ratio = 1
fig, axs = plt.subplots(ncols=num_choices, figsize=(num_choices * scale, ratio*scale))
max = -1
for i in range(len(datasets)):
    dataset = datasets[i]
    color = colors[i]
    obs6 = dataset[5]
    cases = obs6[choices]

    rng = (0, 1)
    def _hist(x):
        global bins
        hist, bins = np.histogram(
            x,
            bins=30,
            range=rng
        )
        return hist
    counts = np.apply_along_axis(_hist, 1, cases)
    
    for j in range(len(choices)):
        axs[j].hist(
            bins[:-1],
            bins=bins,
            weights=counts[j],
            range=rng,
            color=color,
            alpha=.5
        )
    test = np.amax(counts)
    max = max if max > test else test

for i in range(len(axs)):
    ax = axs[i]
    ax.set_ylim(0, max)
    ax.set_title(str(choices[i]) + " Cases")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(
        left=False,
        labelleft=False)

ax = axs[0]
ax.spines[["left"]].set_visible(True)
ax.yaxis.set_tick_params(
    left=True,
    labelleft=True)

plt.tight_layout()
plt.show()