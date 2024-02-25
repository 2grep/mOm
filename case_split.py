import numpy as np
import lib
import matplotlib.pyplot as plt
from ridge import ks_flat as opaks

# * Create the histograms of, for example, 18-, 23-, 28-, and 33-Cases
# * As seen on the slide for a few representative distributions for certain case counts occuring across OPA bins for 6 observers (slide 69 as of 2024-02-24)

def get_args():
    args = {}
    args["colors"] = ["red", "green"]
    args["datapaths"] = [
        "./data/prostate_reader/assisted_5class.npy",
        "./data/prostate_reader/unassisted_5class.npy"
    ]
    args["graphing"] = {
        "scale": 2.5,
        "ratio": (1, 1),
    }

    args["is_case_centered"] = False
    args["case_step"] = 1
    args["num_graphs"] = 4
    args["central_case"] = 43

    return args

def get_datasets(data_paths):
    return np.asarray([
        np.transpose(
            lib.data_reader(path)
        )
    for path in data_paths])

def main():    
    args = get_args()

    datasets = get_datasets(args["datapaths"])

    ## Setup `choices`
    obs_choice = 6 - 2
    choices = []
    if args["is_case_centered"]:
        bounds = (
            args["central_case"] - args["case_step"] * (args["num_graphs"] // 2), 
            args["central_case"] + args["case_step"] * (args["num_graphs"] // 2 + 1)
        )
        choices = np.arange(bounds[0], bounds[1], args["case_step"])
    else:
        choices = np.arange(28, 33 + 1)

    num_choices = len(choices)

    scale = 2.5
    ratio = args["graphing"]["ratio"][1] / args["graphing"]["ratio"][0]
    fig, axs = plt.subplots(
        ncols = num_choices, 
        figsize = (
            num_choices * args["graphing"]["scale"], 
            ratio * args["graphing"]["scale"]
        )
    )

    ## full_counts = []

    max = -1    # max count for consistent y-axis scaling
    
    res = opaks(datasets[0, obs_choice, choices, :],
                datasets[1, obs_choice, choices, :])
    print("All above certainty:", np.all(res == 1))
    print("choices:", choices)
    print("ks:", res)

    for i in range(len(datasets)):
        dataset = datasets[i]
        color = args["colors"][i]
        obs = dataset[obs_choice]
        cases = obs[choices]

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
        ## full_counts.append(counts)
        
        for j in range(len(choices)):
            ax = axs[j]
            ax.hist(
                bins[:-1],
                bins=bins,
                weights=counts[j],
                range=rng,
                align="left",
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
    plt.savefig("./results/case_split.png",
                bbox_inches="tight", transparent=False, dpi=1000)
    # plt.show()

if __name__ == "__main__":
    main()