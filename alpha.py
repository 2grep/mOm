import matplotlib
import numpy as np
import scipy.stats as stats
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

def fit(data: np.ndarray) -> stats.rv_continuous:
    """
    Fit data to beta distribution. Returns `None` if if gets an `Exception`
    """
    try:
        params = stats.beta.fit(data)
        return stats.beta(*params)
    except Exception as e:
        print("Failed fit, returning `None`: ", e)
        return None

def compare(
    assisted: np.ndarray[stats.rv_continuous], 
    unassisted: np.ndarray[stats.rv_continuous], 
    alpha_error: float = .05
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''
    Compare unassisted and assisted to get cutoff and beta
    '''
    try:
        cutoff = unassisted.ppf(1 - alpha_error)
        beta_error = assisted.cdf(cutoff)
        return (cutoff, beta_error)
    except Exception as e:
        return None

def test(datasets) -> tuple[np.ndarray[stats.rv_continuous], list[tuple[int]], float]:
    """
    Functionally equivalent (though slower) to just applying fit along the axes of datasets but gives more information about failure and timing
    """
    start = time.time()
    fits = np.empty_like(datasets[..., 0], dtype=stats.rv_continuous)
    fails = []
    for i in range(datasets.shape[0]):
        for j in range(datasets.shape[1]):
            for k in range(datasets.shape[2]):
                try:
                    fits[i, j, k] = fit(datasets[i, j, k])
                except Exception as e:
                    print(f"Fail on ({i}, {j}, {k}): {e}")
                    fails.append((i, j, k))
    end = time.time()
    return (fits, fails, end - start)


def get_args(data_root: str):
    args = {}
    args["res"] = {}
    args["data_paths"] = (
        f"./data/{data_root}/assisted_5class.npy",
        f"./data/{data_root}/unassisted_5class.npy",
    )
    args["results_directory"] = f"./results/{data_root}/"
    args["graphing"] = {
        "colors": ["red", "blue"],
        "hatches": ["//", "\\\\"],
        "dims": (4, 6),     # Counts of rows and cols; Want observers x cases
        "ratio": (2, 1),    # Horizontal vs. vertical aspect ratio
        "scale": 1.4,
        "histogram_bins": 30,
        "samples": 500,     # How many sample from the theoretical ditribution to take in graphing (basically resolution)
        "is_cdf": True     # We could either analyze as CDF or PDF
    }
    args["alpha_error"] = .05
    return args

def get_data(data_paths: list[str]):
    print("Fetching data...")
    datasets = [np.load(data) for data in data_paths]
    datasets = np.transpose(np.asarray(datasets), (0, 3, 2, 1)) # (assisted/unassisted, observers, cases, surfaces)
    print("Got datasets")
    return datasets

def run_fits(datasets):
    print("Running fits...")

    #                                                 Use one (1) of these (a b)
    # res["test"] = test(datasets)                  # -+
    #                                                  |- a -
    # betas = res["test"][0]                        # -+
    betas = np.apply_along_axis(fit, -1, datasets)   # --- b -

    print("Fits finished.")
    return betas

def get_graphs(
        datasets,
        betas,
        args: dict
) -> str:
    ## * Initialize counters for rows and cols
    obs = np.flip(
        np.linspace(
            datasets.shape[1] - 1, 0, 
            num=args["dims"][0],
            endpoint=False,
            dtype=int
        )
    ) # obs should count up
    cases = np.linspace(
        datasets.shape[2] - 1, 0,
        num=args["dims"][1],
        endpoint=False,
        dtype=int
    ) # cases should count down (to match the top-left [0, 0] of axs)

    ## * Create figure
    fig, axs = plt.subplots(
        ncols=args["dims"][0], 
        nrows=args["dims"][1], 
        figsize=(args["ratio"][0] * args["scale"] * args["dims"][0], args["ratio"][1] * args["scale"] * args["dims"][1]),
        layout="constrained"
    )

    ## * Various tracking variables
    x = np.linspace(0, 1, num=args["samples"])
    constant_indices = True

    ## * Plot out histogram and function
    with np.nditer(
        axs,
        flags=[
            "multi_index",
            "refs_ok"
        ],
        op_flags=[
            "readwrite"
        ],
        op_dtypes=matplotlib.axes._axes.Axes
    ) as it:
        for ax in it:
            ax = ax.item()

            ind = (
                obs[it.multi_index[1]], 
                cases[it.multi_index[0]]
            )
            print(it)
            # Adjust indices to skip None in betas
            while any([betas[g, *ind] == None for g in range(datasets.shape[0])]):
                # increment cases if needed
                ind = (ind[0], ind[1] + 1)
                constant_indices = False

            ax.set_xlabel(ind[0] + 1)
            ax.set_ylabel(ind[1] + 1)
            for group in range(datasets.shape[0]):
                # Finally, graph emperical (hist) and theoretical (plot) pdfs for each SARAPE surface
                group_ind = (group, *ind)
                mpl.rcParams["hatch.color"] = args["colors"][group]

                data = datasets[group_ind[0], group_ind[1], group_ind[2]]
                edges = np.append(np.sort(data), 1)
                vals = np.arange(data.size) / float(data.size)

                ax.stairs(
                    vals,
                    edges,
                    hatch=args["hatches"][group],
                    color=args["colors"][group]
                )
                # hist = ax.hist(
                #     datasets[group_ind[0], group_ind[1], group_ind[2]], 
                #     bins=args["histogram_bins"], 
                #     range=(0, 1), 
                #     density=True, # Draw normalized so area == 1
                #     align="mid",
                #     cumulative=args["is_cdf"], # Draw as CDF
                #     # color=args["colors"][group],
                #     # alpha=.5,
                #     fill=False,
                #     hatch=args["hatches"][group]
                # )[0]

                if args["is_cdf"]:
                    y = betas[group_ind[0], group_ind[1], group_ind[2]].cdf(x)
                else:
                    y = betas[group_ind[0], group_ind[1], group_ind[2]].pdf(x)

                ax.plot(
                    x, y,
                    color=args["colors"][group]
                )

    ## * Clear out excess axes display elements
    with np.nditer(
        axs,
        flags=[
            "multi_index",
            "refs_ok"
        ],
        op_flags=[
            "readwrite"
        ],
        op_dtypes=matplotlib.axes._axes.Axes
    ) as it:
        bottom_row = args["dims"][1] - 1
        left_col = 0
        for ax in it:
            # For some god forsaken reason, ax is a zero-dimensional array which python just DOESN'T want to treat right
            # We use `ndarray.item()` to get the Python scalar value
            ax = ax.item()
            is_bottom_row = it.multi_index[0] == bottom_row
            is_left_col = it.multi_index[1] == left_col
            ax.tick_params(
                axis="both",
                which="both",
                top=False,
                bottom=is_bottom_row,
                left=False,
                right=False,
                labeltop=False,
                labelbottom=is_bottom_row,
                labelleft=False,
                labelright=False,
            )
            
            if constant_indices:
                if not is_left_col:
                    ax.set_ylabel("")
                if not is_bottom_row:
                    ax.set_xlabel("")

    ## * Labeling
    graph_form = "CDF" if args["is_cdf"] else "PDF"
    fig.suptitle(f"Various SARAPE Emperical vs. Theoretical {graph_form}s")
    fig.supxlabel("Observer Count")
    fig.supylabel("Case Count")
    return graph_form

def run_kstest(betas, datasets):
    def unwrapped_kstest(x: np.ndarray[stats.rv_continuous, float]):
        '''
        Apply single-sided kstest from NDArray. Necessary for using apply_along_axis
        since we can't otherwise use two parallel arrays

        x : NDArray
            Of the form `[stats.rv_continuous, float64, ..., float64]`
        
        Returns
        -------
        KstestResult (see `scipy.stats`)
        '''
        beta = x[0]
        if beta == None:
            return None
        data = x[1:].astype('f')
        return stats.kstest(data, beta.cdf)

    combo = np.concatenate((betas[..., np.newaxis], datasets), axis=-1)
    return np.apply_along_axis(unwrapped_kstest, -1, combo)

def run_emperical_vs_theoretical_comparison(
        betas,
        datasets,
        alpha_error: float
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns
    -------
    Tuple of, in order, theoretical beta and cutoff along with difference between emperical and theoretical
    '''
    print("Running cutoffs...", flush=True)
    theoretical = np.apply_along_axis(
        lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error),
        0, 
        betas, 
        alpha_error=alpha_error
    )
    t_beta = theoretical[1]

    # Getting emperical beta values
    def _eppf(p: float, sorted_data: list[float]) -> float:
        return sorted_data[int(p * len(sorted_data))]
    eppf = np.vectorize(
        _eppf,
        signature="(),(n)->()"
    )

    def _ecdf(x: float, sorted_data: list[float]) -> float:
        return np.searchsorted(sorted_data, x) / len(sorted_data)
    ecdf = np.vectorize(
        _ecdf,
        signature="(),(n)->()"
    )

    datasets = np.sort(datasets)
    assisted = datasets[0]
    unassisted = datasets[1]
    cutoff = eppf(1 - alpha_error, unassisted)
    e_beta = ecdf(cutoff, assisted)
    beta_diff = e_beta - t_beta
    emperical = np.stack([cutoff, e_beta], axis=0)
    return (theoretical, emperical, beta_diff)

def save(
        directory: str, 
        betas,
        theoretical, 
        emperical,
        beta_difference, 
        graph_form: str
):
    print("saving...", flush=True)
    np.savetxt(directory + "theoretical_cutoffs.csv", theoretical[:, :, 0], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "theoretical_beta.csv", theoretical[:, :, 1], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "emperical_cutoffs.csv", emperical[:, :, 0], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "emperical_beta.csv", emperical[:, :, 1], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "e-t_beta_diff.csv", beta_difference, delimiter=",", fmt="%.3f")
    np.save(directory + "pure_betas.npy", betas)
    plt.savefig(
        directory + graph_form + ".png",
        bbox_inches="tight",
        transparent=False,
        dpi=1000
    )
    plt.show()

def main():
    print("starting...")

    args = get_args("prostate_reader")
    res = args["res"]

    ## * Fetch datasets
    datasets = get_data(args["data_paths"])

    ## * Fit each sample for observers x cases
    # res["betas"] = run_fits(datasets)
    res["betas"] = np.load(args["results_directory"] + "pure_betas.npy", allow_pickle=True)
    betas = res["betas"]


    ## * Visual confirmation of emperical to theoretical
    graph_form = get_graphs(datasets, betas, args["graphing"])


    ## * Kolmogrov-Smirnov to check theoretical validity within p = .05
    # res["kstest"] = run_kstest(betas, datasets)


    ## * Beta comparison
    res["theoretical"], res["emperical"], res["beta_diff"] = run_emperical_vs_theoretical_comparison(
        betas, 
        datasets, 
        args["alpha_error"]
    )


    ## * Save results
    save(
        args["results_directory"], 
        betas,
        np.transpose(res["theoretical"]), 
        np.transpose(res["emperical"]), 
        np.transpose(res["beta_diff"]), 
        graph_form
    )

if __name__ == '__main__':
    main()