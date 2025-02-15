# import matplotlib # ? mpl.axes._axes.Axes alpha.py:153
import numpy as np
from typing import Optional, Any, TypeVar
import numpy.typing as npt
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# Attempts at typing were made, it turns out to be very complicated
FrozenRV = np.object_
T = TypeVar("T")
NPOptional = T | np.object_

def fit(data: npt.NDArray[np.float_]) -> Optional[npt.NDArray[FrozenRV]]:
    """
    Fit data to beta distribution.

    Parameters: data : Numpy NDArray of floats
    Returns: Numpy NDArray of `stats.rv_frozen` or `None`
        Returns `None` instead of raising for exceptions
    """
    try:
        params = stats.beta.fit(data)
        return stats.beta(*params) # type: ignore
    except Exception as e:
        print("Failed fit, returning `None`: ", e)
        return None

def compare(assisted, unassisted, alpha_error: float = .05) -> Optional[tuple[float, float]]:
    '''
    Compare unassisted and assisted to get cutoff and beta

    Parameters: assisted : `stats.rv_frozen`
                unassisted : `stats.rv_frozen`
                alpha_error : `float`
    Returns: tuple of floats or `None`
        Returns `None` instead of raising for exceptions
    '''
    try:
        cutoff: float = unassisted.ppf(1 - alpha_error)
        beta_error: float = assisted.cdf(cutoff)
        return (cutoff, beta_error)
    except Exception as _:
        return None

# ! I believe this is deprecated
# def test(datasets) -> tuple[DistributionsArray, list[tuple[int]], float]:
#     """
#     Functionally equivalent (though slower) to just applying fit along the axes of datasets but gives more information about failure and timing
#     """
#     start = time.time()
#     fits = np.empty_like(datasets[..., 0], dtype=stats.rv_continuous)
#     fails = []
#     for i in range(datasets.shape[0]):
#         for j in range(datasets.shape[1]):
#             for k in range(datasets.shape[2]):
#                 try:
#                     fits[i, j, k] = fit(datasets[i, j, k])
#                 except Exception as e:
#                     print(f"Fail on ({i}, {j}, {k}): {e}")
#                     fails.append((i, j, k))
#     end = time.time()
#     return (fits, fails, end - start)


def get_args(data_root: str) -> dict[str, Any]:
    args = {}
    args["res"] = {}

    # ! Can we just scrape the directory for all .npy files? And/or .csv?
    # This was just the fastest way to test
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
        "is_cdf": True      # We could either analyze as CDF or PDF
    }
    args["alpha_error"] = .05
    return args

def get_data(data_paths: list[str]) -> npt.NDArray[Any]:
    '''
    Get the CONTEST surfaces from the analysis's .npy file
    '''
    print("Fetching data...")
    datasets = [np.load(data) for data in data_paths]
    datasets = np.transpose(np.asarray(datasets), (0, 3, 2, 1)) # (assisted/unassisted, observers, cases, surfaces)
    print("Got datasets")
    return datasets

def run_fits(datasets: npt.NDArray[np.float_]) -> npt.NDArray[FrozenRV]:
    print("Running fits...")

    #                                                 Use one (1) of these (a b)
    # res["test"] = test(datasets)                  # -+
    #                                                  |- a -
    # betas = res["test"][0]                        # -+
    # ? What is -1 axis?
    betas: npt.NDArray[FrozenRV] = np.apply_along_axis(fit, -1, datasets) # type: ignore  # --- b -

    print("Fits finished.")
    return betas

def get_graphs(datasets, betas, args: dict) -> str:
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
        figsize=(args["ratio"][0] * args["scale"] * args["dims"][0], 
                args["ratio"][1] * args["scale"] * args["dims"][1]),
        layout="constrained"
    )

    ## * Various tracking variables
    x = np.linspace(0, 1, num=args["samples"]) # * [0, 1] bc X is the proportion of agreement
    constant_indices = True

    ## * Plot out histogram and function
    with np.nditer(
        axs,
        flags=[
            "multi_index",
            "refs_ok"
        ],
        op_flags=[
            "readwrite" # type: ignore # !
        ],
        op_dtypes=mpl.axes._axes.Axes # type: ignore # !
    ) as it:
        for ax in it:
            ax = ax.item() # type: ignore # !

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
                # Finally, graph emperical (hist) and theoretical (plot) pdfs for each CONTEST surface
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
                y = betas[group_ind[0], group_ind[1], group_ind[2]]
                if args["is_cdf"]:
                    y = y.cdf(x)
                else:
                    y = y.pdf(x)

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
            "readwrite" # type: ignore # !
        ],
        op_dtypes=mpl.axes._axes.Axes # type: ignore # !
    ) as it:
        bottom_row = args["dims"][1] - 1
        left_col = 0
        for ax in it:
            # For some god forsaken reason, ax is a zero-dimensional array which python just DOESN'T want to treat right
            # We use `ndarray.item()` to get the Python scalar value
            ax = ax.item() # type: ignore # !
            is_bottom_row = it.multi_index[0] == bottom_row
            is_left_col = it.multi_index[1] == left_col
            ax.tick_params(
                axis="both",
                which="both",
                top=False,
                bottom=is_bottom_row,
                left=is_left_col,
                right=False,
                labeltop=False,
                labelbottom=is_bottom_row,
                labelleft=is_left_col,
                labelright=False,
            )
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            if constant_indices:
                if not is_left_col:
                    ax.set_ylabel("")
                if not is_bottom_row:
                    ax.set_xlabel("")

    ## * Labeling
    graph_form = "CDF" if args["is_cdf"] else "PDF"
    fig.suptitle(f"Various CONTEST Emperical vs. Theoretical {graph_form}s")
    fig.supxlabel("Observer Count")
    fig.supylabel("Case Count")
    return graph_form

def run_kstest(betas: npt.NDArray[FrozenRV], datasets: npt.NDArray[np.float_]) -> npt.NDArray[NPOptional[np.float_]]:
    '''
    Parameters: betas : ndarray of `rv_frozen`
                datasets : ndarray of `float`
    Returns: ndarray of (`float` or `None`)
        `None` occurs where `betas` had `None`
    '''
    def unwrapped_kstest(x: npt.NDArray) -> Optional[np.float_]:
        '''
        Apply single-sided kstest from NDArray. Necessary for using apply_along_axis
        since we can't otherwise use two parallel arrays

        x : NDArray
            Of the form `[stats.rv_continuous, float64, ..., float64]`
        
        Returns: `float` or `None`
            KstestResult pvalue (see `scipy.stats`)
            Returns `None` if the `stats.rv_continuous` is `None`
        '''
        beta = x[0]
        if beta == None:
            return None
        data = x[1:].astype('f')
        return stats.kstest(data, beta.cdf).pvalue

    # Reduce the 10,000 surfaces to 25 random surfaces for each (d, o, c)
    # I am doing this to reduce n in kstest for more liberal p-values
    datasets = np.apply_along_axis(np.random.choice, -1, datasets, 50, replace=False)
    combo = np.concatenate((betas[..., np.newaxis], datasets), axis=-1)
    return np.apply_along_axis(unwrapped_kstest, -1, combo) # type: ignore

def run_emperical_vs_theoretical_comparison(
    betas: npt.NDArray[FrozenRV],
    datasets: npt.NDArray[np.float_],
    alpha_error: float
) -> tuple[npt.NDArray[NPOptional[np.float_]], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    '''
    Returns
    -------
    Tuple of, in order, theoretical beta and cutoff along with difference between emperical and theoretical
    '''
    print("Running cutoffs...", flush=True)
    theoretical = np.apply_along_axis(
        lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error), # type: ignore
        0, 
        betas,  
        alpha_error=alpha_error
    )
    t_beta: npt.NDArray[np.float_] = theoretical[1]

    # Getting emperical beta values
    def _eppf(p: float, sorted_data: npt.NDArray[np.float_]) -> np.float_:
        return sorted_data[int(p * len(sorted_data))]
    eppf = np.vectorize(
        _eppf,
        signature="(),(n)->()"
    )

    def _ecdf(x: float, sorted_data: list[float]) -> np.float_:
        return np.searchsorted(sorted_data, x) / len(sorted_data)
    ecdf = np.vectorize(
        _ecdf,
        signature="(),(n)->()"
    )

    datasets = np.sort(datasets)
    assisted: npt.NDArray[np.float_] = datasets[0]
    unassisted: npt.NDArray[np.float_] = datasets[1]
    cutoff: npt.NDArray[np.float_] = eppf(1 - alpha_error, unassisted)
    e_beta: npt.NDArray[np.float_] = ecdf(cutoff, assisted)
    beta_diff = e_beta - t_beta
    emperical = np.stack([cutoff, e_beta], axis=0)
    return (theoretical, emperical, beta_diff)

def save(
    directory: str,
    graph_form: str,
    betas: npt.NDArray,
    theoretical: npt.NDArray,
    emperical: npt.NDArray,
    beta_difference: npt.NDArray,
    kstest: npt.NDArray | None = None,
):
    # ! Make folder if not there

    print("saving...", flush=True)
    np.save(directory + "pure_betas.npy", betas)
    np.savetxt(directory + "theoretical_cutoffs.csv", theoretical[:, :, 0], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "theoretical_beta.csv", theoretical[:, :, 1], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "emperical_cutoffs.csv", emperical[:, :, 0], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "emperical_beta.csv", emperical[:, :, 1], delimiter=",", fmt="%.3f")
    np.savetxt(directory + "e-t_beta_diff.csv", beta_difference, delimiter=",", fmt="%.3f")
    if not kstest is None:
        np.savetxt(directory + "kstest_assisted.csv", kstest[0, ...], delimiter=",", fmt="%.3f")
        np.savetxt(directory + "kstest_unassisted.csv", kstest[1, ...], delimiter=",", fmt="%.3f")
    plt.savefig(
        directory + graph_form + ".png",
        bbox_inches="tight",
        transparent=False,
        dpi=1000
    )
    plt.show()

def main():
    print("starting...")

    global args
    args = get_args("prostate_reader")
    res = args["res"]

    ## * Fetch datasets
    datasets = get_data(args["data_paths"]) # (assisted/unassisted, observers, cases, surfaces) (aka. (d, o, c, s))

    ## * Fit each sample for observers x cases
    res["betas"] = run_fits(datasets)
    # res["betas"] = np.load(args["results_directory"] + "pure_betas.npy", allow_pickle=True) # load from precomputed values for faster testing
    betas = res["betas"]


    ## * Visual confirmation of emperical to theoretical
    print("Getting graphs...")
    graph_form = get_graphs(datasets, betas, args["graphing"])


    ## * Kolmogrov-Smirnov to check theoretical validity within p = .05
    print("Running Kolmogrov-Smirnov tests...")
    res["kstest"] = run_kstest(betas, datasets)


    ## * Beta comparison
    res["theoretical"], res["emperical"], res["beta_diff"] = run_emperical_vs_theoretical_comparison(
        betas,
        datasets,
        args["alpha_error"]
    )


    ## * Save results
    save(
        args["results_directory"],
        graph_form,
        args["res"]["betas"],
        np.transpose(args["res"]["theoretical"]),
        np.transpose(args["res"]["emperical"]),
        np.transpose(args["res"]["beta_diff"]),
        np.transpose(args["res"]["kstest"], (0, 2, 1))
    )

if __name__ == '__main__':
    main()