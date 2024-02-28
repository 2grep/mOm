#!usr/bin/env python3
from typing import Any
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from typing import Callable
from types import SimpleNamespace
import lib

# * This does the initial ONEST and CONTEST analyses to create the analyzed manifolds from the original CSV data
# * and to create the graphs of slide for Basic ONEST procedure for Steiner's Reader Study (slide 32 as of 2024-02-24)

## ARGUMENTS ##
def get_args() -> dict[str, Any]:
    return {
        "dataset_names": [
            "./data/prostate_reader/assisted_5class.csv",
            "./data/prostate_reader/unassisted_5class.csv"
        ],
        # Must be valid matplotlib graphing color
        "colors": [
            "tab:blue",
            "tab:red"
        ],
        "labels": [ 
            "assisted",
            "unassisted"
        ],
        "method": "contest",
        "describe": True,
        "cache": True
    }

# args = parser.parse_args()
args = SimpleNamespace(**get_args()) # allows dictionary dot notation bc previous implementation was using that and I didn't want to change the code

file_names = []
file_exts = []

## FUNCTIONS ##
def overall_proportion_agreement(observer_case_matrix: npt.NDArray[Any]) -> float:
    '''
    Parameters
    ----------
    observer_case_matrix : observers O x cases c matrix
        ```
        [[A0, A1, ..., Ac],
         [B0, B1, ..., Bc],
         ...
         [O0, O1, ..., Oc]]
        ```
    
    Returns
    -------
    OPA : proportion of cases with complete observer agreement
    '''
    case_agreements = np.apply_along_axis(lib.all_match, 0, observer_case_matrix) # type: ignore # match along columns
    #  number of case agreements / number of cases
    return case_agreements.sum() / observer_case_matrix.shape[1]

def contest(
    observer_case_matrix: npt.NDArray[Any], 
    unique_surfaces: int
) -> npt.NDArray[np.float_]:
    # TODO: What does "unique" surfaces mean? Define uniqueness.
    '''
    Calculate CONTEST method on `case_observer_matrix` to generate `unique_surfaces` number of samples from
    the full space of samples. 

    Parameters
    ----------
    observer_case_matrix : observers O x cases c matrix
        ```
        [[A0, A1, ..., Ac],
         [B0, B1, ..., Bc],
         ...
         [O0, O1, ..., Oc]]
        ```
    unique_surfaces : number of surfaces to run
        More surfaces is a better sample of the full space but takes longer. "Unique" meaning the orders 
        of rows and columns of case_observer_matrix are independently (i.e. uniqueness in row is unrelated
        to that of columns) unique orderings.
    
    Returns
    -------
    contest : surfaces calculated with a shape of `(unique surfaces, cases, total observers - 1)`;
                does NOT gaurentee a specific order to the final OPAs
    '''
    # Generators for observers and cases

    num_obs = observer_case_matrix.shape[0]
    num_cases = observer_case_matrix.shape[1]
    obs_gen = lib.random_unique_permutations(np.arange(num_obs), call_count=unique_surfaces)
    cases_gen = lib.random_unique_permutations(np.arange(num_cases), call_count=unique_surfaces)

    space = []
    for new_surface in range(unique_surfaces):
        if new_surface % 10 == 0:
            print("Running surface:", new_surface)

        surf_obs = next(obs_gen)
        surf_cases = next(cases_gen)

        # cases x observers
        grid = []
        for case_ind in range(num_cases):
            reduced_cases = surf_cases[:case_ind + 1]

            observer_opas = []
            for obs_ind in range(2, num_obs + 1):
                reduced_obs = surf_obs[:obs_ind]

                # We need to index twice bc of broadcasting isues between the two indices
                matrix = observer_case_matrix[reduced_obs][:, reduced_cases]
                opa = overall_proportion_agreement(matrix)

                observer_opas.append(opa)
            grid.append(observer_opas)
        space.append(grid)
    return np.array(space, copy=False)

def onest(
    observer_case_matrix: npt.NDArray[Any], 
    unique_curves: int
) -> npt.NDArray[np.float_]:
    '''
    Parameters
    ----------
    observer_case_matrix : observers O x cases c matrix
        ```
        [[A0, A1, ..., Ac],
         [B0, B1, ..., Bc],
         ...
         [O0, O1, ..., Oc]]
        ```
    unique_curves : number of curves to runs

    Returns
    -------
    onest : 
    '''
    onest = []
    num_obs = observer_case_matrix.shape[0]
    observers_generator = lib.random_unique_permutations(np.arange(num_obs), call_count=unique_curves)

    for new_curve in range(unique_curves):
        if new_curve % 10 == 0:
            print("Running curve: ", new_curve)

        curve_obs = next(observers_generator)

        ## Generate single onest curve
        curve = []
        for ind in range(2, num_obs + 1):
            # num of observers x OPA point on the ONEST curve
            reduced_obs = curve_obs[:ind]
            matrix = observer_case_matrix[reduced_obs]
            opa = overall_proportion_agreement(matrix)
            curve.append(opa)

        onest.append(curve)
    return np.array(onest, copy=False)

def get_analyzed_data[T: np.generic](
    datasets: npt.NDArray[T],
    analysis_method: Callable[[npt.NDArray[T], int], npt.NDArray[np.float_]],
    unique_manifolds: int,
    datasets_from_cache: bool,
    cache: bool
) -> npt.NDArray[np.float_]:
    '''
    Get analyzed data. Either the datasets were from a cached analysis and we just return that
    or we run the ONEST analysis on the data
    '''
    if datasets_from_cache:
        return datasets # type: ignore
    cache_name_index = 0
    analyses: list[np.ndarray] = []
    for case_observer_matrix in datasets:
        analysis = analysis_method(np.transpose(case_observer_matrix), unique_manifolds)
        analyses.append(analysis)

        if cache:
            np.save(file_names[cache_name_index] + ".npy", analysis)
            cache_name_index += 1

    return np.asarray(analyses)

def data_to_plot(
    onest_analyses: npt.NDArray[np.float_],
    manifolds_axis: int,
    describe: bool
) -> npt.NDArray[np.float_]:
    if describe:
        # Desribe as min, mean, max if desired
        return np.stack((
            np.amin(onest_analyses, axis=manifolds_axis),
            np.average(onest_analyses, axis=manifolds_axis),
            np.amax(onest_analyses, axis=manifolds_axis),
        ), axis=manifolds_axis)
    else:
        return onest_analyses

def select_method():
    args.datasets = np.asarray([lib.data_reader(set, names=file_names, exts=file_exts) for set in args.dataset_names])
    datasets_from_cache = any(ext in file_exts for ext in [".pkl", ".npy"])

    if args.method == "onest":
        run_onest(datasets_from_cache)

    elif args.method == "contest":
        run_contest(datasets_from_cache)

    else:
        print(f"Unknown method '{args.method}'") 

def run_onest(
    datasets_from_cache: bool = True, 
    unique_curves: int = 1000
):
    ## Convert case_observer matrices to OPAs (i.e. One set (each item in dataset_onest_analyses) of curves for each dataset)
    # pyplot.Axes.plot calls for observers to be last
    onest_analyses = np.transpose(
        get_analyzed_data(args.datasets, onest, unique_curves, datasets_from_cache, args.cache),
        (0, 2, 1)
    )
    onest_curves = data_to_plot(onest_analyses, 2, args.describe)

    ## Plot each analysis
    def plot_curves(onest_curves):
        _, ax = plt.subplots()
        obs_range = (2, onest_analyses.shape[1] + 1)
        xs = np.arange(obs_range[0], obs_range[1] + 1)
        for dataset in range(onest_curves.shape[0]):
            ax.plot(
                xs, onest_curves[dataset],
                ("-" if args.describe else "o-"),
                color=args.colors[dataset % len(args.colors)],
                fillstyle="none",
                linewidth=1 if args.describe else .5
            )

        # adjust plot parameters
        ax.xaxis.set_major_locator(MultipleLocator(6))
        ax.xaxis.set_major_formatter('{x:.0f}')
        # should have 3+ observers or this errors
        ax.set_xlim(*obs_range)
        ax.set_xlabel("Number of Observers")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Overall Proportion Agreement")
        ax.legend(ax.get_lines()[::(3 if args.describe else unique_curves)],
                args.labels if args.labels != None else file_names)
        plt.savefig(
            "./results/onest.png",
            bbox_inches="tight",
            transparent=False,
            dpi=1000
        )
        plt.show()
    plot_curves(onest_curves)

def run_contest(
    datasets_from_cache: bool = True, 
    unique_surfaces: int = 10000,
    colors: list = ["coolwarm", "PiYG"]
):
    case_onest_analyses = get_analyzed_data(args.datasets, contest, unique_surfaces, datasets_from_cache, args.cache)
    dataset_surfaces = data_to_plot(case_onest_analyses, 1, args.describe)

    def plot_data(dataset_surfaces):
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        obs_range = (2, dataset_surfaces.shape[3] + 1)
        observers_axis = np.arange(obs_range[0], obs_range[1] + 1)
        cases_range = (1, dataset_surfaces.shape[2])
        cases_axis = np.arange(cases_range[0], cases_range[1] + 1)
        xyplane = np.meshgrid(cases_axis, observers_axis)

        ## Plot
        # We have to draw things backwards to get them layered better
        for dataset in range(dataset_surfaces.shape[0] - 1, -1, -1):
            for surface in dataset_surfaces[dataset, ::-1]:
                ax.plot_surface(
                    *xyplane,
                    np.transpose(surface),
                    cmap=colors[dataset % len(colors)],
                    linewidth=0,
                    antialiased=False,
                    zsort="max"
                )

        # Customize the z axis.
        ax.yaxis.set_major_locator(MultipleLocator(6))
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.set_xlim(*cases_range)
        ax.set_xlabel("Number of Cases")
        ax.set_ylim(*obs_range)
        ax.set_ylabel("Number of Observers")
        ax.set_zlim(0, 1)
        ax.set_zlabel("Overall Proportion Agreement")

        ax.view_init(azim=45, elev=30)
        plt.savefig(
            "./results/contest.png", 
            bbox_inches="tight", 
            transparent=False,
            dpi=1000
        )
        plt.show()
    plot_data(dataset_surfaces)

def main():
    select_method()

if __name__ == "__main__":
    main()