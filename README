# Preface
None of the errors herein are mine, the editors are conspiring against me.

# Expected Folders and Setup
- data/prostate_reader/assisted_5class.npy
- data/prostate_reader/unassisted_5class.npy

# Original Data
- data/
    - prostate_reader/
        - assisted_5class.csv
        - unassisted_5class.csv
    - nottingham/
        - mitosis.csv
        - nottingham.csv
        - pleomorphismus.csv
        - tubulus.csv

# Instructions
> Instructions last updated 2024-02-27
> These have not been tested for anything other than comparing exactly 2 sets of data at a time

The analysis starts with `onest_analysis.py` to perform the ONEST and/or CONTEST analyses on the CSV data.
Make sure to cache this to create the appropriate NPYs used in the other programs.

## Getting Started
These instructions assume you have the commands [`git`][git-download] and [`python3.12`][python3.12-download] installed on your machine. If you do not have `python3.12`, you may download it [directly][python3.12-download] or as a standalone program in most package managers. Furthermore, we assume you have a basic understanding of the terminal and navigation therein.

Open a terminal and navigate to the folder you want this repository to be under (for the sake of example, these instructions will call said folder `root`). From `root`, copy this repository with SSH with the following command:
```bash
git clone git@github.com:grepgrok/CONTEST.git
```

or with HTTPS:
```bash
git clone https://github.com/grepgrok/CONTEST.git
```

This should create the following file structure:
```
root/
└── CONTEST/
    ├── .gitignore
    ├── LICENSE
    ├── README
    └── ...
```

[git-download]: https://git-scm.com/downloads
[python3.12-download]: https://www.python.org/downloads/release/python-3122/

### Setup Virtual Environment
While not strictly necessary, it is highly recommended to run the code in this repository under a virtual environment. If you do *not* want to use a venv, skip straight to [Install Dependencies](#-install-dependencies). For more thorough instructions, see [the official `venv` documentation][venv-docs]. Here we give a basic overview of what is needed in a bash shell:
1. Navigate to `root/CONTEST`
2. Create the virtual environment: Run `python3.12 -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Check that it was activated properly: `which pip3` should print `path/to/root/CONTEST/.venv/bin/pip3`

The file structure should resemble the following:
```
root/
└── CONTEST/
    ├── .gitignore
    ├── .venv/
    │   ├── bin/
    │   │   ├── activate
    │   │   ├── pip3
    │   │   ├── python3.12
    │   │   ├── python
    │   │   └── ...
    │   ├── include
    │   ├── lib
    │   └── pyenv.cfg
    ├── LICENSE
    ├── README
    └── ...
```
> Note that file structure diagrams following will be rooted in `CONTEST` and only include the files relevant at that point in the instructions (typically excluding `.venv`)

Once you are done with executing the code in this repository, you may execute the command `deactive` to deactivate the environment. The environment may be reactivated at any time with the command in step (3) above.

### Install Dependencies
From `root/CONTEST`, install the requirements with the following command:
```bash
pip3 install -r requirements.txt
```

In the terminal, navigate to the `CONTEST` folder. From here on, subsequent terminal commans will assume the terminal is currently in the `CONTEST` folder.

[venv-docs]: https://docs.python.org/3/library/venv.html#creating-virtual-environments
## CONTEST/ONEST Analysis
1. Place `.csv` data files in the same folder. This folder should be in the folder `./data/`.
    - For the sake of example, these instructions will call the folder `my_data` and the files `treatment.csv` and `control.csv`.
    > There are a few assumptions made about the datasets that should be ensured ahead of time:
    > 1. They have the same dimensions (e.g. both are 240 cases by 20 observers) with each row being a "case"
    > 2. The file only contains numbers in CSV format. Remove any column and row labels.
    > 3. Remove anything that is otherwise *not* data on how the observers graded the cases. Remove any data on ground truth, model prediction, treatment type, etc. from the `.csv` files.
2. Create a `./results/my_data` folder.
- Here is an example of the file system structure so far:
    ```
    CONTEST/
    ├── data/
    │   └── my_data/
    │       ├── treatment.csv
    │       └── control.csv
    ├── results/
    │   └── my_data/
    ├── README
    ├── onest_analysis.py
    └── ...
    ```
3. Open `onest_analysis.py` and scroll to the `get_args` function on line 15.
    1. Change the `dataset_names` on lines 18 and 19 to the paths to the two files. Be mindful of the fact that Python requires a comma `,` at the end of the first file path on line 18.
        ```python
        "dataset_names": [
            "./data/my_data/treatment.csv",
            "./data/my_data/control.csv"
        ]
        ```
    2. Adjust the `colors` and `labels` on lines 22 through 29 as desired. These accordingly control the color and label associated with `treatment.csv` and `control.csv` on the plot of the analysis.
        > The values under `colors` must be [named matplotlib colors][mpl-colors]. The values under `labels` may be any strings.
    3. Set `method` to `onest` or `contest` for the ONEST or CONTEST analyses accordlingly. Note that the subsequent analyses below require this to be `contest`.
    4. If you would like to plot all manifolds of the analysis, set `describe` on line 31 to `False`. Conversely, setting `describe` to `True` will show only the minimum, maximum, and mean of the envelope in the ONEST method and only the minimum and maximum in the CONTEST method.
    5. Make sure `cache` is set to `True`
    6. Advanced: The number of unique manifolds may be adjusted: change the default value for `unique_curves` on line 220 for ONEST; change `unique_surfaces` on line 265 for CONTEST. Larger numbers may be more accurate but will take longer to compute. See [Thoughts and Notes](#thoughts-and-notes) below for commentary on this.
4. In the terminal, run the following command:
```bash
python onest_analysis.py
```

This may take some time, feel free to get some coffee.

The file system structure should now look something like this:
```
CONTEST/
├── data/
│   └── my_data/
│       ├── treatment.csv
│       ├── treatment.npy
│       ├── control.csv
│       └── control.npy
├── results/
│   ├── my_data/
│   └── onest.png
├── README
├── onest_analysis.py
└── ...
```

Subsequent analyses especially require the prescence of the `.npy` file created by this analysis: `./data/my_data/treatment.npy` and `./data/my_data/control.npy`. Also, they assume the executed analysis was the CONTEST analysis (`"method": "contest"` above in Step 3.3).

[mpl-colors]: https://matplotlib.org/stable/gallery/color/named_colors.html

### Running Already Analyzed Data
Since running the analysis can take a lot of time, you can re-run the plotting after obtaining the cached `.npy` files by simply replacing the `.csv` with `.npy` in the `dataset_names` on lines 18 and 19. The program will skip the analysis part, significantly speeding up graphical analysis.

# Thoughts and Notes
## Random, Unique Permutations
As described in the [original paper][onest-paper], the ONEST/CONTEST manifolds are (theoretically) random and unique permutations of the observers (and cases). This means that, for 20 observers and 240 cases, there should be $19!$ (about $1.2 \times 10^{17}$) ONEST curves. Likewise, there should be $19! \cdot 240!$ (about $4.9 \times 10^{485}$) CONTEST surfaces. In practice, the number of CONTEST surfaces is bounded above by the respective number of ONEST curves. This is because we get the observers and cases for a surface at the same time (see `onest_analysis.py:102`). Also, if the number of unique manifolds is set greater than the factorial of one less than the number of observers, the code is liable to enter an infinite loop trying to find the next set of observers for a surface.

## 3D Graphing
The graphing of the CONTEST analysis can be very odd and glitchy. This is a [documented artifact in matplotlib][mpl-mplot3d-faq] and it is suggested to use [Mayavi] (we have not made this switch).

[mpl-mplot3d-faq]: https://matplotlib.org/3.8.3/api/toolkits/mplot3d/faq.html#my-3d-plot-doesn-t-look-right-at-certain-viewing-angles
[Mayavi]: https://docs.enthought.com/mayavi/mayavi/.

# TODO
- Walk through the data folder for the data
- Rename all instances of sarape to contest if possible
- Do better `get_args`
- Automatically create sarapes in alpha.py:get_data
- Choose consistent way to convert list to ndarray (note when other is necessary)
    - `np.array`
    - `np.asarray`
    - `np.empty` -> fill
- Choose consistent way to execute function over ndarray (`alpha.py:289`, `alpha.py:144`, `alpha.py:144`)
- Choose consistent way to identify assissted/unassisted or treatment/control
- Choose consistent style of docstring
- Figure out a consistent style of execution workflow (or decide to give up on it)
- Add detailed docstrings with parameter and return types
- Make sure eveything sends to the same results directory

- Cleanup
- `pip3 freeze > requirements.txt`
- includes `examples` folder
    - Add
        - prostate_reader (DOI: 10.1001/jamanetworkopen.2020.23267)
        - nottingham (DOI: 10.1016/j.prp.2021.153718)
        - [PDL1](https://cran.r-project.org/web/packages/ONEST/index.html) (doi:10.1038/s41379-020-0544-x)
- Write up instructions on running some data from the beginning
- Get dad to run PDL1 from instructions I write up

```python
for root, subdirs, files in os.walk(sys.argv[1]):
    for filename in files:
        path = os.path.join(root, filename)
        nameBool, ext = nameCheck(path)
```

# Future Considerations
We calculate the OPA as the proportion of the number of observer agreements to total number of cases. There may be multiple ways to calculate this. The [FDA discuss overall percent agreement][fda-opa] in a 2-class positive vs. negative context.

[fda-opa]: https://www.fda.gov/files/medical%20devices/published/Guidance-for-Industry-and-FDA-Staff---Statistical-Guidance-on-Reporting-Results-from-Studies-Evaluating-Diagnostic-Tests-%28PDF-Version%29.pdf
[slides]: https://docs.google.com/presentation/d/1b7S_4nVgq0DsrQRkU7vEaCQtnOKafUGa/edit#slide=id.g22e7b4ae203_0_166