import lib
import numpy as np
import scipy.stats as stats

def fit(data):
    try:
        return stats.beta.fit(data)
    
    except:
        return None

def compare(assisted, unassisted, alpha_error=.05):
    cutoff = unassisted.ppf(1 - alpha_error)
    beta_error = assisted.cdf(cutoff)
    return (cutoff, beta_error)
    

# Find the OPA cutoff D for alpha_error probability to be greater than D
exp = "prostate_reader/"
group = "_5class"
root = "./data/"
results = "./results/" + exp + group + "/"
datasets = ["assisted", "unassisted"]
alpha_error = .05

datasets = np.transpose(np.asarray([lib.data_reader(root + exp + dataset + group + ".npy") for dataset in datasets]), (0, 3, 2, 1))

fits = np.apply_along_axis(fit, 3, datasets)
betas = np.apply_along_axis(lambda x: stats.beta(*x), 3, fits)
cutoff_and_beta_error = np.transpose(
    np.apply_along_axis(
        lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error),
        0, 
        betas, 
        alpha_error=alpha_error
    ), 
    (2, 1, 0)
)
np.savetxt(results + "alpha_beta_cutoffs.csv", cutoff_and_beta_error[:, :, 0], delimiter=",", fmt="%.3f")
np.savetxt(results + "alpha_beta_error.csv", cutoff_and_beta_error[:, :, 1], delimiter=",", fmt="%.3f")

