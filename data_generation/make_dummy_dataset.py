import json
import os
import argparse
import pandas as pd
import numpy as np
import utils.feature_helpers
import utils.stats_helpers
import mcerp


## Command-line arguments
parser = argparse.ArgumentParser(description="Generate a dataset with correlated variables.")
parser.add_argument("viASpecsPath", type=str, help="Path to the viA (variable of the first kind, continuous Gaussian-like) specifications JSON file.")
parser.add_argument("viBSpecsPath", type=str, help="Path to the viB (variable of the second kind, discontinuous) specifications JSON file.")
parser.add_argument("--pTScale", type=float, default=1.0, help="The scale of the exponential distribution to generate the pT spectrum.")
parser.add_argument("--etaSmear", type=float, default=0.25, help="The standard deviation of the Gaussian smearing to generate the eta spectrum.")
parser.add_argument("--addNoise", action='store_true', help="Add a uniform noise feature akin to pileup.") # store_true implies a default of false
parser.add_argument(
    "--finalCorrMatPath",
    type=str,
    help="""Path to the JSON file specifying the final desired correlation matrix among all the features.
    If the matrix you give in the JSON file is not psd, it will be slightly adjusted to form a valid corr matrix.
    If none is given, no artifical correlation will be performed""")
parser.add_argument("--nEvts", type=int, default=1000, help="Number of synthetic events to generate. Defaults to 1000.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for the numpy random generator. Defaults to 42.")
parser.add_argument("--outPath", type=str, help="Desired path to the output csv.")
args = parser.parse_args()


## Definitions of few necessary variables from the input arguments
# Read mean, std for Gaussian-like viA from JSON files
with open(args.viASpecsPath, 'r') as viA_json_file:
    viA_specs = json.load(viA_json_file)
means = np.array([viA_specs[viA]['mean'] for viA in viA_specs])
mean_condPreFacs = np.array([viA_specs[viA]['mean_condPreFac'] for viA in viA_specs])
stds = np.array([viA_specs[viA]['std'] for viA in viA_specs])
std_condPreFacs = np.array([viA_specs[viA]['std_condPreFac'] for viA in viA_specs])
# Read relative_fractions, x_thresholds and scales of exponential parts from JSON files (for viB variables)
with open(args.viBSpecsPath, 'r') as viB_json_file:
    viB_specs = json.load(viB_json_file)
rs = np.array([viB_specs[viB]['r'] for viB in viB_specs])
r_condPreFacs = np.array([viB_specs[viB]['r_condPreFac'] for viB in viB_specs])
x_thresholds = np.array([viB_specs[viB]['x_threshold'] for viB in viB_specs])
scales = np.array([viB_specs[viB]['scale'] for viB in viB_specs])
scale_condPreFacs = np.array([viB_specs[viB]['scale_condPreFac'] for viB in viB_specs])
if not args.finalCorrMatPath:
    print("Path to final correlation matrix was not provided.")
    print("No artifical correlation will be performed.")
    #print("Random final covariance matrix will be used.")
    # Create a random matrix with values in the range [-0.5, 0.5]
    #random_matrix = rng.uniform(-0.5, 0.5, size=(len(my_dict), len(my_dict)))
    #np.fill_diagonal(random_matrix, 1) # fill main diagonal with ones
    #matrix = random_matrix + random_matrix.T - np.diag(random_matrix.diagonal()) # Make the matrix symmetric by copying the upper triangular part to the lower triangular part
    #corr_matrix = utils.stats_helpers.find_closest_PSD(matrix) # Make the matrix psd
else:
    with open(args.finalCorrMatPath, 'r') as corr_json_file:
        matrix = np.array(json.load(corr_json_file))
        corr_matrix = utils.stats_helpers.find_closest_PSD(matrix) # Make the matrix psd if it is not already
num_samples = args.nEvts # Number of synthetic events to generate ("samples" to produce in statistics language)
rng = np.random.default_rng(args.seed) # Initialize our random number generator


## Generate kinematics
# Generate a random exponentially falling feature "pT"
pT = rng.exponential(scale=args.pTScale, size=num_samples)  # Adjust the scale parameter as needed
# Generate the "eta" feature that is uniform between -1.5 and 1.5 with exponential edges
uniform_eta = rng.uniform(low=-2, high=2, size=num_samples)
eta = uniform_eta * rng.normal(1, args.etaSmear, num_samples)
eta = np.abs(eta) # Because we assume our non-existent pseudo-detector to be symmetric 

# If desired, generate a flat noise distribution (can be associated with pileup, for example)
# If not desired, we just make zeroes to not have too many if else statements later
# We just ensure that any modifications to viA and viB variables are zero if the noise is zero
# We do not distinguish between data and MC here for now
if args.addNoise: noise = rng.uniform(low=0, high=3, size=num_samples)
else: noise = np.zeros(num_samples)

# Create a dictionary to organize our data
my_dict = {
    'pT': pT,
    'eta': eta,
}
if args.addNoise: my_dict['noise'] = noise # If we do not want to add noise, we just keep two kinematic features

# Generate "viA" variables based on the specifications
desired_shape = (num_samples, len(means)) # e.g. (5000, 3) if we want to have 5000 events and 3 features
modified_means = means - mean_condPreFacs * eta[:, np.newaxis]
modified_stds = stds - std_condPreFacs * pT[:, np.newaxis] + 0.1 * noise[:, np.newaxis] # viA variables get broader with increasing noise, on average
clipped_stds = np.clip(modified_stds, 0.1, None) # Clipping to avoid negative stds
viA_features = rng.normal(modified_means, clipped_stds, desired_shape)

# Add viA variables to the dictionary
for i in range(len(means)):
    viA_col_name = f"v{i+1}A"
    my_dict[viA_col_name] = viA_features[:, i]

# Generate viB distributions (discontinuous)
desired_shape = (num_samples, len(means)) # e.g. (5000, 3) if we want to have 5000 events and 3 features
modified_rs = rs + r_condPreFacs * pT[:, np.newaxis] - 0.1 * noise[:, np.newaxis] # Higher pT: More prominent delta peak at zero. High noise: Less likely to be associated with peak at zero
clipped_rs = np.clip(modified_rs, 0., 1.) # Clipping to avoid probability greater than 1 (and smaller than zero just for safety)
x_thresholds = np.tile(x_thresholds, (num_samples, 1)) # Reshape the x_thresholds array to match the required shape (n_samples, n_features)
modified_scales = scales + scale_condPreFacs * eta[:, np.newaxis] + 0.2 * noise[:, np.newaxis]
viB_features = utils.feature_helpers.generate_discontinuous_distribution(clipped_rs, x_thresholds, modified_scales, num_samples, rng)

# Add viB variables to the dictionary
for i in range(len(rs)):
    viB_col_name = f"v{i+1}B"
    my_dict[viB_col_name] = viB_features[:, i]

## Correlate the features
X = pd.DataFrame(my_dict).values # design matrix
if args.finalCorrMatPath is not None:
    threshold = 3e-6
    noise = np.random.uniform(1e-6, 2e-6, size=X.shape) # Generate random uniform noise between 1e-6 and 2e-6
    zero_indices = (X == 0) # Identify the elements in data that are exactly zero
    X_with_noise = X + zero_indices * noise # Add the noise to the elements that are exactly zero
    X_correlated = mcerp.induce_correlations(X_with_noise, corr_matrix) # correlate with mcerp
    X_correlated[np.abs(X_correlated) < threshold] = 0 # Map the values back to zero (Denoising)
else: 
    X_correlated = X

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(X_correlated, columns=my_dict.keys())

# Get the directory path
dir_path = os.path.dirname(args.outPath)

# Check if the directory exists
if not os.path.exists(dir_path):
    os.makedirs(dir_path) # If the directory doesn't exist, create it

df.to_csv(args.outPath)