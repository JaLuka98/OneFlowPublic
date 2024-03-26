import os
import fnmatch # for unix-like wildcard matching
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.utils import shuffle


# Define the command line arguments
parser = argparse.ArgumentParser(description='Generate plots for MC, data, or together')
parser.add_argument('source', choices=['MC', 'data', 'together'], help='Specify whether to plot MC, data, or together')
parser.add_argument('--postfix', type=str, default='', help='Postfix string for the files to plot. Example: Specify --postfix _withNoise to plot from a file like my_data_withNoise.csv. This postfix is also used to name the output directory.')
args = parser.parse_args()

# Define file paths and prefixes based on the user's choice
if args.source == 'MC':
    csv_files = ['./samples/my_MC' + args.postfix + '.csv']
    plot_prefix = 'MC'
    label = 'MC' # Label for MC
elif args.source == 'data':
    csv_files = ['./samples/my_data' + args.postfix + '.csv']
    plot_prefix = 'data'
    label = 'Data' # Label for data
else:
    csv_files = ['./samples/my_MC' + args.postfix + '.csv', './samples/my_data' + args.postfix + '.csv']
    plot_prefix = 'together'
    label = ['MC','Data']

plot_dir = './plots' + args.postfix
# Create the respective plot directory
if not os.path.exists(plot_dir):
    # If the directory doesn't exist, create it
    os.makedirs(plot_dir)

# Load the CSV files into pandas DataFrames and add the "Label" column
data_frames = [pd.read_csv(file) for file in csv_files]
# Use zip to iterate through data_frames and labels
for df, lab in zip(data_frames, label):
    df['Label'] = lab

# Select only the numerical columns for plotting
numerical_columns = [df.select_dtypes(include=['float64']) for df in data_frames]

# Create six plots, each with two histograms
for column in numerical_columns[0].columns:
    plt.figure()
    N_bins = 100
    if 'pT' in column:
        lo, hi = 0, 6
    elif 'eta' in column:
        lo, hi = 0, 3
    elif 'noise' in column:
        lo, hi = 0, 3
    elif fnmatch.fnmatch(column, 'v*A'):
        lo, hi = -4, 4
    elif fnmatch.fnmatch(column, 'v*B'):
        lo, hi = 0, 6
        N_bins = 30
    else:
        print('PROBLEM')
    binning = np.linspace(lo, hi, N_bins)  
    for i, df in enumerate(data_frames):
        plt.hist(df[column], bins=binning, density=True, alpha=0.5, label=label[i])
    plt.title(f'Histogram of {column} ({plot_prefix})')
    plt.xlabel(column)
    plt.ylabel('Normalized Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plot_dir}/histogram_{plot_prefix}_{column}.pdf')
    plt.clf()

combined_df = pd.concat(data_frames) # Combine the DataFrames into a single DataFrame
combined_df = shuffle(combined_df) # Shuffle the data frame
# Create a pairplot with both sources of data and distinguish them by "Label"
sns.set(style="ticks")
sns.pairplot(combined_df.head(2000).drop(columns=['Unnamed: 0']), hue='Label', kind="scatter", diag_kind='hist')
plt.savefig(f'{plot_dir}/dataset_visualization_{plot_prefix}.pdf') # Save the plot to a PDF file

# Create correlation matrices for both sources of data, using only the first 100,000 rows to save some runtime
correlation_matrices = [numerical.iloc[:100000, :].corr() for numerical in numerical_columns]
for i, correlation_matrix in enumerate(correlation_matrices):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix ({plot_prefix})')
    plt.savefig(f'{plot_dir}/correlation_matrix_{plot_prefix}_{i}.pdf')
