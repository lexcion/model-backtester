import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display
import os
import gdown
import pandas as pd

# File ID from Google Drive
file_id = '1eJfCer2tdxDrgINTxhtDk8i4mrJh3ZO7'  # Replace with your actual file ID

# Direct download URL
direct_link = f'https://drive.google.com/uc?id={file_id}&export=download'

# Output filename
filename = 'cleaned_all.csv'  # Name the file as you wish


# Check if the file exists locally
if not os.path.exists(filename):
    print(f"File '{filename}' not found locally. Downloading from Google Drive...")
    
    # Download the file using gdown
    gdown.download(direct_link, filename, quiet=False)
else:
    
    print(f"File '{filename}' already exists locally.")
    
# Load the CSV into a DataFrame
odds_df = pd.read_csv(filename)

# Display the first few rows to verify
#print(odds_df.head())
def plot_ev_distribution(odds_df, ev_threshold=0.1):
    home_ev_mean = odds_df['Home EV'].mean()
    home_ev_std = odds_df['Home EV'].std()
    away_ev_mean = odds_df['Away EV'].mean()
    away_ev_std = odds_df['Away EV'].std()

    # Print statistics
    print(f"Home EV Mean: {home_ev_mean:.4f}, Home EV Std Dev: {home_ev_std:.4f}")
    print(f"Away EV Mean: {away_ev_mean:.4f}, Away EV Std Dev: {away_ev_std:.4f}")

    # Count EV values
    home_ev_ge_threshold = np.sum(odds_df['Home EV'] >= ev_threshold)
    home_ev_lt_threshold = np.sum(odds_df['Home EV'] < ev_threshold)
    away_ev_ge_threshold = np.sum(odds_df['Away EV'] >= ev_threshold)
    away_ev_lt_threshold = np.sum(odds_df['Away EV'] < ev_threshold)

    print(f"Home EV >= {ev_threshold}: {home_ev_ge_threshold}, Home EV < {ev_threshold}: {home_ev_lt_threshold}")
    print(f"Away EV >= {ev_threshold}: {away_ev_ge_threshold}, Away EV < {ev_threshold}: {away_ev_lt_threshold}")

    # Plotting Home EV
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(odds_df['Home EV'].dropna(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Home EV')
    plt.xlabel('Home EV')
    plt.ylabel('Frequency')

    for patch, left_bin_edge in zip(patches, bins[:-1]):
        if left_bin_edge >= ev_threshold:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')

    plt.show()

    # Plotting Away EV
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(odds_df['Away EV'].dropna(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Away EV')
    plt.xlabel('Away EV')
    plt.ylabel('Frequency')

    for patch, left_bin_edge in zip(patches, bins[:-1]):
        if left_bin_edge >= ev_threshold:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')

    plt.show()
# Create widgets
ev_threshold_slider = widgets.FloatSlider(
    value=0,
    min=-1.0,
    max=1.0,
    step=0.01,
    description='EV Threshold:',
    continuous_update=False
)

# Function to update the plot based on widget input
def update_ev_plot(ev_threshold):
    plot_ev_distribution(odds_df, ev_threshold)

# Link widget to the function
widgets.interactive(update_ev_plot, ev_threshold=ev_threshold_slider)
