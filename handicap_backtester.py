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

# Define the betting simulator function
def simulate_betting(
    df: pd.DataFrame,
    min_ev: float = 0.1,
    max_ev: float = 0.3,
    bet_amount: float = 2500,
    game_max: float = 5000,
    min_samplesize: int = 0,
    max_samplesize: int = 0,
    initial_bankroll: float = 100000,
    cooldown: int = 180,
    q2=True,
    q4=True,
    handicap_range: tuple = (-10, 10),
    pregame_spread_range: tuple = (-15, 15),
    score_diff_range: tuple = (-20, 20),
    time_left_range: tuple = (0, 48)
) -> pd.DataFrame:
    bankroll = initial_bankroll
    total_wagered = 0
    total_profit = 0
    q4profit = 0
    h1profit = 0
    win = 0
    loss = 0
    bet_count = 0
    cumulative_bets = {}
    last_bet_time = {}
    bets_data = []
    bankroll_over_time = [initial_bankroll]
    bankroll_over_time_h1 = [0]
    bankroll_over_time_q4 = [0]
    period_map = {'1st': 36, '2nd': 24, '3rd': 12, '4th': 0}
    df['Sample Size'] = df['Sample Size'].fillna(0)

    for index, row in tqdm(df.iloc[::-1].iterrows(), total=df.shape[0], desc="Simulating Bets", miniters=100):

        if row['Minutes Remaining'] < time_left_range[0] or row['Minutes Remaining'] > time_left_range[1]:
            continue

        if row['Handicap'] < handicap_range[0] or row['Handicap'] > handicap_range[1]:
            continue

        if row['Type'] == "handicap_odds_q4" and q4:
            score_diff = row['Q4 score difference']+row['H1 score difference']

        elif row['Type'] == "handicap_odds_q2" and q2:
            score_diff = row['H1 score difference']
        else:
            continue

        minutes_left = period_map[row['Period']] + int(row['Clock'].split(':')[0])

        if row['Sample Size'] < min_samplesize:
            continue

        if row['Sample Size'] > max_samplesize:
            continue

        if not score_diff_range[0] <= score_diff <= score_diff_range[1]:
            continue

        if not pregame_spread_range[0] <= row['Pregame Spread'] <= pregame_spread_range[1]:
            continue

        game_key = (row['home_team_name'], row['away_team_name'], row['match_date'], row['Type'])

        if game_key not in cumulative_bets:
            cumulative_bets[game_key] = 0

        if game_key in last_bet_time:
            time_since_last_bet = (last_bet_time[game_key] - row['Minutes Remaining']*60)
            if time_since_last_bet < cooldown:
                continue

        if min_ev < row['Home EV'] < max_ev:
            handicap = row['Handicap']
            pregamespread = row['Pregame Spread']
            if not handicap_range[0] <= handicap <= handicap_range[1]:
                continue
            if pregamespread < pregame_spread_range[0] or pregamespread > pregame_spread_range[1]:
                continue

            if cumulative_bets[game_key] + bet_amount <= game_max:
                if score_diff + handicap > 0:
                    profit = bet_amount * (row['Home Odds']) * 1.04
                    outcome = 'Win'
                    win += 1
                elif score_diff + handicap == 0:
                    profit = 0
                    outcome = 'Tie'
                else:
                    profit = -bet_amount
                    outcome = 'Loss'
                    loss += 1

                total_profit += profit
                total_wagered += bet_amount
                bankroll += profit

                bets_data.append({
                    'Wagered Team': 'Home',
                    'Home Team': row['home_team_name'],
                    'Away Team': row['away_team_name'],
                    'EV': row['Home EV'],
                    'Odds': row['Home Decimal Odds'],
                    'Final Score Difference': score_diff,
                    'Handicap': handicap,
                    'Pregame Handicap': row['Pregame Spread'],
                    'Current Period': row['Period'],
                    'Current Clock': row['Clock'],
                    'Current Score': row['Score'],
                    'Prediction Quarter': row['Derivative Quarter'],
                    'Final Score': row['quarter_scores'],
                    'Outcome': outcome,
                    'Profit': profit,
                    'New Bankroll': bankroll
                })

                bankroll_over_time.append(bankroll)

                if row['Type'] == "handicap_odds_q4":
                    q4profit += profit
                    bankroll_over_time_q4.append(q4profit)
                elif row['Type'] == "handicap_odds_q2":
                    h1profit += profit
                    bankroll_over_time_h1.append(h1profit)

                cumulative_bets[game_key] += bet_amount
                last_bet_time[game_key] = row['Minutes Remaining'] * 60
                bet_count += 1

        if min_ev < row['Away EV'] < max_ev:
            score_diff = -score_diff
            handicap = -row['Handicap']
            pregamespread = -row['Pregame Spread']

            if not handicap_range[0] <= handicap <= handicap_range[1]:
                continue
            if pregamespread < pregame_spread_range[0] or pregamespread > pregame_spread_range[1]:
                continue

            if cumulative_bets[game_key] + bet_amount <= game_max:
                if score_diff + handicap > 0:
                    profit = bet_amount * (row['Away Odds']) * 1.04
                    outcome = 'Win'
                    win += 1
                elif score_diff + handicap == 0:
                    profit = 0
                    outcome = 'Tie'
                else:
                    profit = -bet_amount
                    outcome = 'Loss'
                    loss += 1

                total_profit += profit
                total_wagered += bet_amount
                bankroll += profit

                bets_data.append({
                    'Wagered Team': 'Away',
                    'Home Team': row['home_team_name'],
                    'Away Team': row['away_team_name'],
                    'EV': row['Away EV'],
                    'Odds': row['Away Decimal Odds'],
                    'Final Score Difference': score_diff,
                    'Handicap': handicap,
                    'Pregame Handicap': pregamespread,
                    'Current Period': row['Period'],
                    'Current Clock': row['Clock'],
                    'Current Score': row['Score'],
                    'Prediction Quarter': row['Derivative Quarter'],
                    'Final Score': row['quarter_scores'],
                    'Outcome': outcome,
                    'Profit': profit,
                    'New Bankroll': bankroll
                })

                bankroll_over_time.append(bankroll)

                if row['Type'] == "handicap_odds_q4":
                    q4profit += profit
                    bankroll_over_time_q4.append(q4profit)
                elif row['Type'] == "handicap_odds_q2":
                    h1profit += profit
                    bankroll_over_time_h1.append(h1profit)

                cumulative_bets[game_key] += bet_amount
                last_bet_time[game_key] = row['Minutes Remaining'] * 60
                bet_count += 1

    # Calculate ROI
    total_roi = (total_profit / total_wagered) * 100 if total_wagered != 0 else 0

    # Display Results
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Yield: {total_roi:.2f}%")
    print(f"Total Hitrate: {100*win/(win+loss):.2f}%")
    print(f"Sample Size: {bet_count}")
    print(f"Unique Games: {df[['home_team_name', 'away_team_name', 'match_date']].drop_duplicates().shape[0]}")

    # Create a DataFrame for the bets
    bets_df = pd.DataFrame(bets_data)

    # Plot Bankroll Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_over_time, marker='o', linestyle='-', color='b')
    plt.title('Total Bankroll Over Time')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.show()

    # Plot Bankroll for H1
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_over_time_h1, marker='o', linestyle='-', color='r')
    plt.title('Bankroll Over Time for H1 Bets')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.show()

    # Plot Bankroll for Q4
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_over_time_q4, marker='o', linestyle='-', color='g')
    plt.title('Bankroll Over Time for Q4 Bets')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.show()

    # Display the bets DataFrame
    print("\nBets Details:")
    print(bets_df.to_string(index=False))
    
    return bets_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Load your data
odds_df = pd.read_csv("cleaned_all.csv")

# Create a layout for the sliders
slider_layout = widgets.Layout(width='500px', description_width='150px')

# Create widgets for each parameter
min_ev_slider = widgets.FloatSlider(
    value=0,
    min=-1.0,
    max=1.0,
    step=0.01,
    description='Min EV:',
    continuous_update=False,
    layout=slider_layout
)

max_ev_slider = widgets.FloatSlider(
    value=1.0,
    min=-1.0,
    max=1.0,
    step=0.01,
    description='Max EV:',
    continuous_update=False,
    layout=slider_layout
)

bet_amount_slider = widgets.IntSlider(
    value=5000,
    min=100,
    max=25000,
    step=100,
    description='Bet Amount:',
    continuous_update=False,
    layout=slider_layout
)

game_max_slider = widgets.IntSlider(
    value=15000,
    min=1000,
    max=100000,
    step=500,
    description='Game Max:',
    continuous_update=False,
    layout=slider_layout
)

min_samplesize_slider = widgets.IntSlider(
    value=10,
    min=0,
    max=1000,
    step=10,
    description='Min Model Sample:',
    continuous_update=False,
    layout=slider_layout
)

max_samplesize_slider = widgets.IntSlider(
    value=100000,
    min=0,
    max=10000,
    step=1000,
    description='Max Model Sample:',
    continuous_update=False,
    layout=slider_layout
)

initial_bankroll_slider = widgets.IntSlider(
    value=100000,
    min=10000,
    max=1000000,
    step=5000,
    description='Start Bankroll:',
    continuous_update=False,
    layout=slider_layout
)

cooldown_slider = widgets.IntSlider(
    value=120,
    min=0,
    max=1440,
    step=10,
    description='Cooldown (s):',
    continuous_update=False,
    layout=slider_layout
)

q2_toggle = widgets.Checkbox(
    value=False,
    description='Consider Q2 Bets',
    disabled=False
)

q4_toggle = widgets.Checkbox(
    value=True,
    description='Consider Q4 Bets',
    disabled=False
)

handicap_range_slider = widgets.IntRangeSlider(
    value=(-30, 30),
    min=-30,
    max=30,
    step=1,
    description='Current Spread:',
    continuous_update=False,
    layout=slider_layout
)

pregame_spread_range_slider = widgets.IntRangeSlider(
    value=(-30, 30),
    min=-30,
    max=30,
    step=1,
    description='Pregame Spread:',
    continuous_update=False,
    layout=slider_layout
)

score_diff_range_slider = widgets.IntRangeSlider(
    value=(-30, 30),
    min=-30,
    max=30,
    step=1,
    description='Score Diff:',
    continuous_update=False,
    layout=slider_layout
)

time_left_range_slider = widgets.IntRangeSlider(
    value=(0, 40),
    min=0,
    max=48,
    step=1,
    description='Time Left:',
    continuous_update=False,
    layout=slider_layout
)

# Additional widgets for sample size and random state
sample_size_slider = widgets.FloatSlider(
    value=0.33,
    min=0.1,
    max=1.0,
    step=0.01,
    description='Sample %:',
    continuous_update=False,
    layout=slider_layout
)

random_state_slider = widgets.IntSlider(
    value=12,
    min=0,
    max=100,
    step=1,
    description='Seed:',
    continuous_update=False,
    layout=slider_layout
)

# Function to run the simulation
def run_simulation(min_ev, max_ev, bet_amount, game_max, min_samplesize, max_samplesize,
                   initial_bankroll, cooldown, q2, q4, handicap_range, pregame_spread_range,
                   score_diff_range, time_left_range, sample_size, random_state):

    # Sample the data
    odds_df_sample = odds_df.sample(frac=sample_size, random_state=random_state)

    # Run the betting simulation
    bets_df = simulate_betting(
        df=odds_df_sample,
        min_ev=min_ev,
        max_ev=max_ev,
        bet_amount=bet_amount,
        game_max=game_max,
        min_samplesize=min_samplesize,
        max_samplesize=max_samplesize,
        initial_bankroll=initial_bankroll,
        cooldown=cooldown,
        q2=q2,
        q4=q4,
        handicap_range=handicap_range,
        pregame_spread_range=pregame_spread_range,
        score_diff_range=score_diff_range,
        time_left_range=time_left_range
    )
    return bets_df

# Link widgets to the simulation function
out = widgets.interactive_output(run_simulation, {
    'min_ev': min_ev_slider,
    'max_ev': max_ev_slider,
    'bet_amount': bet_amount_slider,
    'game_max': game_max_slider,
    'min_samplesize': min_samplesize_slider,
    'max_samplesize': max_samplesize_slider,
    'initial_bankroll': initial_bankroll_slider,
    'cooldown': cooldown_slider,
    'q2': q2_toggle,
    'q4': q4_toggle,
    'handicap_range': handicap_range_slider,
    'pregame_spread_range': pregame_spread_range_slider,
    'score_diff_range': score_diff_range_slider,
    'time_left_range': time_left_range_slider,
    'sample_size': sample_size_slider,
    'random_state': random_state_slider
})

# Arrange and display the UI components
ui = widgets.VBox([
    min_ev_slider,
    max_ev_slider,
    bet_amount_slider,
    game_max_slider,
    min_samplesize_slider,
    max_samplesize_slider,
    initial_bankroll_slider,
    cooldown_slider,
    q2_toggle,
    q4_toggle,
    handicap_range_slider,
    pregame_spread_range_slider,
    score_diff_range_slider,
    time_left_range_slider,
    sample_size_slider,
    random_state_slider
])

# Display the widgets and output
display(ui, out)

