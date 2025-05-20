import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Set the page to wide mode
st.set_page_config(layout="wide", page_title="Staking Reward Simulator")

# Title for the application
st.title("Staking Reward Simulator")
st.subheader("Simulate staking rewards and APY over time with different staking trajectories")

# Define the Gaussian function
def f(staked_amt, target_amt, sigma=0.02):
    return np.exp(- (staked_amt - target_amt)**2 / (2 * sigma**2))

# Define the simulation function
def simulate_apy(
    total_supply=2_000_000_000, 
    staking_reward_amt=600_000_000, 
    staking_reward_timeline_yrs=10, 
    seed=42,
    initial_stake_pct_mean=0.2, 
    trajectory_type="flat", 
    increasing_rate=0.02, 
    decreasing_rate=0.02, 
    stake_pct_trajectory_std=0.03,
    max_apy=0.5,
    target_lock_pct=0.2,
    f_sigma=0.05, # This is the sigma for the Gaussian function f()
    airdrop_amount=1000000,
):
    
    rng = np.random.RandomState(seed)

    yearly_nominal_release = staking_reward_amt / staking_reward_timeline_yrs
    validator_reward_pot = yearly_nominal_release
    airdrop_by_year = staking_reward_timeline_yrs * [airdrop_amount]
    
    cs_pct_by_yr = [0.30, 0.47, 0.62, 0.75, 0.80, 1, 1, 1, 1, 1]
    if len(cs_pct_by_yr) < staking_reward_timeline_yrs:
        cs_pct_by_yr.extend([1] * (staking_reward_timeline_yrs - len(cs_pct_by_yr)))
    
    target_lock_pct_by_yr = staking_reward_timeline_yrs * [target_lock_pct]
    # Use the passed f_sigma to create the sigma_by_yr list
    sigma_by_yr = staking_reward_timeline_yrs * [f_sigma]


    apy_with_airdrop_yr = []
    apy_without_airdrop_yr = []
    staked_pct_by_yr = []
    validator_reward_pot_by_yr = []
    validator_rewards_distributed_by_yr = []
    years = []
    
    current_stake_mean = initial_stake_pct_mean

    for yr in range(1, staking_reward_timeline_yrs + 1):
        years.append(yr)
        
        if yr > 1: # Adjust stake_pct_mean for years > 1 based on trajectory
            if trajectory_type == "increasing":
                current_stake_mean = current_stake_mean * (1 + increasing_rate)
            elif trajectory_type == "decreasing":
                current_stake_mean = current_stake_mean * (1 - decreasing_rate)
        
        current_stake_mean = np.clip(current_stake_mean, 0.001, 0.999)

        cs_nominal = total_supply * cs_pct_by_yr[yr-1]
        target_stake_pct_for_year = target_lock_pct_by_yr[yr-1] # Use a distinct name
        f_sigma_for_year = sigma_by_yr[yr-1] # Use a distinct name

        stake_pct = rng.normal(loc=current_stake_mean, scale=stake_pct_trajectory_std)
        stake_pct = np.clip(stake_pct, 0.001, 0.999) 
        staked_amt = cs_nominal * stake_pct
        
        release_pct = f(stake_pct, target_stake_pct_for_year, sigma=f_sigma_for_year)
        remaining_pct = 1-release_pct
        validator_rewards_release_amt = release_pct * validator_reward_pot
        validator_reward_pot = (remaining_pct * validator_reward_pot) + yearly_nominal_release
        
        apy_without_airdrop = (validator_rewards_release_amt / staked_amt) if staked_amt > 0 else 0
        
        if apy_without_airdrop > max_apy:
            actual_release_amt = staked_amt * max_apy
            validator_reward_pot += (validator_rewards_release_amt - actual_release_amt)
            validator_rewards_release_amt = actual_release_amt
            apy_without_airdrop = max_apy
        
        release_with_airdrop = validator_rewards_release_amt + airdrop_by_year[yr-1]
        apy_with_airdrop = (release_with_airdrop / staked_amt) if staked_amt > 0 else 0

        validator_reward_pot_by_yr.append(validator_reward_pot)
        staked_pct_by_yr.append(stake_pct)
        apy_with_airdrop_yr.append(apy_with_airdrop)
        apy_without_airdrop_yr.append(apy_without_airdrop)
        validator_rewards_distributed_by_yr.append(validator_rewards_release_amt)
    
    results_df = pd.DataFrame({
        'Year': years,
        'Staked Percentage': staked_pct_by_yr,
        'APY with Airdrop': apy_with_airdrop_yr,
        'APY without Airdrop': apy_without_airdrop_yr,
        'Validator Reward Pot (M)': [x/1e6 for x in validator_reward_pot_by_yr],
        'Validator Rewards Distributed (M)': [x/1e6 for x in validator_rewards_distributed_by_yr],
        'Cumulative Validator Rewards (M)': np.cumsum([x/1e6 for x in validator_rewards_distributed_by_yr])
    })
    
    return results_df

# Sidebar for parameter configuration
st.sidebar.title("Simulation Parameters")

# Group 1: Supply and Reward Parameters
st.sidebar.header("Supply and Reward Parameters")
total_supply = st.sidebar.number_input("Total Supply", 
                                       min_value=1000000000, 
                                       max_value=10000000000, 
                                       value=2000000000, 
                                       step=100000000,
                                       format="%d")

staking_reward_amt = st.sidebar.number_input("Staking Reward Amount", 
                                             min_value=100000000, 
                                             max_value=1000000000, 
                                             value=600000000, 
                                             step=50000000,
                                             format="%d")

staking_reward_timeline_yrs = st.sidebar.slider("Staking Reward Timeline (years)", 
                                               min_value=5, 
                                               max_value=20, 
                                               value=10)

# Group 2: Staking Trajectory Parameters
st.sidebar.header("Staking Trajectory Parameters")
initial_stake_pct_mean = st.sidebar.slider("Initial Mean Staking Percentage", 
                                             min_value=0.05, 
                                             max_value=0.50, 
                                             value=0.15, 
                                             step=0.01,
                                             format="%.2f",
                                             help="The starting mean staking percentage for all trajectories.")

increasing_rate_pct_yr = st.sidebar.slider("Increasing Rate (% per year for mean staking)",
                                           min_value=0.0,
                                           max_value=10.0,
                                           value=3.0,
                                           step=0.5,
                                           format="%.1f",
                                           help="Annual percentage increase for the mean staking percentage in the 'Increasing' trajectory.")

decreasing_rate_pct_yr = st.sidebar.slider("Decreasing Rate (% per year for mean staking)",
                                           min_value=0.0,
                                           max_value=10.0,
                                           value=3.0,
                                           step=0.5,
                                           format="%.1f",
                                           help="Annual percentage decrease for the mean staking percentage in the 'Decreasing' trajectory.")

stake_pct_trajectory_std = st.sidebar.slider("Standard Deviation for Staking Percentage", 
                                            min_value=0.00, 
                                            max_value=0.10, 
                                            value=0.01, 
                                            step=0.01,
                                            format="%.2f",
                                            help="Standard deviation around the mean staking percentage for each year's random draw.")

# Group 3: Advanced Parameters
st.sidebar.header("Advanced Parameters")
max_apy = st.sidebar.slider("Maximum APY", 
                           min_value=0.1, 
                           max_value=1.0, 
                           value=0.5, 
                           step=0.1,
                           format="%.1f")

target_lock_pct = st.sidebar.slider("Target Lock Percentage (for reward function)", 
                                   min_value=0.05, 
                                   max_value=0.40, 
                                   value=0.20, 
                                   step=0.05,
                                   help="The target staking percentage for the Gaussian reward distribution function.")

f_sigma = st.sidebar.slider("Sigma for Gaussian Reward Function", 
                           min_value=0.01, 
                           max_value=0.10, 
                           value=0.05, 
                           step=0.01,
                           help="Controls the spread of the Gaussian reward distribution function.")

airdrop_amount = st.sidebar.number_input("Annual Airdrop Amount", 
                                        min_value=0, 
                                        max_value=5000000, 
                                        value=1000000, 
                                        step=100000)

seed = st.sidebar.number_input("Base Random Seed", 
                              min_value=1, 
                              max_value=1000, 
                              value=42,
                              help="The base seed for simulations. Each run within a trajectory will use this + an increment.")

# Group 4: Simulation Control
st.sidebar.header("Simulation Control")
num_simulations_per_trajectory = st.sidebar.slider("Simulations per Trajectory (for median)", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=10,
                                   help="Number of simulations to run for each trajectory type. Results shown are medians over these runs.")

# Run simulations and display results
if st.button("Run Simulations"):
    st.header("Median Simulation Results by Trajectory")
    
    all_median_results_list = []
    trajectory_types = ["Flat", "Increasing", "Decreasing"]
    
    increase_rate_decimal = increasing_rate_pct_yr / 100.0
    decrease_rate_decimal = decreasing_rate_pct_yr / 100.0

    for trajectory_name in trajectory_types:
        current_trajectory_runs_dfs = [] 
        
        for i in range(num_simulations_per_trajectory):
            current_run_seed = seed + i 
            
            sim_df = simulate_apy(
                total_supply=total_supply,
                staking_reward_amt=staking_reward_amt,
                staking_reward_timeline_yrs=staking_reward_timeline_yrs,
                seed=current_run_seed,
                initial_stake_pct_mean=initial_stake_pct_mean,
                trajectory_type=trajectory_name.lower(), # "flat", "increasing", "decreasing"
                increasing_rate=increase_rate_decimal,
                decreasing_rate=decrease_rate_decimal,
                stake_pct_trajectory_std=stake_pct_trajectory_std,
                max_apy=max_apy,
                target_lock_pct=target_lock_pct,
                f_sigma=f_sigma, # Pass the sidebar f_sigma here
                airdrop_amount=airdrop_amount
            )
            current_trajectory_runs_dfs.append(sim_df)
        
        if current_trajectory_runs_dfs:
            # Concatenate all DataFrames for the current trajectory type
            concatenated_trajectory_df = pd.concat(current_trajectory_runs_dfs)
            # Group by 'Year' and calculate the median for all numeric columns
            median_df_for_trajectory = concatenated_trajectory_df.groupby('Year').median().reset_index()
            median_df_for_trajectory['Trajectory Type'] = trajectory_name 
            all_median_results_list.append(median_df_for_trajectory)
            
    if all_median_results_list:
        combined_median_results_df = pd.concat(all_median_results_list)
    else:
        st.write("No simulation data generated. Adjust parameters and try again.")
        combined_median_results_df = pd.DataFrame() 

    if not combined_median_results_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Median Staking Percentage")
            staking_chart = alt.Chart(combined_median_results_df).mark_line(strokeDash=[5,5]).encode(
                x=alt.X('Year:Q', title='Year'),
                y=alt.Y('Staked Percentage:Q', title='Median Staking Percentage', axis=alt.Axis(format='%')),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(staking_chart, use_container_width=True)

            st.subheader("Median APY")
            base_apy_chart = alt.Chart(combined_median_results_df).encode(
                x=alt.X('Year:Q', title='Year'),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            )
            
            apy_with_chart = base_apy_chart.mark_line().encode(
                y=alt.Y('APY with Airdrop:Q', title='Median APY', axis=alt.Axis(format='%'))
            )
            
            apy_without_chart = base_apy_chart.mark_line(strokeDash=[1, 1]).encode(
                y=alt.Y('APY without Airdrop:Q', axis=alt.Axis(format='%')) # Title inherited from apy_with_chart
            )
            
            apy_charts = alt.layer(
                apy_with_chart,
                apy_without_chart
            ).resolve_scale(
                y='shared' # APYs share the same scale
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(apy_charts, use_container_width=True)
            
            st.subheader("Median Validator Rewards Distributed")
            rewards_dist_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Year:Q', title='Year'),
                y=alt.Y('Validator Rewards Distributed (M):Q', title='Median Rewards Distributed (M-Tokens)'),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(rewards_dist_chart, use_container_width=True)
            
        with col2:
            st.subheader("Median Validator Reward Pot")
            reward_pot_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Year:Q', title='Year'),
                y=alt.Y('Validator Reward Pot (M):Q', title='Median Pot Size (M-Tokens)'),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(reward_pot_chart, use_container_width=True)
            
            st.subheader("Median Cumulative Validator Rewards")
            cumulative_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Year:Q', title='Year'),
                y=alt.Y('Cumulative Validator Rewards (M):Q', title='Median Cumulative Rewards (M-Tokens)'),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(cumulative_chart, use_container_width=True)
        
        st.markdown("""
        **Chart Legend:**
        - Colors differentiate staking trajectory types (Flat, Increasing, Decreasing).
        - **Median Staking Percentage**: Dashed lines (- - -) in its dedicated chart.
        - **Median APY with Airdrop**: Solid lines (—) in the APY chart.
        - **Median APY without Airdrop**: Dotted lines (···) in the APY chart.
        
        *All charts show median values over the specified number of simulations per trajectory.*
        """)
        
        with st.expander("View Median Simulation Data by Trajectory"):
            st.dataframe(combined_median_results_df)

# Add some explanatory text
st.markdown("""
## How to use this simulator

1.  **Adjust Parameters**: Use the sidebar to configure supply, rewards, staking trajectory details (initial mean, increase/decrease rates, standard deviation), advanced settings (max APY, target lock for rewards, Gaussian sigma), airdrops, and simulation controls.
2.  **Set Trajectory Rates**: Define how the mean staking percentage should evolve for the 'Increasing' and 'Decreasing' trajectories.
3.  **Number of Simulations**: Specify how many simulations to run *for each* of the three trajectory types (Flat, Increasing, Decreasing). The displayed charts will show the *median* results from these runs.
4.  **Run Simulations**: Click the "Run Simulations" button to generate and visualize the results.

### Key Parameters Explained

-   **Initial Mean Staking Percentage**: The starting average percentage of circulating supply expected to be staked. This is the baseline for all three trajectory types.
-   **Increasing/Decreasing Rate**: Annual percentage change applied to the mean staking percentage for the respective trajectories.
-   **Std Dev for Staking Percentage**: Variability (randomness) around the mean staking percentage each year.
-   **Target Lock Percentage (for reward function)**: The staking percentage where the Gaussian reward function `f()` peaks, influencing reward distribution.
-   **Sigma for Gaussian Reward Function**: Controls the spread/steepness of the reward distribution curve around the target lock percentage.
-   **Simulations per Trajectory**: Determines how many individual simulation runs are performed for each of the "Flat", "Increasing", and "Decreasing" scenarios to calculate the median outcomes.

### Understanding the Results

The charts display the **median** outcomes over the simulations for each trajectory type:
-   **Median Staking Percentage**: The median yearly percentage of circulating supply staked.
-   **Median APY with/without Airdrop**: The median Annual Percentage Yield for stakers.
-   **Median Validator Reward Pot**: The median size of the remaining undistributed rewards.
-   **Median Validator Rewards Distributed**: The median rewards distributed per year.
-   **Median Cumulative Validator Rewards**: The median total rewards distributed over time.
""")