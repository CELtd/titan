import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from core_sim import simulate_apy_monthly, STATIC_RELEASE_BUCKETS, convert_quarterly_to_monthly
# Set the page to wide mode
st.set_page_config(layout="wide", page_title="Staking Reward Simulator")

# Title for the application
st.title("Staking Reward Simulator")
st.subheader("Simulate staking rewards and APY over time with different staking trajectories")

# Create tabs for main content
config_tab, simulation_tab, help_tab = st.tabs(["Configuration", "APY Simulation", "Help & Instructions"])

# Move the help text to the help tab
with help_tab:
    st.markdown("""
    ## How to use this simulator

    1.  **Adjust Parameters**: Use the sidebar to configure supply, rewards, staking trajectory details (initial mean, increase/decrease rates, standard deviation), advanced settings (max APY, target lock for rewards, Gaussian sigma), and simulation controls.
    2.  **Configure Monthly Schedules**: Use the "Monthly Configuration" tab to set the "Circulating Supply %" and "Airdrop Amount (Tokens)" for each month of the simulation.
    3.  **Set Trajectory Rates**: Define how the mean staking percentage should evolve for the 'Increasing' and 'Decreasing' trajectories.
    4.  **Number of Simulations**: Specify how many simulations to run *for each* of the three trajectory types (Flat, Increasing, Decreasing). The displayed charts will show the *median* results from these runs.
    5.  **Run Simulations**: Click the "Run Simulations" button to generate and visualize the results.

    ### Key Parameters Explained

    -   **Monthly Configuration**: Allows month-by-month customization of:
        -   *Circulating Supply %*: The percentage of the total token supply that is liquid and available in the market each month.
        -   *Airdrop Amount (Tokens)*: The number of additional tokens airdropped to stakers each month.
    -   **Initial Mean Staking Percentage**: The starting average percentage of circulating supply expected to be staked. This is the baseline for all three trajectory types.
    -   **Increasing/Decreasing Rate**: Annual percentage change applied to the mean staking percentage for the respective trajectories.
    -   **Std Dev for Staking Percentage**: Variability (randomness) around the mean staking percentage each month.
    -   **Target Lock Percentage (for reward function)**: The staking percentage where the Gaussian reward function `f()` peaks, influencing reward distribution.
    -   **Sigma for Gaussian Reward Function**: Controls the spread/steepness of the reward distribution curve around the target lock percentage.
    -   **Simulations per Trajectory**: Determines how many individual simulation runs are performed for each of the "Flat", "Increasing", and "Decreasing" scenarios to calculate the median outcomes.

    ### Understanding the Results

    The charts display the **median** outcomes over the simulations for each trajectory type:
    -   **Median Staking Percentage**: The median monthly percentage of circulating supply staked.
    -   **Median APY with/without Airdrop**: The median Annual Percentage Yield for stakers (annualized from monthly rates).
    -   **Median Validator Reward Pot**: The median size of the remaining undistributed rewards.
    -   **Median Validator Rewards Distributed**: The median rewards distributed per month.
    -   **Median Cumulative Validator Rewards**: The median total rewards distributed over time.
    """)

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

staking_reward_amt = st.sidebar.number_input("Total Staking Reward Pool Size", 
                                             min_value=100000000, 
                                             max_value=1000000000, 
                                             value=200000000, 
                                             step=50000000,
                                             format="%d")

staking_reward_timeline_yrs = st.sidebar.slider("Staking Reward Timeline (years)", 
                                               min_value=5, 
                                               max_value=20, 
                                               value=10)

# Run simulations and display results in the simulation tab
seed = st.sidebar.number_input("Base Random Seed", 
                              min_value=1, 
                              max_value=1000, 
                              value=42,
                              help="The base seed for simulations. Each run within a trajectory will use this + an increment.")

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
                                            help="Standard deviation around the mean staking percentage for each month's random draw.")

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

# Group 4: Simulation Control
st.sidebar.header("Simulation Control")
num_simulations_per_trajectory = st.sidebar.slider("Simulations per Trajectory (for median)", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=10,
                                   help="Number of simulations to run for each trajectory type. Results shown are medians over these runs.")

# Calculate total months
total_months = staking_reward_timeline_yrs * 12

# Initialize session state variables
if 'static_release_schedule' not in st.session_state:
    st.session_state.static_release_schedule = STATIC_RELEASE_BUCKETS.copy()

if 'dynamic_params' not in st.session_state:
    st.session_state.dynamic_params = {
        'monthly_revenue': 1000000,  # Default $1M monthly revenue
        'token_price': 1.0,  # Default $1 token price
        'delta': 0.1  # Default 10% delta
    }

if 'target_staking_pct' not in st.session_state:
    st.session_state.target_staking_pct = [0.2] * (staking_reward_timeline_yrs * 12)  # Default 20% for all months

if 'airdrop_schedule' not in st.session_state:
    # Default even distribution
    default_airdrop_value_per_year = 1000000
    default_airdrop_list = []
    for _ in range(staking_reward_timeline_yrs):
        monthly_airdrop = default_airdrop_value_per_year / 12
        default_airdrop_list.extend([monthly_airdrop] * 12)
    st.session_state.airdrop_schedule = default_airdrop_list

with config_tab:
    st.header("Configuration")
    
    # Step 1: Static Release Schedule Section
    st.subheader("1. Static Release Schedule")
    st.write("Configure the quarterly release schedule for each bucket. Values are cumulative.")
    
    # Create a DataFrame for the static release schedule
    quarters = [f"Q{i+1}" for i in range(23)]  # 23 quarters
    static_release_data = {bucket: values for bucket, values in st.session_state.static_release_schedule.items()}
    static_release_df = pd.DataFrame(static_release_data, index=quarters)
    
    # Display editable table
    edited_static_release_df = st.data_editor(
        static_release_df,
        key="static_release_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                help=f"Cumulative tokens released for {col}",
                min_value=0,
                step=1000000,
                format="%d"
            ) for col in static_release_df.columns
        }
    )
    
    # Update session state with edited values
    st.session_state.static_release_schedule = edited_static_release_df.to_dict()
    
    # Add monthly breakdown display
    st.subheader("Monthly Release Schedule")
    st.write("This table shows the monthly token releases for each bucket (calculated from quarterly values).")
    
    # Calculate monthly values for each bucket
    monthly_release_data = {}
    for bucket, quarterly_values in st.session_state.static_release_schedule.items():
        quarterly_values_list = list(quarterly_values.values())
        monthly_values = convert_quarterly_to_monthly(quarterly_values_list, len(quarterly_values_list) * 3)
        monthly_release_data[bucket] = monthly_values
    
    # Create DataFrame for monthly values
    months = [f"Month {i+1}" for i in range(len(next(iter(monthly_release_data.values()))))]
    monthly_release_df = pd.DataFrame(monthly_release_data, index=months)
    
    # Display monthly breakdown
    st.dataframe(
        monthly_release_df,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                help=f"Monthly tokens released for {col}",
                format="%d"
            ) for col in monthly_release_df.columns
        }
    )
    
    # Step 2: Dynamic Components Section
    st.subheader("2. Deal Rewards Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_revenue = st.number_input(
            "Monthly Revenue Target (USD)",
            min_value=0,
            value=st.session_state.dynamic_params['monthly_revenue'],
            step=100000,
            format="%d",
            help="Target monthly revenue in USD"
        )
    
    with col2:
        token_price = st.number_input(
            "Token Price (USD)",
            min_value=0.01,
            value=st.session_state.dynamic_params['token_price'],
            step=0.01,
            format="%.2f",
            help="Current token price in USD"
        )
    
    with col3:
        delta = st.slider(
            "Delta Parameter",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.dynamic_params['delta'],
            step=0.01,
            help="Sensitivity to revenue changes (higher = more rewards)"
        )
    
    # Update session state with dynamic parameters
    st.session_state.dynamic_params = {
        'monthly_revenue': monthly_revenue,
        'token_price': token_price,
        'delta': delta
    }
    
    # Step 3: Airdrop Configuration
    st.subheader("3. Validator Airdrop Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        airdrop_start_amount = st.number_input(
            "Starting Airdrop Amount (Titan Tokens)",
            min_value=0,
            value=1000000,
            step=100000,
            format="%d",
            help="The amount of tokens to airdrop at the start of the schedule"
        )
        
        airdrop_end_amount = st.number_input(
            "Ending Airdrop Amount (Titan Tokens)",
            min_value=0,
            value=100000,
            step=10000,
            format="%d",
            help="The amount of tokens to airdrop at the end of the schedule"
        )
        
        airdrop_start_month = st.number_input(
            "Start Month",
            min_value=1,
            max_value=total_months,
            value=1,
            step=1,
            help="The month to start the airdrop schedule"
        )
        
        airdrop_end_month = st.number_input(
            "End Month",
            min_value=1,
            max_value=total_months,
            value=12,
            step=1,
            help="The month to end the airdrop schedule"
        )
    
    with col2:
        airdrop_schedule_type = st.selectbox(
            "Schedule Type",
            options=["Linear", "Exponential Decay", "Step Function"],
            help="The type of schedule to use for distributing airdrops"
        )
        
        if airdrop_schedule_type == "Step Function":
            num_steps = st.number_input(
                "Number of Steps",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Number of steps in the step function"
            )
        
        if airdrop_schedule_type == "Exponential Decay":
            decay_rate = st.slider(
                "Decay Rate",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Rate of exponential decay (higher = faster decay)"
            )
    
    # Calculate airdrop schedule based on configuration
    def calculate_airdrop_schedule():
        schedule = np.zeros(total_months)
        
        if airdrop_start_month > airdrop_end_month:
            st.error("Start month must be before end month")
            return schedule
            
        schedule_length = airdrop_end_month - airdrop_start_month + 1
        
        if airdrop_schedule_type == "Linear":
            # Linear interpolation between start and end amounts
            schedule[airdrop_start_month-1:airdrop_end_month] = np.linspace(
                airdrop_start_amount, airdrop_end_amount, schedule_length
            )
            
        elif airdrop_schedule_type == "Exponential Decay":
            # Exponential decay from start to end amount
            x = np.linspace(0, 1, schedule_length)
            decay = np.exp(-decay_rate * x)
            normalized_decay = (decay - decay[-1]) / (decay[0] - decay[-1])
            schedule[airdrop_start_month-1:airdrop_end_month] = (
                airdrop_start_amount * normalized_decay + 
                airdrop_end_amount * (1 - normalized_decay)
            )
            
        elif airdrop_schedule_type == "Step Function":
            # Create steps between start and end amounts
            steps = np.linspace(airdrop_start_amount, airdrop_end_amount, num_steps)
            step_size = schedule_length // (num_steps - 1)
            for i in range(num_steps - 1):
                start_idx = airdrop_start_month - 1 + i * step_size
                end_idx = start_idx + step_size
                schedule[start_idx:end_idx] = steps[i]
            # Fill any remaining months with the last step
            schedule[airdrop_start_month - 1 + (num_steps - 1) * step_size:airdrop_end_month] = steps[-1]
        
        return schedule
    
    if st.button("Generate Airdrop Schedule"):
        airdrop_schedule = calculate_airdrop_schedule()
        # Update the session state with the new schedule
        st.session_state.airdrop_schedule = airdrop_schedule.tolist()
        st.success("Airdrop schedule generated successfully!")
    
    # Step 4: Monthly Configuration
    st.subheader("4. Monthly Configuration")
    st.write("Configure the target staking percentage for each month and view the airdrop schedule.")
    
    # Create monthly configuration DataFrame
    monthly_config_df_data = {
        "Month": list(range(1, total_months + 1)),
        "Target Staking %": st.session_state.target_staking_pct,
    }
    
    # Add airdrop amounts if they exist
    if 'airdrop_schedule' in st.session_state:
        monthly_config_df_data["Airdrop Amount"] = st.session_state.airdrop_schedule
    
    # Create the DataFrame
    monthly_df_for_editing = pd.DataFrame(monthly_config_df_data)
    
    st.caption("Edit Target Staking % per month and view Airdrop Amounts:")
    edited_monthly_config_df = st.data_editor(
        monthly_df_for_editing,
        key="monthly_config_editor",
        num_rows="fixed",
        height=400,
        use_container_width=True,
        column_config={
            "Month": st.column_config.NumberColumn(
                "Month",
                disabled=True,
                format="%d"
            ),
            "Target Staking %": st.column_config.NumberColumn(
                "Target Staking %",
                help="Target staking percentage for each month (0.0 to 1.0).",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
            ),
            "Airdrop Amount": st.column_config.NumberColumn(
                "Airdrop Amount",
                help="Monthly airdrop amount in tokens (generated from airdrop configuration).",
                disabled=True,
                format="%d"
            )
        },
        hide_index=True
    )
    
    # Update session state with edited values
    st.session_state.target_staking_pct = edited_monthly_config_df["Target Staking %"].tolist()
    
    # Run simulation automatically after configuration
    if st.button("Run Simulation with Current Configuration"):
        st.session_state.simulation_results = None  # Clear previous results
        
        all_median_results_list = []
        trajectory_types = ["Flat", "Increasing", "Decreasing"]
        increase_rate_decimal = increasing_rate_pct_yr / 100.0
        decrease_rate_decimal = decreasing_rate_pct_yr / 100.0

        for trajectory_name in trajectory_types:
            current_trajectory_runs_dfs = [] 
            
            # Get the target staking percentages from the edited DataFrame
            target_staking_list_from_editor = edited_monthly_config_df["Target Staking %"].tolist()

            for i in range(num_simulations_per_trajectory):
                current_run_seed = seed + i 
                
                sim_results = simulate_apy_monthly(
                    total_supply=total_supply,
                    staking_reward_amt=staking_reward_amt,
                    staking_reward_timeline_yrs=staking_reward_timeline_yrs,
                    seed=current_run_seed,
                    initial_stake_pct_mean=initial_stake_pct_mean,
                    trajectory_type=trajectory_name.lower(),
                    increasing_rate=increase_rate_decimal,
                    decreasing_rate=decrease_rate_decimal,
                    stake_pct_trajectory_std=stake_pct_trajectory_std,
                    max_apy=max_apy,
                    target_lock_pct_by_mo_input=target_staking_list_from_editor,
                    f_sigma=f_sigma,
                    static_release_schedule=st.session_state.static_release_schedule,
                    dynamic_params=st.session_state.dynamic_params,
                    airdrop_schedule=st.session_state.airdrop_schedule
                )
                # Unpack the tuple into results_df and supply_breakdown_df
                sim_df, _ = sim_results
                current_trajectory_runs_dfs.append(sim_df)
            
            if current_trajectory_runs_dfs:
                # Concatenate all DataFrames for the current trajectory type
                concatenated_trajectory_df = pd.concat(current_trajectory_runs_dfs)
                # Group by 'Month' and calculate the median for all numeric columns
                median_df_for_trajectory = concatenated_trajectory_df.groupby('Month').median().reset_index()
                median_df_for_trajectory['Trajectory Type'] = trajectory_name
                # Add target staking percentage to each trajectory's DataFrame
                median_df_for_trajectory['Target Staking %'] = edited_monthly_config_df['Target Staking %'].values
                all_median_results_list.append(median_df_for_trajectory)
        
        if all_median_results_list:
            combined_median_results_df = pd.concat(all_median_results_list)
            st.session_state.simulation_results = combined_median_results_df
            st.success("Simulation completed! Switch to the 'APY Simulation' tab to view the results.")
        else:
            st.error("No simulation data generated. Please check your configuration.")

with simulation_tab:
    if 'simulation_results' in st.session_state and st.session_state.simulation_results is not None:
        st.header("Median Simulation Results by Trajectory")
        combined_median_results_df = st.session_state.simulation_results
        
        # Create two rows of plots
        staking_row1, staking_row2 = st.columns(2)
        
        with staking_row1:
            st.subheader("Median Staking Percentage")
            # Create base chart for staking percentage
            base_staking_chart = alt.Chart(combined_median_results_df).encode(
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('Staked Percentage:Q', title='Median Staking Percentage', axis=alt.Axis(format='%')),
            )
            
            # Add trajectory lines
            trajectory_lines = base_staking_chart.mark_line(strokeDash=[5,5]).encode(
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            )
            
            # Add target line
            target_line = base_staking_chart.mark_line(
                color='black',
                strokeDash=[2,2]
            ).encode(
                y=alt.Y('Target Staking %:Q', title='Median Staking Percentage', axis=alt.Axis(format='%'))
            )
            
            # Combine the charts
            staking_chart = (trajectory_lines + target_line).properties(
                width='container',
                height=300
            ).interactive()
            
            st.altair_chart(staking_chart, use_container_width=True)
            
            # Add legend explanation
            st.caption("""
            **Chart Legend:**
            - Colored dashed lines: Median staking percentage for each trajectory type
            - Black dashed line: Target staking percentage
            """)

        with staking_row2:
            st.subheader("Supply Chart (Log Scale)")
            # Create base chart for staked amount
            base_staked_chart = alt.Chart(combined_median_results_df).encode(
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('Staked Amount (M):Q', 
                       title='Amount (M-Tokens)', 
                       scale=alt.Scale(type='log')),
            )
            
            # Add trajectory lines
            staked_amount_lines = base_staked_chart.mark_line(strokeDash=[5,5]).encode(
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            )
            
            # Add circulating supply line
            circulating_supply_line = base_staked_chart.mark_line(
                color='black',
                strokeDash=[2,2]
            ).encode(
                y=alt.Y('Circulating Supply (M):Q', 
                       title='Amount (M-Tokens)',
                       scale=alt.Scale(type='log'))
            )
            
            # Add target staked amount line
            target_staked_line = base_staked_chart.mark_line(
                color='gray',
                strokeDash=[3,3]
            ).encode(
                y=alt.Y('Target Staked Amount (M):Q', 
                       title='Amount (M-Tokens)',
                       scale=alt.Scale(type='log'))
            )
            
            # Combine the charts
            staked_amount_chart = (staked_amount_lines + circulating_supply_line + target_staked_line).properties(
                width='container',
                height=300
            ).interactive()
            
            st.altair_chart(staked_amount_chart, use_container_width=True)
            st.caption("""
            **Chart Legend:**
            - Colored dashed lines: Median staked amount for each trajectory type
            - Black dashed line: Circulating supply amount
            - Gray dashed line: Target staked amount
            - Note: Y-axis is logarithmic scale
            """)

        # Create APY plots side by side
        apy_col1, apy_col2 = st.columns(2)
        
        # Calculate shared y-scale for APY plots
        max_apy = max(
            combined_median_results_df['APY with Airdrop'].max(),
            combined_median_results_df['APY without Airdrop'].max()
        )
        min_apy = min(
            combined_median_results_df['APY with Airdrop'].min(),
            combined_median_results_df['APY without Airdrop'].min()
        )
        
        with apy_col1:
            st.subheader("Validator APY with Airdrop")
            apy_with_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('APY with Airdrop:Q', 
                       title='Validator APY', 
                       axis=alt.Axis(format='%'),
                       scale=alt.Scale(domain=[min_apy, max_apy])),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(apy_with_chart, use_container_width=True)
        
        with apy_col2:
            st.subheader("Validator APY without Airdrop")
            apy_without_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('APY without Airdrop:Q', 
                       title='Validator APY', 
                       axis=alt.Axis(format='%'),
                       scale=alt.Scale(domain=[min_apy, max_apy])),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(apy_without_chart, use_container_width=True)

        # Create two columns for the remaining plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Median Validator Rewards Distributed")
            rewards_dist_chart = alt.Chart(combined_median_results_df).mark_line().encode(
                x=alt.X('Month:Q', title='Month'),
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
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('Validator Reward Pot (M):Q', title='Median Pot Size (M-Tokens)'),
                color=alt.Color('Trajectory Type:N', title='Trajectory Type')
            ).properties(
                width='container',
                height=300
            ).interactive()
            st.altair_chart(reward_pot_chart, use_container_width=True)
        
        # Add cumulative rewards at the bottom
        st.subheader("Median Cumulative Validator Rewards")
        cumulative_chart = alt.Chart(combined_median_results_df).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
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
        
        # Add supply breakdown table
        st.subheader("Supply Breakdown")
        st.write("This table shows the breakdown of supply components for each month:")
        st.write("- Circulating Supply: Total supply in circulation")
        st.write("- Staked Amount: Amount of tokens locked in staking")
        st.write("- Validator Rewards: Rewards distributed to validators")
        st.write("- Static Releases: Monthly releases from predefined buckets")
        st.write("- Dynamic Rewards: Monthly rewards based on revenue")
        st.write("- Airdrop Amount: Monthly airdrop distribution")
        
        # Get the supply breakdown from the first trajectory (they should be similar)
        first_trajectory = trajectory_types[0]
        first_trajectory_df = combined_median_results_df[combined_median_results_df['Trajectory Type'] == first_trajectory]
        
        # Create supply breakdown DataFrame
        supply_breakdown_df = pd.DataFrame({
            'Month': first_trajectory_df['Month'],
            'Year': first_trajectory_df['Year'],
            'Circulating Supply (M)': first_trajectory_df['Circulating Supply (M)'],
            'Staked Amount (M)': first_trajectory_df['Staked Amount (M)'],
            'Validator Rewards (M)': first_trajectory_df['Validator Rewards Distributed (M)'],
            'Static Releases (M)': first_trajectory_df['Static Releases (M)'],
            'Dynamic Rewards (M)': first_trajectory_df['Dynamic Rewards (M)'],
            'Airdrop Amount (M)': first_trajectory_df['Airdrop Amount (M)']
        })
        
        st.dataframe(supply_breakdown_df)
    else:
        st.info("Please complete the configuration and run the simulation to see results here.")
