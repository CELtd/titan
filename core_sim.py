import numpy as np
import pandas as pd

# Define the Gaussian function
def f(staked_pct, target_pct, sigma=0.02):
    return np.exp(- (staked_pct - target_pct)**2 / (2 * sigma**2))

# Static release schedule buckets
STATIC_RELEASE_BUCKETS = {
    'Titan Labs': [8333333, 16666667, 25000000, 33333333, 58333333, 83333333, 108333333, 133333333, 158333333, 183333333, 208333333, 233333333, 258333333, 283333333, 308333333, 333333333, 358333333, 383333333, 400000000, 400000000, 400000000, 400000000, 400000000],
    'Titan Foundation': [4166667, 8333333, 12500000, 16666667, 29166667, 41666667, 54166667, 66666667, 79166667, 91666667, 104166667, 116666667, 129166667, 141666667, 154166667, 166666667, 179166667, 191666667, 200000000, 200000000, 200000000, 200000000, 200000000],
    'Fundraising Seed': [944444, 1888889, 2833333, 3777778, 6611111, 9444444, 12277778, 15111111, 17944444, 20777778, 23611111, 26444444, 29277778, 32111111, 34000000, 34000000, 34000000, 34000000, 34000000, 34000000, 34000000, 34000000, 34000000],
    'Fundraising A': [4611111, 9222222, 13833333, 18444444, 32277778, 46111111, 59944444, 73777778, 87611111, 101444444, 115277778, 129111111, 142944444, 156777778, 166000000, 166000000, 166000000, 166000000, 166000000, 166000000, 166000000, 166000000, 166000000],
    'Ecosystem': [2083333, 4166667, 6250000, 8333333, 14583333, 20833333, 27083333, 33333333, 39583333, 45833333, 52083333, 58333333, 64583333, 70833333, 77083333, 83333333, 89583333, 95833333, 100000000, 100000000, 100000000, 100000000, 100000000],
    'Testnet Token Allocation': [7500000, 7500000, 22500000, 30000000, 52500000, 75000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000],
    'Market Making': [7500000, 7500000, 22500000, 30000000, 52500000, 75000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000, 90000000],
    'Dev Grants': [0, 500000, 1000000, 1500000, 3000000, 4500000, 6000000, 7500000, 9000000, 10500000, 12000000, 13500000, 15000000, 16500000, 18000000, 19500000, 21000000, 22500000, 24000000, 25500000, 27000000, 28500000, 30000000],
    'RPGF': [0, 500000, 1000000, 1500000, 3000000, 4500000, 6000000, 7500000, 9000000, 10500000, 12000000, 13500000, 15000000, 16500000, 18000000, 19500000, 21000000, 22500000, 24000000, 25500000, 27000000, 28500000, 30000000]
}

def convert_quarterly_to_monthly(quarterly_values, total_months):
    """Convert quarterly cumulative values to monthly values."""
    monthly_values = []
    for q in range(len(quarterly_values)):
        if q == 0:
            # For first quarter, divide total by 3 for each month
            monthly_value = quarterly_values[q] / 3
            monthly_values.extend([monthly_value] * 3)
        else:
            # For subsequent quarters, calculate the difference from previous quarter
            # and divide by 3 for each month
            quarter_difference = quarterly_values[q] - quarterly_values[q-1]
            monthly_value = quarter_difference / 3
            monthly_values.extend([monthly_value] * 3)
    
    # If we need more months than quarters * 3, extend with the last monthly value
    if len(monthly_values) < total_months:
        last_monthly_value = monthly_values[-1]
        monthly_values.extend([last_monthly_value] * (total_months - len(monthly_values)))
    
    return monthly_values

def simulate_apy_monthly(
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
    f_sigma=0.05,
    *, # Make subsequent arguments keyword-only
    target_lock_pct_by_mo_input: list,
    static_release_schedule: dict = None,
    dynamic_params: dict = None,
    airdrop_schedule: list = None,
):
    """
    Simulate APY on a monthly basis, with annualized rates.
    Similar to simulate_apy but with monthly granularity.
    
    Args:
        ... (other args same as before)
        target_lock_pct_by_mo_input: List of target staking percentages for each month
        static_release_schedule: Dictionary of quarterly release schedules for each bucket
        dynamic_params: Dictionary containing monthly_revenue, token_price, delta
        airdrop_schedule: List of monthly airdrop amounts
    """
    rng = np.random.RandomState(seed)

    # Convert yearly amounts to monthly
    monthly_nominal_release = staking_reward_amt / (staking_reward_timeline_yrs * 12)
    validator_reward_pot = monthly_nominal_release
    
    # Convert yearly rates to monthly
    monthly_increasing_rate = (1 + increasing_rate) ** (1/12) - 1
    monthly_decreasing_rate = (1 + decreasing_rate) ** (1/12) - 1
    
    # Validate input lengths
    total_months = staking_reward_timeline_yrs * 12
    if len(target_lock_pct_by_mo_input) != total_months:
        raise ValueError(f"target_lock_pct must have length {total_months} (years * 12), got {len(target_lock_pct_by_mo_input)}")
    
    # Initialize tracking variables
    current_stake_mean = initial_stake_pct_mean
    previous_month_supply = {
        'cumulative_supply': 0,
        'staked_amount': 0,
        'validator_rewards': 0
    }
    
    # Initialize results storage
    results = {
        'Month': [],
        'Year': [],
        'Staked Percentage': [],
        'Staked Amount (M)': [],
        'Circulating Supply (M)': [],
        'Target Staked Amount (M)': [],
        'APY with Airdrop': [],
        'APY without Airdrop': [],
        'Validator Reward Pot (M)': [],
        'Validator Rewards Distributed (M)': [],
        'Cumulative Validator Rewards (M)': [],
        'Static Releases (M)': [],
        'Dynamic Rewards (M)': [],
        'Airdrop Amount (M)': [],
        'Total Monthly Emissions (M)': []
    }
    
    for month in range(1, total_months + 1):
        # Calculate supply components for this month
        supply_components = calculate_monthly_supply_components(
            month=month,
            static_release_schedule=static_release_schedule,
            dynamic_params=dynamic_params,
            airdrop_schedule=airdrop_schedule,
            previous_month_supply=previous_month_supply,
            total_supply=total_supply
        )
        
        # Adjust stake percentage based on trajectory
        if month > 1:
            if trajectory_type == "increasing":
                current_stake_mean = current_stake_mean * (1 + monthly_increasing_rate)
            elif trajectory_type == "decreasing":
                current_stake_mean = current_stake_mean * (1 - monthly_decreasing_rate)
        
        current_stake_mean = np.clip(current_stake_mean, 0.001, 0.999)
        
        # Calculate staking based on current circulating supply
        stake_pct = rng.normal(loc=current_stake_mean, scale=stake_pct_trajectory_std)
        stake_pct = np.clip(stake_pct, 0.001, 0.999)
        staked_amt = supply_components['cumulative_supply'] * stake_pct
        
        # Calculate validator rewards
        target_stake_pct = target_lock_pct_by_mo_input[month-1]
        release_pct = f(stake_pct, target_stake_pct, sigma=f_sigma)
        validator_rewards = release_pct * validator_reward_pot
        
        # Update validator reward pot
        validator_reward_pot -= validator_rewards
        validator_reward_pot += monthly_nominal_release
        
        # Calculate APY
        monthly_apy_without_airdrop = (validator_rewards / staked_amt) if staked_amt > 0 else 0
        apy_without_airdrop = monthly_apy_without_airdrop * 12
        
        # Apply max APY cap if needed
        if apy_without_airdrop > max_apy:
            actual_rewards = staked_amt * (max_apy / 12)
            validator_reward_pot += (validator_rewards - actual_rewards)
            validator_rewards = actual_rewards
            apy_without_airdrop = max_apy
        
        # Calculate APY with airdrop
        total_rewards = validator_rewards + supply_components['airdrop']
        monthly_apy_with_airdrop = (total_rewards / staked_amt) if staked_amt > 0 else 0
        apy_with_airdrop = monthly_apy_with_airdrop * 12
        
        # Update previous month's supply for next iteration
        previous_month_supply = {
            'cumulative_supply': supply_components['cumulative_supply'],
            'staked_amount': staked_amt,
            'validator_rewards': validator_rewards
        }
        
        # Store results
        results['Month'].append(month)
        results['Year'].append((month-1) // 12 + 1)
        results['Staked Percentage'].append(stake_pct)
        results['Staked Amount (M)'].append(staked_amt / 1e6)
        results['Circulating Supply (M)'].append(supply_components['cumulative_supply'] / 1e6)
        results['Target Staked Amount (M)'].append(supply_components['cumulative_supply'] * target_stake_pct / 1e6)
        results['APY with Airdrop'].append(apy_with_airdrop)
        results['APY without Airdrop'].append(apy_without_airdrop)
        results['Validator Reward Pot (M)'].append(validator_reward_pot / 1e6)
        results['Validator Rewards Distributed (M)'].append(validator_rewards / 1e6)
        results['Static Releases (M)'].append(sum(supply_components['static_releases'].values()) / 1e6)
        results['Dynamic Rewards (M)'].append(supply_components['dynamic_rewards'] / 1e6)
        results['Airdrop Amount (M)'].append(supply_components['airdrop'] / 1e6)
        results['Total Monthly Emissions (M)'].append(supply_components['total_monthly_emissions'] / 1e6)
    
    # Calculate cumulative validator rewards
    results['Cumulative Validator Rewards (M)'] = np.cumsum(results['Validator Rewards Distributed (M)'])
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create supply breakdown DataFrame
    supply_breakdown_df = pd.DataFrame({
        'Month': results['Month'],
        'Year': results['Year'],
        'Base Circulating Supply (M)': [x/1e6 for x in [total_supply * pct for pct in [supply_components['circulating_supply_pct'] for _ in range(total_months)]]],
        'Staked Amount (M)': results['Staked Amount (M)'],
        'Validator Rewards (M)': results['Validator Rewards Distributed (M)'],
        'Actual Circulating Supply (M)': results['Circulating Supply (M)']
    })
    
    return results_df, supply_breakdown_df

def calculate_monthly_supply_components(
    month: int,
    static_release_schedule: dict,
    dynamic_params: dict,
    airdrop_schedule: list,
    previous_month_supply: dict,
    total_supply: int
) -> dict:
    """
    Calculate all supply components for a single month.
    
    Args:
        month: Current month (1-indexed)
        static_release_schedule: Dictionary of quarterly release schedules
        dynamic_params: Dictionary containing monthly_revenue, token_price, delta
        airdrop_schedule: List of monthly airdrop amounts
        previous_month_supply: Dictionary of previous month's supply metrics
        total_supply: Total token supply
        
    Returns:
        Dictionary containing:
        - static_releases: Dict of monthly releases from each bucket
        - dynamic_rewards: Monthly deal inflationary rewards
        - airdrop: Monthly airdrop amount
        - total_monthly_emissions: Sum of all emissions
        - cumulative_supply: Total supply up to this month
        - circulating_supply_pct: Percentage of total supply circulating
    """
    # Calculate static releases for this month
    static_releases = {}
    for bucket, quarterly_values in static_release_schedule.items():
        # Convert quarterly values to monthly
        quarterly_values_list = list(quarterly_values.values())
        # Calculate total months needed (23 quarters * 3 months per quarter)
        total_months = len(quarterly_values_list) * 3
        monthly_values = convert_quarterly_to_monthly(quarterly_values_list, total_months)
        # Ensure we have enough months
        if month <= len(monthly_values):
            static_releases[bucket] = monthly_values[month - 1]
        else:
            # If we're past the schedule, use the last month's value
            static_releases[bucket] = monthly_values[-1]
    
    # Calculate dynamic rewards
    monthly_revenue = dynamic_params['monthly_revenue']
    token_price = dynamic_params['token_price']
    delta = dynamic_params['delta']
    dynamic_rewards = (monthly_revenue * token_price * delta) / 12
    
    # Get airdrop amount
    if month <= len(airdrop_schedule):
        airdrop = airdrop_schedule[month - 1]
    else:
        # If we're past the schedule, use the last month's value
        airdrop = airdrop_schedule[-1]
    
    # Calculate total monthly emissions
    total_monthly_emissions = sum(static_releases.values()) + dynamic_rewards + airdrop
    
    # Calculate cumulative supply
    if month == 1:
        cumulative_supply = total_monthly_emissions
    else:
        cumulative_supply = previous_month_supply['cumulative_supply'] + total_monthly_emissions
    
    # Calculate circulating supply percentage
    circulating_supply_pct = cumulative_supply / total_supply
    
    return {
        'static_releases': static_releases,
        'dynamic_rewards': dynamic_rewards,
        'airdrop': airdrop,
        'total_monthly_emissions': total_monthly_emissions,
        'cumulative_supply': cumulative_supply,
        'circulating_supply_pct': circulating_supply_pct
    }