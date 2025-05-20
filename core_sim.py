import numpy as np
import pandas as pd

# Define the Gaussian function
def f(staked_pct, target_pct, sigma=0.02):
    return np.exp(- (staked_pct - target_pct)**2 / (2 * sigma**2))

# # Define the simulation function
# def simulate_apy(
#     total_supply=2_000_000_000, 
#     staking_reward_amt=600_000_000, 
#     staking_reward_timeline_yrs=10, 
#     seed=42,
#     initial_stake_pct_mean=0.2, 
#     trajectory_type="flat", 
#     increasing_rate=0.02, 
#     decreasing_rate=0.02, 
#     stake_pct_trajectory_std=0.03,
#     max_apy=0.5,
#     target_lock_pct=0.2,
#     f_sigma=0.05, # This is the sigma for the Gaussian function f()
#     *, # Make subsequent arguments keyword-only
#     cs_pct_by_yr_input: list,
#     airdrop_by_year_input: list,
# ):
    
#     rng = np.random.RandomState(seed)

#     yearly_nominal_release = staking_reward_amt / staking_reward_timeline_yrs
#     validator_reward_pot = yearly_nominal_release
    
#     # Use inputs directly
#     cs_pct_by_yr = cs_pct_by_yr_input
#     airdrop_by_year = airdrop_by_year_input
    
#     # Ensure cs_pct_by_yr and airdrop_by_year have lengths equal to staking_reward_timeline_yrs
#     # This should be guaranteed by the calling code that prepares these lists based on the timeline.
#     # For robustness, one might add assertions here, but for this app, it's handled upstream.

#     target_lock_pct_by_yr = staking_reward_timeline_yrs * [target_lock_pct]
#     # Use the passed f_sigma to create the sigma_by_yr list
#     sigma_by_yr = staking_reward_timeline_yrs * [f_sigma]


#     apy_with_airdrop_yr = []
#     apy_without_airdrop_yr = []
#     staked_pct_by_yr = []
#     validator_reward_pot_by_yr = []
#     validator_rewards_distributed_by_yr = []
#     years = []
    
#     current_stake_mean = initial_stake_pct_mean

#     for yr in range(1, staking_reward_timeline_yrs + 1):
#         years.append(yr)
        
#         if yr > 1: # Adjust stake_pct_mean for years > 1 based on trajectory
#             if trajectory_type == "increasing":
#                 current_stake_mean = current_stake_mean * (1 + increasing_rate)
#             elif trajectory_type == "decreasing":
#                 current_stake_mean = current_stake_mean * (1 - decreasing_rate)
        
#         current_stake_mean = np.clip(current_stake_mean, 0.001, 0.999)

#         cs_nominal = total_supply * cs_pct_by_yr[yr-1]
#         target_stake_pct_for_year = target_lock_pct_by_yr[yr-1] # Use a distinct name
#         f_sigma_for_year = sigma_by_yr[yr-1] # Use a distinct name

#         stake_pct = rng.normal(loc=current_stake_mean, scale=stake_pct_trajectory_std)
#         stake_pct = np.clip(stake_pct, 0.001, 0.999) 
#         staked_amt = cs_nominal * stake_pct
        
#         release_pct = f(stake_pct, target_stake_pct_for_year, sigma=f_sigma_for_year)
#         remaining_pct = 1-release_pct
#         validator_rewards_release_amt = release_pct * validator_reward_pot
#         validator_reward_pot = (remaining_pct * validator_reward_pot) + yearly_nominal_release
        
#         apy_without_airdrop = (validator_rewards_release_amt / staked_amt) if staked_amt > 0 else 0
        
#         if apy_without_airdrop > max_apy:
#             actual_release_amt = staked_amt * max_apy
#             validator_reward_pot += (validator_rewards_release_amt - actual_release_amt)
#             validator_rewards_release_amt = actual_release_amt
#             apy_without_airdrop = max_apy
        
#         release_with_airdrop = validator_rewards_release_amt + airdrop_by_year[yr-1]
#         apy_with_airdrop = (release_with_airdrop / staked_amt) if staked_amt > 0 else 0

#         validator_reward_pot_by_yr.append(validator_reward_pot)
#         staked_pct_by_yr.append(stake_pct)
#         apy_with_airdrop_yr.append(apy_with_airdrop)
#         apy_without_airdrop_yr.append(apy_without_airdrop)
#         validator_rewards_distributed_by_yr.append(validator_rewards_release_amt)
    
#     results_df = pd.DataFrame({
#         'Year': years,
#         'Staked Percentage': staked_pct_by_yr,
#         'APY with Airdrop': apy_with_airdrop_yr,
#         'APY without Airdrop': apy_without_airdrop_yr,
#         'Validator Reward Pot (M)': [x/1e6 for x in validator_reward_pot_by_yr],
#         'Validator Rewards Distributed (M)': [x/1e6 for x in validator_rewards_distributed_by_yr],
#         'Cumulative Validator Rewards (M)': np.cumsum([x/1e6 for x in validator_rewards_distributed_by_yr])
#     })
    
#     return results_df

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
    cs_pct_by_mo_input: list,
    airdrop_by_mo_input: list,
    target_lock_pct_by_mo_input: list,  # This is now a list of monthly target percentages
):
    """
    Simulate APY on a monthly basis, with annualized rates.
    Similar to simulate_apy but with monthly granularity.
    
    Args:
        ... (other args same as simulate_apy)
        target_lock_pct: List of target staking percentages for each month
        cs_pct_by_mo_input: List of circulating supply percentages for each month
        airdrop_by_mo_input: List of airdrop amounts for each month
    
    Note:
        target_lock_pct, cs_pct_by_mo_input, and airdrop_by_mo_input must have length equal to
        staking_reward_timeline_yrs * 12
    """
    rng = np.random.RandomState(seed)

    # Convert yearly amounts to monthly
    monthly_nominal_release = staking_reward_amt / (staking_reward_timeline_yrs * 12)
    validator_reward_pot = monthly_nominal_release
    
    # Convert yearly rates to monthly
    monthly_increasing_rate = (1 + increasing_rate) ** (1/12) - 1
    monthly_decreasing_rate = (1 + decreasing_rate) ** (1/12) - 1
    
    # Validate input lengths
    expected_months = staking_reward_timeline_yrs * 12
    if len(cs_pct_by_mo_input) != expected_months:
        raise ValueError(f"cs_pct_by_mo_input must have length {expected_months} (years * 12), got {len(cs_pct_by_mo_input)}")
    if len(airdrop_by_mo_input) != expected_months:
        raise ValueError(f"airdrop_by_mo_input must have length {expected_months} (years * 12), got {len(airdrop_by_mo_input)}")
    if len(target_lock_pct_by_mo_input) != expected_months:
        raise ValueError(f"target_lock_pct must have length {expected_months} (years * 12), got {len(target_lock_pct_by_mo_input)}")
    
    # Use monthly inputs directly
    cs_pct_by_month = cs_pct_by_mo_input
    airdrop_by_month = airdrop_by_mo_input
    target_lock_pct_by_month = target_lock_pct_by_mo_input
    
    # Use the passed f_sigma to create the sigma_by_month list
    sigma_by_month = len(cs_pct_by_month) * [f_sigma]

    apy_with_airdrop_month = []
    apy_without_airdrop_month = []
    staked_pct_by_month = []
    validator_reward_pot_by_month = []
    validator_rewards_distributed_by_month = []
    months = []
    
    current_stake_mean = initial_stake_pct_mean
    total_months = staking_reward_timeline_yrs * 12

    for month in range(1, total_months + 1):
        months.append(month)
        
        if month > 1: # Adjust stake_pct_mean for months > 1 based on trajectory
            if trajectory_type == "increasing":
                current_stake_mean = current_stake_mean * (1 + monthly_increasing_rate)
            elif trajectory_type == "decreasing":
                current_stake_mean = current_stake_mean * (1 - monthly_decreasing_rate)
        
        current_stake_mean = np.clip(current_stake_mean, 0.001, 0.999)

        cs_nominal = total_supply * cs_pct_by_month[month-1]
        target_stake_pct_for_month = target_lock_pct_by_month[month-1]
        f_sigma_for_month = sigma_by_month[month-1]

        stake_pct = rng.normal(loc=current_stake_mean, scale=stake_pct_trajectory_std)
        stake_pct = np.clip(stake_pct, 0.001, 0.999) 
        staked_amt = cs_nominal * stake_pct
        
        release_pct = f(stake_pct, target_stake_pct_for_month, sigma=f_sigma_for_month)
        validator_rewards_release_amt = release_pct * validator_reward_pot
        validator_reward_pot -= validator_rewards_release_amt  # Subtract the released amount
        validator_reward_pot += monthly_nominal_release  # Add the nominal release for the next month
        
        # Calculate monthly APY and annualize it
        monthly_apy_without_airdrop = (validator_rewards_release_amt / staked_amt) if staked_amt > 0 else 0
        apy_without_airdrop = monthly_apy_without_airdrop * 12  # Annualize
        
        if apy_without_airdrop > max_apy:
            actual_release_amt = staked_amt * (max_apy / 12)  # Convert max APY to monthly rate
            validator_reward_pot += (validator_rewards_release_amt - actual_release_amt)  # Add back the difference
            validator_rewards_release_amt = actual_release_amt
            apy_without_airdrop = max_apy

        # print(month, release_pct, validator_rewards_release_amt, validator_reward_pot)
        
        release_with_airdrop = validator_rewards_release_amt + airdrop_by_month[month-1]
        monthly_apy_with_airdrop = (release_with_airdrop / staked_amt) if staked_amt > 0 else 0
        apy_with_airdrop = monthly_apy_with_airdrop * 12  # Annualize

        validator_reward_pot_by_month.append(validator_reward_pot)
        staked_pct_by_month.append(stake_pct)
        apy_with_airdrop_month.append(apy_with_airdrop)
        apy_without_airdrop_month.append(apy_without_airdrop)
        validator_rewards_distributed_by_month.append(validator_rewards_release_amt)
    
    # Create DataFrame with both month and year columns
    results_df = pd.DataFrame({
        'Month': months,
        'Year': [(m-1) // 12 + 1 for m in months],  # Convert month number to year
        'Staked Percentage': staked_pct_by_month,
        'Staked Amount (M)': [x/1e6 for x in [cs_nominal * pct for cs_nominal, pct in zip([total_supply * pct for pct in cs_pct_by_month], staked_pct_by_month)]],  # Calculate staked amount in millions
        'Circulating Supply (M)': [x/1e6 for x in [total_supply * pct for pct in cs_pct_by_month]],  # Calculate circulating supply in millions
        'Target Staked Amount (M)': [x/1e6 for x in [cs_nominal * target for cs_nominal, target in zip([total_supply * pct for pct in cs_pct_by_month], target_lock_pct_by_month)]],  # Calculate target staked amount in millions
        'APY with Airdrop': apy_with_airdrop_month,
        'APY without Airdrop': apy_without_airdrop_month,
        'Validator Reward Pot (M)': [x/1e6 for x in validator_reward_pot_by_month],
        'Validator Rewards Distributed (M)': [x/1e6 for x in validator_rewards_distributed_by_month],
        'Cumulative Validator Rewards (M)': np.cumsum([x/1e6 for x in validator_rewards_distributed_by_month])
    })
    
    return results_df