import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import copy
from dataclasses import dataclass, field

## TODO:
##   1 - Update static emissions to be vested over an alternate schedule.  This
##       means 2 additional features: 1) there is an initial unlock (could be 0)
##       as well as a vesting schedule.  This is important for the Testnet Token Allocation
##   2 - Enforce max-cap on deal emissions.  NOTE that we don't need to
##       worry about deal rewards, because those are taken from the market
##       and then given to the participants.  This means that essentially
##       they are neutral from a circulating supply perspective, because they
##       are taken from CS, and then given to the miners (still CS from protocol perspective)


@dataclass
class EmissionBucket:
    """Represents a token emission bucket for a specific stakeholder."""
    name: str
    emissions_per_month: float
    start_month: int  # Month index when emissions start (0 = simulation start)
    cap: Optional[float] = None  # Maximum emissions for this bucket
    months_to_emit: Optional[int] = None  # Alternative to cap: number of months to emit
    initial_supply: float = 0.0  # Supply at TGE (Token Generation Event)
    
    def __post_init__(self):
        # Validate that either cap or months_to_emit is provided
        if self.cap is None and self.months_to_emit is None:
            raise ValueError("Either 'cap' or 'months_to_emit' must be provided")
        
        # If months_to_emit is provided but cap is not, calculate cap
        if self.cap is None and self.months_to_emit is not None:
            self.cap = self.emissions_per_month * self.months_to_emit
            
        # Ensure initial_supply is not negative
        if self.initial_supply < 0:
            raise ValueError("Initial supply cannot be negative")


@dataclass
class Deal:
    """Represents a deal in the Titan token ecosystem."""
    value: float  # Deal value in dollars
    start_month: int  # Month index when the deal is made
    geography: str  # Geography identifier
    token_price: float  # Token price at the time of the deal


@dataclass
class TitanSimulationConfig:
    """Configuration for the Titan token simulation."""
    buckets: List[EmissionBucket]
    delta: float  # Deal-to-token conversion rate
    beta: float  # Decay rate for inflationary rewards
    geography_multipliers: Dict[str, float]  # Multipliers for different geographies
    buyback_portion: float  # Portion of deal value used for buybacks
    simulation_months: int  # Total simulation period in months
    buyback_vesting_months: int = 12  # Number of months over which buybacks are vested
    validator_total_rewards: float = 0.0  # Total amount of tokens to be emitted as validator rewards
    validator_half_life_months: float = 0.0  # Half-life of validator rewards in months
    locking_vector: Optional[np.ndarray] = None  # Vector of locked token amounts for each month
    
    def __post_init__(self):
        # Validate buyback portion is between 0 and 1
        if not 0 <= self.buyback_portion <= 1:
            raise ValueError("Buyback portion must be between 0 and 1")
        
        # Validate delta and beta are positive
        if self.delta <= 0:
            raise ValueError("Delta must be positive")
        if self.beta <= 0:
            raise ValueError("Beta must be positive")
        
        # Validate buyback vesting months is positive
        if self.buyback_vesting_months <= 0:
            raise ValueError("Buyback vesting months must be positive")
            
        # Validate validator rewards parameters
        if self.validator_total_rewards < 0:
            raise ValueError("Validator total rewards cannot be negative")
        if self.validator_half_life_months < 0:
            raise ValueError("Validator half-life months cannot be negative")
            
        # Validate locking vector if provided
        if self.locking_vector is not None:
            if len(self.locking_vector) != self.simulation_months:
                raise ValueError(f"Locking vector length ({len(self.locking_vector)}) must match simulation months ({self.simulation_months})")
            if np.any(self.locking_vector < 0):
                raise ValueError("Locking vector cannot contain negative values")
        else:
            self.locking_vector = np.zeros(self.simulation_months)
            


class TitanTokenSimulation:
    """
    A flexible simulation class for the Titan token ecosystem.
    """
    
    def __init__(self, config: TitanSimulationConfig):
        """
        Initialize the simulation with the given configuration.
        
        Args:
            config: Configuration parameters for the simulation
        """
        self.config = copy.deepcopy(config)
        self.simulation_months = np.arange(config.simulation_months)
        self.results = None
        
    def run(self, deals: List[Deal]) -> pd.DataFrame:
        """
        Run the simulation with the provided list of deals.
        
        Args:
            deals: List of deals to include in the simulation
            
        Returns:
            DataFrame containing simulation results over time
        """
        # Initialize the results dataframe with months as index
        self.results = pd.DataFrame(index=self.simulation_months)
        self.results.index.name = 'Month'
        
        # Process fixed emissions for each bucket
        for bucket in self.config.buckets:
            self._process_fixed_emissions(bucket)
        
        # Process deal-based emissions
        for deal in deals:
            self._process_deal(deal)
            
        # Process validator rewards
        self._process_validator_rewards()
        
        # Calculate cumulative emissions and total supply
        self._calculate_totals()
        
        return self.results
    
    def _process_fixed_emissions(self, bucket: EmissionBucket) -> None:
        """
        Process fixed emissions for a single bucket.
        
        Args:
            bucket: The emission bucket to process
        """
        # Create column for this bucket if it doesn't exist
        bucket_col = f"fixed_{bucket.name}"
        if bucket_col not in self.results.columns:
            self.results[bucket_col] = 0.0
        
        # Add initial supply at month 0
        if bucket.initial_supply > 0:
            self.results.loc[0, bucket_col] += bucket.initial_supply
        
        # Process monthly emissions
        emitted_so_far = bucket.initial_supply
        for month in self.simulation_months[bucket.start_month:]:
            # Check if we've hit the cap
            if emitted_so_far >= bucket.cap or month >= (bucket.months_to_emit + bucket.start_month):
                break
                
            # Calculate emissions for this month (respecting the cap)
            month_emission = min(bucket.emissions_per_month, bucket.cap - emitted_so_far)
            self.results.loc[month, bucket_col] += month_emission
            emitted_so_far += month_emission
    
    def _process_deal(self, deal: Deal) -> None:
        """
        Process a single deal and its emissions.
        
        Args:
            deal: The deal to process
        """
        # Get geography multiplier (default to 1.0 if not found)
        geo_multiplier = self.config.geography_multipliers.get(deal.geography, 1.0)
        
        # Calculate total rewards for this deal
        total_rewards = self.config.delta * deal.value * geo_multiplier / self.config.beta
        # TODO: ensure that this total_rewards stays under overall token supply.
        
        ##########################################################
        # Create deal emissions columns if they don't exist
        deal_inflationary_rewards_emissions = f"deal_emissions"
        # buyback_col = f"buybacks"
        total_emissions_col = f"total_dynamic_emissions"
        buyback_reduction_col = f"buyback_reduction"
        vested_buybacks_col = f"vested_buybacks"
        
        if deal_inflationary_rewards_emissions not in self.results.columns:
            self.results[deal_inflationary_rewards_emissions] = 0.0
        if total_emissions_col not in self.results.columns:
            self.results[total_emissions_col] = 0.0
        if buyback_reduction_col not in self.results.columns:
            self.results[buyback_reduction_col] = 0.0
        if vested_buybacks_col not in self.results.columns:
            self.results[vested_buybacks_col] = 0.0
        ##########################################################
        
        ##########################################################
        #### Compute the amount given to miners through buybacks associated with deals
        # Calculate total buyback amount
        buyback_amount = deal.value * self.config.buyback_portion
        buyback_tokens = buyback_amount / deal.token_price  # Convert dollar amount to tokens
        distribution_period = min(self.config.buyback_vesting_months, len(self.simulation_months) - deal.start_month)
        buyback_monthly_distribution_amt = buyback_tokens / distribution_period
        self.results.loc[deal.start_month, buyback_reduction_col] += buyback_tokens

        for month in range(deal.start_month, min(deal.start_month + distribution_period, len(self.simulation_months))):
            self.results.loc[month, vested_buybacks_col] += buyback_monthly_distribution_amt
            # self.results.loc[month, buyback_col] += monthly_buyback
            # self.results.loc[month, buyback_reduction_col] += monthly_buyback
        ##########################################################
        
        # Process inflationary emissions related to a deal
        for month in self.simulation_months[deal.start_month:]:
            months_since_deal = month - deal.start_month
            
            # R(t) = delta * V * f(g) * exp(-Beta * (t-t_0))
            emission_rate = (
                self.config.delta * 
                buyback_amount * 
                geo_multiplier * 
                np.exp(-self.config.beta * months_since_deal)
            )
            
            self.results.loc[month, deal_inflationary_rewards_emissions] += emission_rate
            
        # Update total emissions by adding this deal's contributions
        self.results[total_emissions_col] += self.results[deal_inflationary_rewards_emissions] #+ self.results[vested_buybacks_col]
    
    def _process_validator_rewards(self) -> None:
        """
        Process validator rewards using an exponentially decaying emission schedule.
        The emission rate follows: R(t) = R_0 * exp(-λt)
        where λ = ln(2)/half_life and R_0 is chosen to ensure total emissions match validator_total_rewards.
        """
        # Create column for validator rewards if it doesn't exist
        validator_col = "validator_rewards"
        if validator_col not in self.results.columns:
            self.results[validator_col] = 0.0
            
        # Calculate decay constant λ from half-life
        decay_constant = np.log(2) / self.config.validator_half_life_months
        
        # Calculate initial rate R_0 to ensure total emissions match validator_total_rewards
        # We need to solve: ∫(R_0 * exp(-λt))dt from 0 to ∞ = validator_total_rewards
        # This gives: R_0 = validator_total_rewards * λ
        initial_rate = self.config.validator_total_rewards * decay_constant
        
        # Calculate emissions for each month
        for month in self.simulation_months:
            emission_rate = initial_rate * np.exp(-decay_constant * month)
            self.results.loc[month, validator_col] = emission_rate
            
        # Update total emissions
        if 'total_dynamic_emissions' in self.results.columns:
            self.results['total_dynamic_emissions'] += self.results[validator_col]
    
    def _calculate_totals(self) -> None:
        """Calculate cumulative emissions and total supply."""
        # Get all fixed emission columns
        fixed_cols = [col for col in self.results.columns if col.startswith('fixed_')]
        
        # Calculate total fixed emissions per month
        self.results['total_fixed_emissions'] = self.results[fixed_cols].sum(axis=1)
        
        # Calculate total emissions per month (fixed + deal-based + buybacks)
        # Note: total_emissions column already contains deal_emissions + buybacks
        self.results['total_emissions'] = (
            self.results['total_fixed_emissions'] + 
            self.results['total_dynamic_emissions']
        )
        
        # Calculate cumulative total emissions
        self.results['cumulative_emissions'] = self.results['total_emissions'].cumsum()
        
        # Calculate net buyback impact (vested buybacks - buyback reduction)
        self.results['net_buyback_impact'] = (
            self.results['vested_buybacks'] - 
            self.results['buyback_reduction']
        )
        
        # Calculate circulating supply
        # Start with cumulative emissions, then adjust for buyback impact
        self.results['circulating_supply'] = (
            self.results['cumulative_emissions'] + 
            self.results['net_buyback_impact'].cumsum()
        )
        
        # Handle locking if configured
        self.results['locked_supply'] = self.config.locking_vector
        self.results['circulating_supply'] -= self.results['locked_supply']
    