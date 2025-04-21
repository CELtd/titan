def linear_vesting(age, period):
    """Linear vesting over the period"""
    return min(age / period, 1.0) if period > 0 else 1.0

def cliff_vesting(age, period, cliff_months=3):
    """No vesting until cliff, then linear"""
    if age < cliff_months:
        return 0.0
    else:
        return min((age - cliff_months) / (period - cliff_months), 1.0) if period > cliff_months else 1.0

def accelerated_vesting(age, period, power=0.5):
    """Accelerated vesting (front-loaded, using a power function)"""
    return min((age / period) ** power, 1.0) if period > 0 else 1.0

def decelerated_vesting(age, period, power=2.0):
    """Decelerated vesting (back-loaded, using a power function)"""
    return min((age / period) ** power, 1.0) if period > 0 else 1.0

def step_vesting(age, period, steps=4):
    """Step-wise vesting (e.g., 25% increments)"""
    step_size = period / steps
    completed_steps = int(age / step_size)
    return min(completed_steps / steps, 1.0)

class TokenEmissionSource:
    """
    Represents a single source of token emissions with its own vesting schedule.
    """
    def __init__(
        self,
        name,
        emission_per_month,
        vesting_period,
        emission_cap=float('inf'),
        start_month=1,
        end_month=float('inf'),
        vesting_function=None,
        vesting_function_kwargs={}
    ):
        self.name = name
        self.emission_per_month = emission_per_month
        self.vesting_period = vesting_period
        self.emission_cap = emission_cap
        self.start_month = start_month
        self.end_month = end_month
        self.vesting_function_kwargs = vesting_function_kwargs
        # Default to linear vesting if no function is provided
        if vesting_function is None:
            # self.vesting_function = lambda age, period: min(age / period, 1.0) if period > 0 else 1.0
            self.vesting_function = linear_vesting
        else:
            self.vesting_function = vesting_function
            
        # Initialize vault to track batches
        self.vault = []
        
        # Track source-specific metrics
        self.metrics = {
            'month': [],
            'newly_emitted': [],
            'newly_vested': [],
            'total_circulating': [],
            'total_locked': [],
            'total_emission': []
        }
        
    def simulate_month(self, current_month):
        """
        Simulate one month of emission and vesting for this source.
        
        Args:
            current_month (int): The current month being simulated
            
        Returns:
            dict: Metrics for this month
        """
        # Check if we should emit tokens this month
        if current_month < self.start_month or current_month > self.end_month:
            newly_emitted = 0
        else:
            # Calculate total emission to date
            total_emission = sum(batch['amount'] for batch in self.vault)
            
            # Check if we've reached the emission cap
            if total_emission >= self.emission_cap:
                newly_emitted = 0
            else:
                newly_emitted = self.emission_per_month
                
                # If this would exceed the cap, only emit up to the cap
                if total_emission + newly_emitted > self.emission_cap:
                    newly_emitted = self.emission_cap - total_emission
        
        new_batch = {
            'emission_month': current_month,
            'amount': newly_emitted,
            'vested_amount': 0,
            'locked_amount': newly_emitted
        }
        self.vault.append(new_batch)
        
        newly_vested = 0
        total_circulating = 0
        total_locked = 0
        
        # Update each batch in the vault
        for batch in self.vault:
            # Calculate batch age
            batch_age = current_month - batch['emission_month']
            
            # Previous vested amount
            previous_vested = batch['vested_amount']
            
            # Calculate new vested percentage
            vested_fraction = self.vesting_function(batch_age, self.vesting_period, **self.vesting_function_kwargs)
            
            # Update batch vested and locked amounts
            new_vested_amount = batch['amount'] * vested_fraction
            new_locked_amount = batch['amount'] - new_vested_amount
            
            # Calculate newly vested tokens this month
            newly_vested_from_batch = new_vested_amount - previous_vested
            newly_vested += newly_vested_from_batch
            
            # Update batch status
            batch['vested_amount'] = new_vested_amount
            batch['locked_amount'] = new_locked_amount
            
            # Add to totals
            total_circulating += batch['vested_amount']
            total_locked += batch['locked_amount']
        
        # Calculate total emission to date
        total_emission = sum(batch['amount'] for batch in self.vault)
        
        # Store metrics for this month
        self.metrics['month'].append(current_month)
        self.metrics['newly_emitted'].append(newly_emitted)
        self.metrics['newly_vested'].append(newly_vested)
        self.metrics['total_circulating'].append(total_circulating)
        self.metrics['total_locked'].append(total_locked)
        self.metrics['total_emission'].append(total_emission)
        
        # Return metrics for this month for aggregation
        return {
            'newly_emitted': newly_emitted,
            'newly_vested': newly_vested,
            'total_circulating': total_circulating,
            'total_locked': total_locked,
            'total_emission': total_emission
        }


class TokenNetwork:
    """
    Manages multiple token emission sources and tracks network-wide metrics.
    """
    def __init__(self):
        self.sources = {}
        
        # Network-wide metrics
        self.metrics = {
            'month': [],
            'newly_emitted': [],
            'newly_vested': [],
            'total_circulating': [],
            'total_locked': [],
            'total_emission': [],
            'sources_newly_emitted': {},
            'sources_newly_vested': {},
            'sources_total_circulating': {},
            'sources_total_locked': {},
            'sources_total_emission': {}
        }
    
    def add_emission_source(self, source):
        """
        Add a token emission source to the network.
        
        Args:
            source (TokenEmissionSource): The emission source to add
        """
        self.sources[source.name] = source
        
        # Initialize source-specific tracking in network metrics
        self.metrics['sources_newly_emitted'][source.name] = []
        self.metrics['sources_newly_vested'][source.name] = []
        self.metrics['sources_total_circulating'][source.name] = []
        self.metrics['sources_total_locked'][source.name] = []
        self.metrics['sources_total_emission'][source.name] = []
    
    def simulate(self, months_to_simulate):
        """
        Simulate token emissions across all sources.
        
        Args:
            months_to_simulate (int): Number of months to simulate
            
        Returns:
            dict: Network-wide metrics
        """
        # Reset metrics
        self.metrics = {
            'month': [],
            'newly_emitted': [],
            'newly_vested': [],
            'total_circulating': [],
            'total_locked': [],
            'total_emission': [],
            'sources_newly_emitted': {src: [] for src in self.sources},
            'sources_newly_vested': {src: [] for src in self.sources},
            'sources_total_circulating': {src: [] for src in self.sources},
            'sources_total_locked': {src: [] for src in self.sources},
            'sources_total_emission': {src: [] for src in self.sources}
        }
        
        # Simulate each month
        for current_month in range(1, months_to_simulate + 1):
            month_newly_emitted = 0
            month_newly_vested = 0
            month_total_circulating = 0
            month_total_locked = 0
            month_total_emission = 0
            
            # Simulate each source
            for source_name, source in self.sources.items():
                source_metrics = source.simulate_month(current_month)
                
                # Aggregate metrics
                month_newly_emitted += source_metrics['newly_emitted']
                month_newly_vested += source_metrics['newly_vested']
                month_total_circulating += source_metrics['total_circulating']
                month_total_locked += source_metrics['total_locked']
                month_total_emission += source_metrics['total_emission']
                
                # Track source-specific metrics in network
                self.metrics['sources_newly_emitted'][source_name].append(source_metrics['newly_emitted'])
                self.metrics['sources_newly_vested'][source_name].append(source_metrics['newly_vested'])
                self.metrics['sources_total_circulating'][source_name].append(source_metrics['total_circulating'])
                self.metrics['sources_total_locked'][source_name].append(source_metrics['total_locked'])
                self.metrics['sources_total_emission'][source_name].append(source_metrics['total_emission'])
            
            # Store network-wide metrics for this month
            self.metrics['month'].append(current_month)
            self.metrics['newly_emitted'].append(month_newly_emitted)
            self.metrics['newly_vested'].append(month_newly_vested)
            self.metrics['total_circulating'].append(month_total_circulating)
            self.metrics['total_locked'].append(month_total_locked)
            self.metrics['total_emission'].append(month_total_emission)
        
        return self.metrics
    
    def get_source_metrics(self, source_name):
        """
        Get metrics for a specific source.
        
        Args:
            source_name (str): Name of the source
            
        Returns:
            dict: Source-specific metrics
        """
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not found")
        
        return self.sources[source_name].metrics
