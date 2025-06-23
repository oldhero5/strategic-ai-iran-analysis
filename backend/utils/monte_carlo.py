"""
Monte Carlo Simulation and Sampling Utilities
Advanced probabilistic sampling methods for robust game theory analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from ..models.mcmc_model import GameState, Strategy, Outcome
from ..models.robust_gametheory import UncertaintyBounds


@dataclass
class SamplingConfig:
    """Configuration for Monte Carlo sampling"""
    n_samples: int = 10000
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = 42
    sampling_method: str = "lhs"  # "random", "lhs", "sobol", "halton"
    antithetic: bool = True  # Use antithetic variates for variance reduction


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    samples: np.ndarray
    summary_stats: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    convergence_metrics: Dict[str, float]
    sensitivity_indices: Optional[Dict[str, float]] = None


class AdvancedSampler:
    """
    Advanced sampling methods for Monte Carlo simulation.
    
    Implements:
    - Latin Hypercube Sampling (LHS)
    - Sobol sequences
    - Halton sequences
    - Antithetic variates
    - Control variates
    - Importance sampling
    """
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def sample_uniform(self, 
                      dimensions: int,
                      bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Sample from uniform distributions using specified method.
        
        Args:
            dimensions: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            
        Returns:
            Array of shape (n_samples, dimensions)
        """
        
        if self.config.sampling_method == "lhs":
            return self._latin_hypercube_sample(dimensions, bounds)
        elif self.config.sampling_method == "sobol":
            return self._sobol_sample(dimensions, bounds)
        elif self.config.sampling_method == "halton":
            return self._halton_sample(dimensions, bounds)
        else:
            return self._random_sample(dimensions, bounds)
    
    def _latin_hypercube_sample(self, 
                               dimensions: int,
                               bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Latin Hypercube Sampling for better space coverage"""
        
        n = self.config.n_samples
        
        # Generate LHS samples in unit hypercube
        samples = np.zeros((n, dimensions))
        
        for i in range(dimensions):
            # Create stratified samples
            segments = np.arange(n, dtype=float) / n
            uniform_within_segments = np.random.uniform(0, 1/n, n)
            stratified_samples = segments + uniform_within_segments
            
            # Permute to break correlation between dimensions
            permuted_samples = np.random.permutation(stratified_samples)
            samples[:, i] = permuted_samples
        
        # Transform to desired bounds
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = low + (high - low) * samples[:, i]
        
        return samples
    
    def _sobol_sample(self, 
                     dimensions: int,
                     bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Sobol quasi-random sequences"""
        try:
            from scipy.stats import qmc
            
            # Create Sobol sampler
            sampler = qmc.Sobol(d=dimensions, scramble=True)
            
            # Generate samples
            unit_samples = sampler.random(self.config.n_samples)
            
            # Transform to bounds
            samples = np.zeros_like(unit_samples)
            for i, (low, high) in enumerate(bounds):
                samples[:, i] = low + (high - low) * unit_samples[:, i]
            
            return samples
            
        except ImportError:
            # Fallback to LHS if scipy.stats.qmc not available
            return self._latin_hypercube_sample(dimensions, bounds)
    
    def _halton_sample(self, 
                      dimensions: int,
                      bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Halton quasi-random sequences"""
        try:
            from scipy.stats import qmc
            
            # Create Halton sampler
            sampler = qmc.Halton(d=dimensions, scramble=True)
            
            # Generate samples
            unit_samples = sampler.random(self.config.n_samples)
            
            # Transform to bounds
            samples = np.zeros_like(unit_samples)
            for i, (low, high) in enumerate(bounds):
                samples[:, i] = low + (high - low) * unit_samples[:, i]
            
            return samples
            
        except ImportError:
            # Fallback to LHS
            return self._latin_hypercube_sample(dimensions, bounds)
    
    def _random_sample(self, 
                      dimensions: int,
                      bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Standard random sampling"""
        
        n = self.config.n_samples
        samples = np.zeros((n, dimensions))
        
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = np.random.uniform(low, high, n)
        
        return samples
    
    def sample_correlated(self, 
                         means: np.ndarray,
                         covariance: np.ndarray,
                         antithetic: bool = True) -> np.ndarray:
        """
        Sample from multivariate normal with correlation structure.
        
        Args:
            means: Mean vector
            covariance: Covariance matrix
            antithetic: Use antithetic variates for variance reduction
            
        Returns:
            Correlated samples
        """
        
        n_base = self.config.n_samples // 2 if antithetic else self.config.n_samples
        
        # Generate base samples
        base_samples = np.random.multivariate_normal(means, covariance, n_base)
        
        if antithetic:
            # Create antithetic variates
            antithetic_samples = 2 * means - base_samples
            samples = np.vstack([base_samples, antithetic_samples])
        else:
            samples = base_samples
        
        return samples


class VarianceReduction:
    """
    Variance reduction techniques for Monte Carlo simulation.
    
    Implements:
    - Control variates
    - Antithetic variates
    - Importance sampling
    - Stratified sampling
    """
    
    @staticmethod
    def antithetic_variates(base_function: Callable,
                           uniform_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply antithetic variates technique.
        
        Args:
            base_function: Function to evaluate
            uniform_samples: Base uniform samples in [0,1]
            
        Returns:
            Tuple of (base_results, antithetic_results)
        """
        
        # Evaluate base samples
        base_results = base_function(uniform_samples)
        
        # Create antithetic samples (1 - U)
        antithetic_samples = 1.0 - uniform_samples
        antithetic_results = base_function(antithetic_samples)
        
        return base_results, antithetic_results
    
    @staticmethod
    def control_variates(target_results: np.ndarray,
                        control_results: np.ndarray,
                        control_mean: float) -> np.ndarray:
        """
        Apply control variates for variance reduction.
        
        Args:
            target_results: Results from target function
            control_results: Results from control function (known mean)
            control_mean: Known mean of control function
            
        Returns:
            Variance-reduced estimates
        """
        
        # Estimate optimal control coefficient
        covariance = np.cov(target_results, control_results)[0, 1]
        control_variance = np.var(control_results)
        
        if control_variance > 0:
            optimal_c = -covariance / control_variance
        else:
            optimal_c = 0
        
        # Apply control variate
        reduced_results = target_results + optimal_c * (control_results - control_mean)
        
        return reduced_results
    
    @staticmethod
    def importance_sampling(target_function: Callable,
                           importance_samples: np.ndarray,
                           importance_weights: np.ndarray) -> np.ndarray:
        """
        Apply importance sampling.
        
        Args:
            target_function: Function to evaluate
            importance_samples: Samples from importance distribution
            importance_weights: Importance weights (target_pdf / importance_pdf)
            
        Returns:
            Importance sampling estimates
        """
        
        # Evaluate function at importance samples
        function_values = target_function(importance_samples)
        
        # Weight by importance ratios
        weighted_values = function_values * importance_weights
        
        return weighted_values


class GameStateSimulator:
    """
    Monte Carlo simulator for game theory scenarios.
    
    Specialized for sampling and simulating game states with complex
    dependencies and constraints.
    """
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampler = AdvancedSampler(config)
        
        # Parameter correlation structure (based on expert knowledge)
        self.correlation_matrix = self._build_correlation_matrix()
    
    def _build_correlation_matrix(self) -> np.ndarray:
        """Build correlation matrix for game state parameters"""
        
        # Parameters: [regime_cohesion, economic_stress, proxy_support, 
        #             external_support, nuclear_progress, oil_price_norm]
        
        correlation = np.array([
            [1.00, -0.60, 0.40, 0.30, -0.20, 0.10],  # regime_cohesion
            [-0.60, 1.00, -0.30, -0.20, 0.40, 0.20],  # economic_stress
            [0.40, -0.30, 1.00, 0.50, 0.20, -0.10],  # proxy_support
            [0.30, -0.20, 0.50, 1.00, 0.30, -0.20],  # external_support
            [-0.20, 0.40, 0.20, 0.30, 1.00, 0.10],   # nuclear_progress
            [0.10, 0.20, -0.10, -0.20, 0.10, 1.00]   # oil_price_norm
        ])
        
        return correlation
    
    def simulate_game_states(self, 
                           base_state: Optional[GameState] = None,
                           correlation_strength: float = 0.7) -> List[GameState]:
        """
        Simulate realistic game states with parameter correlations.
        
        Args:
            base_state: Base game state to perturb around
            correlation_strength: Strength of parameter correlations (0-1)
            
        Returns:
            List of simulated game states
        """
        
        if base_state is None:
            # Use default base state
            base_means = np.array([0.4, 0.8, 0.2, 0.3, 0.7, 0.5])  # Normalized oil price
        else:
            base_means = np.array([
                base_state.regime_cohesion,
                base_state.economic_stress,
                base_state.proxy_support,
                base_state.external_support,
                base_state.nuclear_progress,
                (base_state.oil_price - 50) / 100  # Normalize to [0,1]
            ])
        
        # Build covariance matrix
        std_devs = np.array([0.15, 0.10, 0.15, 0.20, 0.15, 0.20])
        
        # Scale correlation by strength parameter
        scaled_correlation = (correlation_strength * self.correlation_matrix + 
                            (1 - correlation_strength) * np.eye(6))
        
        covariance = np.outer(std_devs, std_devs) * scaled_correlation
        
        # Sample correlated parameters
        raw_samples = self.sampler.sample_correlated(base_means, covariance)
        
        # Clip to valid ranges and create game states
        game_states = []
        
        for sample in raw_samples:
            # Clip to valid ranges
            regime_cohesion = np.clip(sample[0], 0, 1)
            economic_stress = np.clip(sample[1], 0, 1)
            proxy_support = np.clip(sample[2], 0, 1)
            external_support = np.clip(sample[3], 0, 1)
            nuclear_progress = np.clip(sample[4], 0, 1)
            oil_price = np.clip(sample[5] * 100 + 50, 50, 150)  # Denormalize
            
            game_states.append(GameState(
                regime_cohesion=regime_cohesion,
                economic_stress=economic_stress,
                proxy_support=proxy_support,
                oil_price=oil_price,
                external_support=external_support,
                nuclear_progress=nuclear_progress
            ))
        
        return game_states
    
    def scenario_analysis(self, 
                         scenario_definitions: Dict[str, Dict[str, float]],
                         n_samples_per_scenario: int = 1000) -> Dict[str, List[GameState]]:
        """
        Generate samples for multiple predefined scenarios.
        
        Args:
            scenario_definitions: Dict mapping scenario names to parameter means
            n_samples_per_scenario: Number of samples per scenario
            
        Returns:
            Dict mapping scenario names to lists of game states
        """
        
        results = {}
        
        original_n_samples = self.config.n_samples
        self.config.n_samples = n_samples_per_scenario
        
        for scenario_name, params in scenario_definitions.items():
            # Create base state from scenario parameters
            base_state = GameState(
                regime_cohesion=params.get("regime_cohesion", 0.4),
                economic_stress=params.get("economic_stress", 0.8),
                proxy_support=params.get("proxy_support", 0.2),
                oil_price=params.get("oil_price", 97.0),
                external_support=params.get("external_support", 0.3),
                nuclear_progress=params.get("nuclear_progress", 0.7)
            )
            
            # Simulate variations around this scenario
            scenario_states = self.simulate_game_states(base_state)
            results[scenario_name] = scenario_states
        
        # Restore original configuration
        self.config.n_samples = original_n_samples
        
        return results
    
    def sensitivity_sampling(self, 
                           parameter_name: str,
                           parameter_range: Tuple[float, float],
                           other_params_fixed: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate samples for sensitivity analysis of a specific parameter.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_range: (min, max) range for the parameter
            other_params_fixed: If True, fix other parameters; if False, sample them
            
        Returns:
            Dict with parameter values and corresponding samples
        """
        
        # Generate parameter values to test
        param_values = np.linspace(parameter_range[0], parameter_range[1], 
                                 self.config.n_samples)
        
        if other_params_fixed:
            # Fix other parameters at their base values
            base_values = {
                "regime_cohesion": 0.4,
                "economic_stress": 0.8,
                "proxy_support": 0.2,
                "oil_price": 97.0,
                "external_support": 0.3,
                "nuclear_progress": 0.7
            }
            
            game_states = []
            for param_val in param_values:
                state_params = base_values.copy()
                state_params[parameter_name] = param_val
                game_states.append(GameState(**state_params))
        
        else:
            # Sample other parameters while varying the target parameter
            game_states = []
            base_state = GameState(
                regime_cohesion=0.4,
                economic_stress=0.8,
                proxy_support=0.2,
                oil_price=97.0,
                external_support=0.3,
                nuclear_progress=0.7
            )
            
            # Generate base samples
            base_samples = self.simulate_game_states(base_state)
            
            for i, param_val in enumerate(param_values):
                # Take a base sample and modify the target parameter
                if i < len(base_samples):
                    state = base_samples[i]
                    state_dict = state.__dict__.copy()
                    state_dict[parameter_name] = param_val
                    game_states.append(GameState(**state_dict))
        
        return {
            "parameter_values": param_values,
            "game_states": game_states
        }


class ConvergenceDiagnostics:
    """
    Convergence diagnostics for Monte Carlo simulations.
    
    Implements various metrics to assess simulation quality and convergence.
    """
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray, 
                            max_lag: Optional[int] = None) -> float:
        """
        Compute effective sample size accounting for autocorrelation.
        
        Args:
            samples: Array of samples
            max_lag: Maximum lag for autocorrelation calculation
            
        Returns:
            Effective sample size
        """
        
        n = len(samples)
        if max_lag is None:
            max_lag = min(n // 4, 200)
        
        # Compute autocorrelation function
        autocorr = np.correlate(samples - np.mean(samples), 
                               samples - np.mean(samples), 
                               mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first negative autocorrelation or use max_lag
        cutoff = max_lag
        for i in range(1, min(len(autocorr), max_lag)):
            if autocorr[i] <= 0:
                cutoff = i
                break
        
        # Compute integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
        
        # Effective sample size
        ess = n / (2 * tau_int + 1)
        
        return max(1, ess)
    
    @staticmethod
    def monte_carlo_standard_error(samples: np.ndarray) -> float:
        """
        Compute Monte Carlo standard error.
        
        Args:
            samples: Array of samples
            
        Returns:
            Monte Carlo standard error
        """
        
        n = len(samples)
        ess = ConvergenceDiagnostics.effective_sample_size(samples)
        
        # Standard error accounting for autocorrelation
        mcse = np.std(samples) / np.sqrt(ess)
        
        return mcse
    
    @staticmethod
    def gelman_rubin_diagnostic(chains: List[np.ndarray]) -> float:
        """
        Compute Gelman-Rubin potential scale reduction factor (R-hat).
        
        Args:
            chains: List of sample chains
            
        Returns:
            R-hat statistic (should be close to 1.0 for convergence)
        """
        
        if len(chains) < 2:
            return 1.0
        
        # Number of chains and samples per chain
        m = len(chains)
        n = len(chains[0])
        
        # Chain means and overall mean
        chain_means = [np.mean(chain) for chain in chains]
        overall_mean = np.mean(chain_means)
        
        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in chains])
        
        # Marginal posterior variance estimate
        var_hat = ((n - 1) / n) * W + (1 / n) * B
        
        # R-hat
        if W > 0:
            r_hat = np.sqrt(var_hat / W)
        else:
            r_hat = 1.0
        
        return r_hat
    
    @staticmethod
    def convergence_summary(samples: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """
        Compute comprehensive convergence summary.
        
        Args:
            samples: Single array or list of chains
            
        Returns:
            Dictionary with convergence metrics
        """
        
        if isinstance(samples, list):
            # Multiple chains
            all_samples = np.concatenate(samples)
            r_hat = ConvergenceDiagnostics.gelman_rubin_diagnostic(samples)
        else:
            # Single chain
            all_samples = samples
            r_hat = 1.0
        
        ess = ConvergenceDiagnostics.effective_sample_size(all_samples)
        mcse = ConvergenceDiagnostics.monte_carlo_standard_error(all_samples)
        
        return {
            "n_samples": len(all_samples),
            "effective_sample_size": ess,
            "monte_carlo_se": mcse,
            "r_hat": r_hat,
            "efficiency": ess / len(all_samples),
            "converged": r_hat < 1.1 and ess > 400
        }


def create_default_simulator() -> GameStateSimulator:
    """Create default game state simulator with reasonable settings"""
    
    config = SamplingConfig(
        n_samples=5000,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_seed=42,
        sampling_method="lhs",
        antithetic=True
    )
    
    return GameStateSimulator(config)


if __name__ == "__main__":
    # Example usage
    simulator = create_default_simulator()
    
    print("Creating Monte Carlo simulator...")
    
    # Generate base game states
    print("Simulating game states...")
    base_state = GameState(
        regime_cohesion=0.4,
        economic_stress=0.9,
        proxy_support=0.2,
        oil_price=97.0,
        external_support=0.3,
        nuclear_progress=0.7
    )
    
    simulated_states = simulator.simulate_game_states(base_state)
    
    print(f"Generated {len(simulated_states)} game states")
    
    # Extract parameter distributions
    regime_cohesions = [state.regime_cohesion for state in simulated_states]
    nuclear_progress = [state.nuclear_progress for state in simulated_states]
    
    print(f"Regime cohesion: mean={np.mean(regime_cohesions):.3f}, "
          f"std={np.std(regime_cohesions):.3f}")
    print(f"Nuclear progress: mean={np.mean(nuclear_progress):.3f}, "
          f"std={np.std(nuclear_progress):.3f}")
    
    # Convergence diagnostics
    diagnostics = ConvergenceDiagnostics.convergence_summary(np.array(regime_cohesions))
    print(f"\nConvergence diagnostics:")
    for metric, value in diagnostics.items():
        print(f"  {metric}: {value}")
    
    # Scenario analysis
    scenarios = {
        "regime_collapse": {"regime_cohesion": 0.1, "economic_stress": 0.95},
        "nuclear_breakout": {"nuclear_progress": 0.95, "external_support": 0.1},
        "oil_shock": {"oil_price": 130, "economic_stress": 0.7}
    }
    
    print("\nScenario analysis...")
    scenario_results = simulator.scenario_analysis(scenarios, n_samples_per_scenario=1000)
    
    for scenario_name, states in scenario_results.items():
        avg_nuclear = np.mean([s.nuclear_progress for s in states])
        print(f"{scenario_name}: avg nuclear progress = {avg_nuclear:.3f}")