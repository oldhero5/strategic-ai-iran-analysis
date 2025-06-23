"""
Robust Game Theory Model with MCMC Integration
Combines traditional game theory with Bayesian uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from .mcmc_model import BayesianGameModel, Strategy, Outcome, GameState
from .bayesian_engine import BayesianInferenceEngine, StrategyRecommendation
from .players import USA, Iran, Israel


@dataclass
class UncertaintyBounds:
    """Structure for uncertainty quantification"""
    mean: float
    std: float
    ci_lower: float  # 95% confidence interval
    ci_upper: float
    percentile_25: float
    percentile_75: float


@dataclass
class RobustOutcome:
    """Game outcome with uncertainty quantification"""
    outcome: Outcome
    probability: UncertaintyBounds
    utility_usa: UncertaintyBounds
    utility_iran: UncertaintyBounds
    utility_israel: UncertaintyBounds
    risk_metrics: Dict[str, float]


@dataclass
class StrategyAnalysis:
    """Complete strategy analysis with uncertainty"""
    strategy: Strategy
    expected_utility: UncertaintyBounds
    outcomes: List[RobustOutcome]
    risk_assessment: Dict[str, UncertaintyBounds]
    robustness_score: float  # How robust is this strategy to uncertainty?
    regret_analysis: Dict[str, float]  # Minimax regret analysis


class RobustGameTheoryModel:
    """
    Enhanced game theory model with MCMC-based uncertainty quantification.
    
    Features:
    - Probabilistic outcome modeling with credible intervals
    - Robust strategy evaluation under parameter uncertainty
    - Minimax regret analysis for worst-case scenarios
    - Sensitivity analysis with uncertainty propagation
    - Dynamic belief updating based on new evidence
    """
    
    def __init__(self, 
                 mcmc_samples: int = 2000,
                 random_seed: int = 42):
        
        self.mcmc_samples = mcmc_samples
        self.random_seed = random_seed
        
        # Initialize Bayesian components
        self.bayesian_engine = BayesianInferenceEngine()
        self.mcmc_model = self.bayesian_engine.mcmc_model
        
        # Game theory components
        self.players = {
            "USA": USA(),
            "Iran": Iran(), 
            "Israel": Israel()
        }
        
        # Current beliefs and uncertainties
        self.parameter_distributions = self._initialize_parameter_distributions()
        self.strategy_cache = {}
        
    def _initialize_parameter_distributions(self) -> Dict[str, Dict[str, float]]:
        """Initialize parameter uncertainty distributions"""
        return {
            "regime_cohesion": {
                "distribution": "beta",
                "alpha": 2.0,
                "beta": 3.0,
                "support": (0, 1)
            },
            "economic_stress": {
                "distribution": "beta", 
                "alpha": 4.0,
                "beta": 1.5,
                "support": (0, 1)
            },
            "nuclear_progress": {
                "distribution": "beta",
                "alpha": 3.0,
                "beta": 1.5, 
                "support": (0, 1)
            },
            "proxy_support": {
                "distribution": "beta",
                "alpha": 1.0,
                "beta": 4.0,
                "support": (0, 1)
            },
            "external_support": {
                "distribution": "beta",
                "alpha": 1.5,
                "beta": 3.0,
                "support": (0, 1)
            },
            "oil_price": {
                "distribution": "normal",
                "mean": 97.0,
                "std": 15.0,
                "support": (50, 150)
            }
        }
    
    def sample_game_states(self, n_samples: int = 1000) -> List[GameState]:
        """Sample game states from parameter uncertainty distributions"""
        
        samples = []
        
        for _ in range(n_samples):
            # Sample each parameter from its distribution
            regime_cohesion = np.random.beta(
                self.parameter_distributions["regime_cohesion"]["alpha"],
                self.parameter_distributions["regime_cohesion"]["beta"]
            )
            
            economic_stress = np.random.beta(
                self.parameter_distributions["economic_stress"]["alpha"],
                self.parameter_distributions["economic_stress"]["beta"]
            )
            
            nuclear_progress = np.random.beta(
                self.parameter_distributions["nuclear_progress"]["alpha"],
                self.parameter_distributions["nuclear_progress"]["beta"]
            )
            
            proxy_support = np.random.beta(
                self.parameter_distributions["proxy_support"]["alpha"],
                self.parameter_distributions["proxy_support"]["beta"]
            )
            
            external_support = np.random.beta(
                self.parameter_distributions["external_support"]["alpha"],
                self.parameter_distributions["external_support"]["beta"]
            )
            
            oil_price = np.clip(
                np.random.normal(
                    self.parameter_distributions["oil_price"]["mean"],
                    self.parameter_distributions["oil_price"]["std"]
                ),
                50, 150
            )
            
            samples.append(GameState(
                regime_cohesion=regime_cohesion,
                economic_stress=economic_stress,
                proxy_support=proxy_support,
                oil_price=oil_price,
                external_support=external_support,
                nuclear_progress=nuclear_progress
            ))
        
        return samples
    
    def analyze_strategy_robustly(self, 
                                 strategy: Strategy,
                                 base_game_state: Optional[GameState] = None,
                                 n_samples: int = 1000) -> StrategyAnalysis:
        """
        Analyze strategy with full uncertainty quantification.
        
        Args:
            strategy: Strategy to analyze
            base_game_state: Base game state (if None, samples from uncertainty)
            n_samples: Number of Monte Carlo samples
            
        Returns:
            StrategyAnalysis with uncertainty bounds and robustness metrics
        """
        
        # Sample game states
        if base_game_state is None:
            game_states = self.sample_game_states(n_samples)
        else:
            # Perturb base state with uncertainty
            game_states = self._perturb_game_state(base_game_state, n_samples)
        
        # Collect outcomes across samples
        outcome_samples = {outcome: [] for outcome in Outcome}
        utility_samples = {"USA": [], "Iran": [], "Israel": []}
        
        for game_state in game_states:
            # Get strategy recommendation for this game state
            recommendation = self.bayesian_engine.recommend_strategy(game_state)
            
            # Sample outcomes from MCMC model
            if self.mcmc_model.trace is None:
                self.mcmc_model.build_model()
                self.mcmc_model.sample_posterior(draws=500, tune=250, chains=2)
            
            strategy_results = self.mcmc_model.analyze_strategies()
            
            if strategy in strategy_results:
                for outcome in Outcome:
                    if outcome in strategy_results[strategy]:
                        prob_mean, _, _ = strategy_results[strategy][outcome]
                        outcome_samples[outcome].append(prob_mean)
            
            # Sample utilities (simplified - would use full utility model)
            utilities = self._compute_utilities(game_state, strategy)
            for player, utility in utilities.items():
                utility_samples[player].append(utility)
        
        # Create uncertainty bounds for outcomes
        robust_outcomes = []
        for outcome in Outcome:
            if outcome_samples[outcome]:
                prob_bounds = self._create_uncertainty_bounds(outcome_samples[outcome])
                
                # Create utility bounds (simplified)
                usa_utility = self._create_uncertainty_bounds(utility_samples["USA"])
                iran_utility = self._create_uncertainty_bounds(utility_samples["Iran"])
                israel_utility = self._create_uncertainty_bounds(utility_samples["Israel"])
                
                # Risk metrics
                risk_metrics = self._compute_risk_metrics(outcome_samples[outcome])
                
                robust_outcomes.append(RobustOutcome(
                    outcome=outcome,
                    probability=prob_bounds,
                    utility_usa=usa_utility,
                    utility_iran=iran_utility,
                    utility_israel=israel_utility,
                    risk_metrics=risk_metrics
                ))
        
        # Overall expected utility with uncertainty
        total_utilities = []
        for i in range(len(utility_samples["USA"])):
            total_utility = 0.0
            for outcome in Outcome:
                if i < len(outcome_samples[outcome]):
                    total_utility += outcome_samples[outcome][i] * utility_samples["USA"][i]
            total_utilities.append(total_utility)
        
        expected_utility_bounds = self._create_uncertainty_bounds(total_utilities)
        
        # Risk assessment with uncertainty
        risk_assessment = self._assess_strategy_risks_robust(strategy, game_states)
        
        # Robustness score
        robustness_score = self._compute_robustness_score(total_utilities, robust_outcomes)
        
        # Regret analysis
        regret_analysis = self._compute_regret_analysis(strategy, game_states)
        
        return StrategyAnalysis(
            strategy=strategy,
            expected_utility=expected_utility_bounds,
            outcomes=robust_outcomes,
            risk_assessment=risk_assessment,
            robustness_score=robustness_score,
            regret_analysis=regret_analysis
        )
    
    def _perturb_game_state(self, 
                           base_state: GameState, 
                           n_samples: int) -> List[GameState]:
        """Create perturbed versions of base game state"""
        
        perturbed_states = []
        
        for _ in range(n_samples):
            # Add noise to each parameter
            noise_scale = 0.1  # 10% relative noise
            
            regime_cohesion = np.clip(
                base_state.regime_cohesion + np.random.normal(0, noise_scale * base_state.regime_cohesion),
                0, 1
            )
            
            economic_stress = np.clip(
                base_state.economic_stress + np.random.normal(0, noise_scale * base_state.economic_stress),
                0, 1
            )
            
            nuclear_progress = np.clip(
                base_state.nuclear_progress + np.random.normal(0, noise_scale * base_state.nuclear_progress),
                0, 1
            )
            
            proxy_support = np.clip(
                base_state.proxy_support + np.random.normal(0, noise_scale * base_state.proxy_support),
                0, 1
            )
            
            external_support = np.clip(
                base_state.external_support + np.random.normal(0, noise_scale * base_state.external_support),
                0, 1
            )
            
            oil_price = np.clip(
                base_state.oil_price + np.random.normal(0, noise_scale * base_state.oil_price),
                50, 150
            )
            
            perturbed_states.append(GameState(
                regime_cohesion=regime_cohesion,
                economic_stress=economic_stress,
                proxy_support=proxy_support,
                oil_price=oil_price,
                external_support=external_support,
                nuclear_progress=nuclear_progress
            ))
        
        return perturbed_states
    
    def _create_uncertainty_bounds(self, samples: List[float]) -> UncertaintyBounds:
        """Create uncertainty bounds from samples"""
        
        if not samples:
            return UncertaintyBounds(0, 0, 0, 0, 0, 0)
        
        samples_array = np.array(samples)
        
        return UncertaintyBounds(
            mean=np.mean(samples_array),
            std=np.std(samples_array),
            ci_lower=np.percentile(samples_array, 2.5),
            ci_upper=np.percentile(samples_array, 97.5),
            percentile_25=np.percentile(samples_array, 25),
            percentile_75=np.percentile(samples_array, 75)
        )
    
    def _compute_utilities(self, 
                          game_state: GameState, 
                          strategy: Strategy) -> Dict[str, float]:
        """Compute player utilities for given game state and strategy"""
        
        # Simplified utility computation - would use full preference models
        base_utilities = {
            "USA": 0.5,
            "Iran": 0.3,
            "Israel": 0.4
        }
        
        # Adjust based on game state
        if game_state.regime_cohesion < 0.3:
            base_utilities["Iran"] -= 0.2
            base_utilities["USA"] -= 0.1
        
        if game_state.nuclear_progress > 0.7:
            base_utilities["Israel"] -= 0.3
            base_utilities["USA"] -= 0.2
        
        # Adjust based on strategy
        if strategy in [Strategy.ESCALATION_DIPLOMACY, Strategy.ESCALATION_ULTIMATUM]:
            base_utilities["Iran"] -= 0.2
            base_utilities["Israel"] += 0.1
        
        return base_utilities
    
    def _compute_risk_metrics(self, probability_samples: List[float]) -> Dict[str, float]:
        """Compute risk metrics for outcome probabilities"""
        
        if not probability_samples:
            return {}
        
        probs = np.array(probability_samples)
        
        return {
            "value_at_risk_95": np.percentile(probs, 95),  # 95th percentile
            "expected_shortfall": np.mean(probs[probs >= np.percentile(probs, 95)]),
            "volatility": np.std(probs),
            "skewness": stats.skew(probs),
            "tail_risk": np.mean(probs > np.percentile(probs, 90))
        }
    
    def _assess_strategy_risks_robust(self, 
                                    strategy: Strategy, 
                                    game_states: List[GameState]) -> Dict[str, UncertaintyBounds]:
        """Assess strategy risks with uncertainty quantification"""
        
        risk_types = ["escalation_risk", "miscalculation_risk", "third_party_risk", "domestic_risk"]
        risk_samples = {risk_type: [] for risk_type in risk_types}
        
        for game_state in game_states:
            # Base risk assessment
            base_risks = self.bayesian_engine._assess_strategy_risks(strategy, game_state)
            
            for risk_type in risk_types:
                if risk_type in base_risks:
                    # Add noise to risk assessment
                    noisy_risk = base_risks[risk_type] * (1 + np.random.normal(0, 0.1))
                    risk_samples[risk_type].append(max(0, min(1, noisy_risk)))
        
        # Create uncertainty bounds for each risk type
        risk_bounds = {}
        for risk_type, samples in risk_samples.items():
            risk_bounds[risk_type] = self._create_uncertainty_bounds(samples)
        
        return risk_bounds
    
    def _compute_robustness_score(self, 
                                 utility_samples: List[float],
                                 outcomes: List[RobustOutcome]) -> float:
        """
        Compute robustness score - how well does strategy perform across uncertainty.
        
        Higher score means more robust (less sensitive to parameter uncertainty).
        """
        
        if not utility_samples:
            return 0.0
        
        utilities = np.array(utility_samples)
        
        # Robustness metrics
        mean_utility = np.mean(utilities)
        worst_case_10pct = np.percentile(utilities, 10)  # 10th percentile
        coefficient_of_variation = np.std(utilities) / np.abs(mean_utility) if mean_utility != 0 else float('inf')
        
        # Combine into single robustness score (0-1, higher is better)
        robustness = (
            0.4 * min(mean_utility, 1.0) +  # Reward high mean utility
            0.4 * min(worst_case_10pct + 1.0, 1.0) +  # Reward good worst case
            0.2 * max(0, 1.0 - coefficient_of_variation)  # Penalize high variability
        )
        
        return max(0, min(1, robustness))
    
    def _compute_regret_analysis(self, 
                               strategy: Strategy,
                               game_states: List[GameState]) -> Dict[str, float]:
        """
        Compute minimax regret analysis.
        
        Regret = difference between chosen strategy and best possible strategy
        in each scenario.
        """
        
        regrets = []
        
        for game_state in game_states:
            # Get utility of chosen strategy
            chosen_utility = self._compute_utilities(game_state, strategy)["USA"]
            
            # Find best possible utility in this scenario
            best_utility = chosen_utility
            for alt_strategy in Strategy:
                alt_utility = self._compute_utilities(game_state, alt_strategy)["USA"]
                best_utility = max(best_utility, alt_utility)
            
            # Regret for this scenario
            regret = best_utility - chosen_utility
            regrets.append(regret)
        
        regrets_array = np.array(regrets)
        
        return {
            "max_regret": np.max(regrets_array),
            "mean_regret": np.mean(regrets_array),
            "regret_95th_percentile": np.percentile(regrets_array, 95),
            "fraction_scenarios_with_regret": np.mean(regrets_array > 0.01)
        }
    
    def compare_strategies_robust(self, 
                                 strategies: List[Strategy],
                                 base_game_state: Optional[GameState] = None) -> Dict[Strategy, StrategyAnalysis]:
        """
        Compare multiple strategies with full uncertainty quantification.
        
        Returns ranking with robustness considerations.
        """
        
        results = {}
        
        for strategy in strategies:
            analysis = self.analyze_strategy_robustly(strategy, base_game_state)
            results[strategy] = analysis
        
        return results
    
    def sensitivity_analysis_robust(self, 
                                   parameter: str,
                                   parameter_range: Tuple[float, float],
                                   n_points: int = 10,
                                   base_game_state: Optional[GameState] = None) -> Dict[str, Any]:
        """
        Perform robust sensitivity analysis with uncertainty propagation.
        
        Args:
            parameter: Parameter name to vary
            parameter_range: (min, max) range for parameter
            n_points: Number of points to test
            base_game_state: Base game state
            
        Returns:
            Sensitivity analysis results with uncertainty bounds
        """
        
        if base_game_state is None:
            base_game_state = GameState(
                regime_cohesion=0.4,
                economic_stress=0.8,
                proxy_support=0.2,
                oil_price=97.0,
                external_support=0.3,
                nuclear_progress=0.7
            )
        
        parameter_values = np.linspace(parameter_range[0], parameter_range[1], n_points)
        
        results = {
            "parameter_values": parameter_values,
            "strategies": {}
        }
        
        for strategy in Strategy:
            strategy_results = {
                "expected_utility_mean": [],
                "expected_utility_ci_lower": [],
                "expected_utility_ci_upper": [],
                "robustness_scores": []
            }
            
            for param_value in parameter_values:
                # Create modified game state
                modified_state_dict = asdict(base_game_state)
                modified_state_dict[parameter] = param_value
                modified_state = GameState(**modified_state_dict)
                
                # Analyze strategy robustly
                analysis = self.analyze_strategy_robustly(strategy, modified_state, n_samples=200)
                
                strategy_results["expected_utility_mean"].append(analysis.expected_utility.mean)
                strategy_results["expected_utility_ci_lower"].append(analysis.expected_utility.ci_lower)
                strategy_results["expected_utility_ci_upper"].append(analysis.expected_utility.ci_upper)
                strategy_results["robustness_scores"].append(analysis.robustness_score)
            
            results["strategies"][strategy] = strategy_results
        
        return results
    
    def export_robust_analysis(self, 
                              filepath: str,
                              analyses: Dict[Strategy, StrategyAnalysis]) -> None:
        """Export robust analysis results to files"""
        
        # Convert to serializable format
        export_data = {}
        
        for strategy, analysis in analyses.items():
            export_data[strategy.value] = {
                "expected_utility": asdict(analysis.expected_utility),
                "robustness_score": analysis.robustness_score,
                "regret_analysis": analysis.regret_analysis,
                "outcomes": [
                    {
                        "outcome": outcome.outcome.value,
                        "probability": asdict(outcome.probability),
                        "risk_metrics": outcome.risk_metrics
                    }
                    for outcome in analysis.outcomes
                ]
            }
        
        # Export to JSON
        import json
        with open(f"{filepath}_robust_analysis.json", 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Export to CSV for spreadsheet analysis
        df_rows = []
        for strategy, analysis in analyses.items():
            for outcome in analysis.outcomes:
                df_rows.append({
                    "strategy": strategy.value,
                    "outcome": outcome.outcome.value,
                    "prob_mean": outcome.probability.mean,
                    "prob_ci_lower": outcome.probability.ci_lower,
                    "prob_ci_upper": outcome.probability.ci_upper,
                    "robustness_score": analysis.robustness_score,
                    "max_regret": analysis.regret_analysis.get("max_regret", 0)
                })
        
        df = pd.DataFrame(df_rows)
        df.to_csv(f"{filepath}_robust_analysis.csv", index=False)
        
        print(f"Robust analysis exported to {filepath}_robust_analysis.json and .csv")


def create_robust_model() -> RobustGameTheoryModel:
    """Create default robust game theory model"""
    return RobustGameTheoryModel(mcmc_samples=2000)


if __name__ == "__main__":
    # Example usage
    model = create_robust_model()
    
    print("Creating robust game theory model...")
    
    # Example game state
    game_state = GameState(
        regime_cohesion=0.4,
        economic_stress=0.9,
        proxy_support=0.2,
        oil_price=97.0,
        external_support=0.3,
        nuclear_progress=0.7
    )
    
    print("Analyzing strategies robustly...")
    
    # Analyze all strategies
    all_strategies = list(Strategy)
    strategy_analyses = model.compare_strategies_robust(all_strategies, game_state)
    
    print("\nRobust Strategy Analysis Results:")
    print("=" * 50)
    
    # Sort by robustness score
    sorted_strategies = sorted(
        strategy_analyses.items(),
        key=lambda x: x[1].robustness_score,
        reverse=True
    )
    
    for strategy, analysis in sorted_strategies:
        print(f"\n{strategy.value}:")
        print(f"  Expected Utility: {analysis.expected_utility.mean:.3f} "
              f"({analysis.expected_utility.ci_lower:.3f}, {analysis.expected_utility.ci_upper:.3f})")
        print(f"  Robustness Score: {analysis.robustness_score:.3f}")
        print(f"  Max Regret: {analysis.regret_analysis['max_regret']:.3f}")
        
        print("  Outcome Probabilities:")
        for outcome in analysis.outcomes:
            print(f"    {outcome.outcome.value}: {outcome.probability.mean:.3f} "
                  f"({outcome.probability.ci_lower:.3f}, {outcome.probability.ci_upper:.3f})")
    
    print(f"\nMost Robust Strategy: {sorted_strategies[0][0].value}")