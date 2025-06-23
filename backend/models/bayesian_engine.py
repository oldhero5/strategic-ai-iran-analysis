"""
Bayesian Inference Engine for Game Theory Model
Integrates MCMC sampling with existing strategic analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .mcmc_model import BayesianGameModel, Strategy, Outcome, GameState, MCMCResults
from .players import USA, Iran, Israel


@dataclass
class BayesianUpdate:
    """Structure for Bayesian belief updates"""
    timestamp: datetime
    prior_belief: Dict[str, float]
    evidence: Dict[str, Any]
    posterior_belief: Dict[str, float]
    kl_divergence: float  # Measure of belief change
    

@dataclass
class StrategyRecommendation:
    """AI-generated strategy recommendation with uncertainty"""
    recommended_strategy: Strategy
    expected_utility: float
    utility_confidence_interval: Tuple[float, float]
    risk_assessment: Dict[str, float]
    alternative_strategies: List[Tuple[Strategy, float]]
    reasoning: str
    certainty_level: float


class BayesianInferenceEngine:
    """
    Advanced Bayesian inference engine for strategic game theory.
    
    Features:
    - Real-time belief updating based on new evidence
    - Uncertainty quantification for all predictions
    - Strategy optimization under uncertainty
    - Counterfactual analysis
    - Robust decision making under model uncertainty
    """
    
    def __init__(self, mcmc_model: Optional[BayesianGameModel] = None):
        self.mcmc_model = mcmc_model or BayesianGameModel()
        self.belief_history: List[BayesianUpdate] = []
        self.current_posterior = None
        self.strategy_cache = {}
        
        # Initialize with expert priors
        self.expert_priors = self._load_expert_priors()
        
    def _load_expert_priors(self) -> Dict[str, Dict[str, float]]:
        """Load expert prior beliefs from research"""
        return {
            "iran_regime_stability": {
                "low": 0.3,      # 30% chance regime is unstable
                "medium": 0.5,   # 50% chance moderately stable  
                "high": 0.2      # 20% chance highly stable
            },
            "nuclear_timeline": {
                "months_1_3": 0.1,   # 10% chance breakout in 1-3 months
                "months_3_6": 0.2,   # 20% chance in 3-6 months
                "months_6_12": 0.4,  # 40% chance in 6-12 months
                "years_1_plus": 0.3  # 30% chance 1+ years
            },
            "escalation_dynamics": {
                "rapid_escalation": 0.25,
                "gradual_escalation": 0.45,
                "controlled_escalation": 0.30
            }
        }
    
    def update_beliefs(self, 
                      evidence: Dict[str, Any], 
                      evidence_reliability: float = 0.8) -> BayesianUpdate:
        """
        Update beliefs based on new evidence using Bayes' theorem.
        
        Args:
            evidence: Dictionary of observed evidence
            evidence_reliability: Reliability score for the evidence (0-1)
            
        Returns:
            BayesianUpdate object with prior, posterior, and change metrics
        """
        
        # Store prior beliefs
        prior_belief = self._extract_current_beliefs()
        
        # Update model with evidence
        posterior_samples = self._incorporate_evidence(evidence, evidence_reliability)
        
        # Compute posterior beliefs
        posterior_belief = self._compute_posterior_beliefs(posterior_samples)
        
        # Measure belief change
        kl_div = self._compute_kl_divergence(prior_belief, posterior_belief)
        
        # Create update record
        update = BayesianUpdate(
            timestamp=datetime.now(),
            prior_belief=prior_belief,
            evidence=evidence,
            posterior_belief=posterior_belief,
            kl_divergence=kl_div
        )
        
        self.belief_history.append(update)
        self.current_posterior = posterior_samples
        
        return update
    
    def _extract_current_beliefs(self) -> Dict[str, float]:
        """Extract current beliefs from model state"""
        if self.current_posterior is None:
            # Use expert priors as initial beliefs
            return {
                "regime_stability": 0.4,
                "nuclear_breakout_6mo": 0.2,
                "war_probability": 0.15,
                "deal_probability": 0.35
            }
        else:
            # Extract from posterior samples
            return {
                "regime_stability": np.mean(self.current_posterior["regime_cohesion"]),
                "nuclear_breakout_6mo": np.mean(self.current_posterior["p_nuclear"]),
                "war_probability": np.mean(self.current_posterior["p_war_det_dip"]),
                "deal_probability": np.mean(self.current_posterior["p_deal_det_dip"])
            }
    
    def _incorporate_evidence(self, 
                            evidence: Dict[str, Any], 
                            reliability: float) -> Dict[str, np.ndarray]:
        """Incorporate new evidence into the model"""
        
        # Re-sample with evidence constraints
        # This is a simplified version - full implementation would modify priors
        
        if self.mcmc_model.trace is None:
            self.mcmc_model.build_model()
            self.mcmc_model.sample_posterior(draws=1000, tune=500)
        
        # Weight samples by evidence compatibility
        samples = {}
        for var_name in ["regime_cohesion", "p_nuclear", "p_war_det_dip", "p_deal_det_dip"]:
            if var_name in self.mcmc_model.trace.posterior.data_vars:
                raw_samples = self.mcmc_model.trace.posterior[var_name].values.flatten()
                
                # Apply evidence weighting (simplified)
                weights = np.ones(len(raw_samples))
                if "regime_stability_signal" in evidence:
                    signal = evidence["regime_stability_signal"]
                    if var_name == "regime_cohesion":
                        # Weight samples closer to observed signal
                        weights *= np.exp(-0.5 * ((raw_samples - signal) / 0.1) ** 2)
                
                # Resample with weights
                indices = np.random.choice(len(raw_samples), size=len(raw_samples), p=weights/weights.sum())
                samples[var_name] = raw_samples[indices]
        
        return samples
    
    def _compute_posterior_beliefs(self, samples: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute posterior beliefs from samples"""
        return {
            "regime_stability": np.mean(samples.get("regime_cohesion", [0.4])),
            "nuclear_breakout_6mo": np.mean(samples.get("p_nuclear", [0.2])),
            "war_probability": np.mean(samples.get("p_war_det_dip", [0.15])),
            "deal_probability": np.mean(samples.get("p_deal_det_dip", [0.35]))
        }
    
    def _compute_kl_divergence(self, prior: Dict[str, float], posterior: Dict[str, float]) -> float:
        """Compute KL divergence between prior and posterior beliefs"""
        kl_div = 0.0
        for key in prior.keys():
            if key in posterior:
                p = max(prior[key], 1e-10)  # Avoid log(0)
                q = max(posterior[key], 1e-10)
                kl_div += p * np.log(p / q)
        return kl_div
    
    def recommend_strategy(self, 
                          game_state: GameState,
                          usa_preferences: Optional[Dict[Outcome, float]] = None) -> StrategyRecommendation:
        """
        Generate optimal strategy recommendation with uncertainty quantification.
        
        Args:
            game_state: Current state of the strategic game
            usa_preferences: USA utility function (defaults to standard preferences)
            
        Returns:
            StrategyRecommendation with optimal strategy and uncertainty bounds
        """
        
        if usa_preferences is None:
            usa_preferences = {
                Outcome.DEAL: 1.0,
                Outcome.LIMITED_RETALIATION: 0.6,
                Outcome.FROZEN_CONFLICT: 0.4,
                Outcome.FULL_WAR: 0.0,
                Outcome.NUCLEAR_BREAKOUT: -0.5
            }
        
        # Sample strategies under uncertainty
        strategy_utilities = self._sample_strategy_utilities(game_state, usa_preferences)
        
        # Find optimal strategy
        optimal_strategy = max(strategy_utilities.keys(), 
                             key=lambda s: np.mean(strategy_utilities[s]))
        
        optimal_utility_samples = strategy_utilities[optimal_strategy]
        expected_utility = np.mean(optimal_utility_samples)
        utility_ci = (np.percentile(optimal_utility_samples, 2.5),
                     np.percentile(optimal_utility_samples, 97.5))
        
        # Risk assessment
        risk_assessment = self._assess_strategy_risks(optimal_strategy, game_state)
        
        # Alternative strategies
        alternatives = sorted(
            [(s, np.mean(utils)) for s, utils in strategy_utilities.items() if s != optimal_strategy],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(optimal_strategy, game_state, risk_assessment)
        
        # Compute certainty level
        certainty = self._compute_certainty(strategy_utilities)
        
        return StrategyRecommendation(
            recommended_strategy=optimal_strategy,
            expected_utility=expected_utility,
            utility_confidence_interval=utility_ci,
            risk_assessment=risk_assessment,
            alternative_strategies=alternatives,
            reasoning=reasoning,
            certainty_level=certainty
        )
    
    def _sample_strategy_utilities(self, 
                                  game_state: GameState, 
                                  preferences: Dict[Outcome, float]) -> Dict[Strategy, np.ndarray]:
        """Sample utility distributions for each strategy"""
        
        # Ensure we have posterior samples
        if self.mcmc_model.trace is None:
            self.mcmc_model.build_model()
            self.mcmc_model.sample_posterior(draws=1000)
        
        strategy_results = self.mcmc_model.analyze_strategies()
        
        # Convert to utility samples
        strategy_utilities = {}
        
        for strategy in Strategy:
            if strategy in strategy_results:
                utilities = []
                
                # Sample from uncertainty distributions
                for _ in range(1000):
                    total_utility = 0.0
                    for outcome in Outcome:
                        if outcome in strategy_results[strategy]:
                            mean_prob, lower_ci, upper_ci = strategy_results[strategy][outcome]
                            # Sample probability from beta distribution approximation
                            alpha, beta = self._fit_beta_from_ci(mean_prob, lower_ci, upper_ci)
                            prob_sample = np.random.beta(alpha, beta)
                            total_utility += prob_sample * preferences[outcome]
                    
                    utilities.append(total_utility)
                
                strategy_utilities[strategy] = np.array(utilities)
        
        return strategy_utilities
    
    def _fit_beta_from_ci(self, mean: float, lower: float, upper: float) -> Tuple[float, float]:
        """Fit beta distribution parameters from mean and confidence interval"""
        # Method of moments approximation
        if mean <= 0 or mean >= 1:
            return 1.0, 1.0
            
        variance = ((upper - lower) / 3.92) ** 2  # Approximate from 95% CI
        variance = min(variance, mean * (1 - mean) * 0.99)  # Ensure valid variance
        
        alpha = mean * ((mean * (1 - mean)) / variance - 1)
        beta = (1 - mean) * ((mean * (1 - mean)) / variance - 1)
        
        return max(alpha, 0.1), max(beta, 0.1)
    
    def _assess_strategy_risks(self, 
                              strategy: Strategy, 
                              game_state: GameState) -> Dict[str, float]:
        """Assess risks associated with a strategy"""
        
        base_risks = {
            "escalation_risk": 0.2,
            "miscalculation_risk": 0.15,
            "third_party_risk": 0.1,
            "domestic_political_risk": 0.05
        }
        
        # Adjust risks based on strategy and game state
        if strategy in [Strategy.ESCALATION_DIPLOMACY, Strategy.ESCALATION_ULTIMATUM]:
            base_risks["escalation_risk"] *= 2.0
            base_risks["miscalculation_risk"] *= 1.5
        
        if game_state.regime_cohesion < 0.3:
            base_risks["miscalculation_risk"] *= 1.5
            
        if game_state.nuclear_progress > 0.7:
            base_risks["escalation_risk"] *= 1.3
        
        return base_risks
    
    def _generate_reasoning(self, 
                           strategy: Strategy, 
                           game_state: GameState, 
                           risks: Dict[str, float]) -> str:
        """Generate human-readable reasoning for strategy choice"""
        
        reasoning_parts = []
        
        # Strategy justification
        if strategy == Strategy.DETERRENCE_DIPLOMACY:
            reasoning_parts.append("Halt & Deter + Diplomacy offers the best balance of containment and de-escalation.")
        elif strategy == Strategy.DETERRENCE_ULTIMATUM:
            reasoning_parts.append("Halt & Deter + Ultimatum provides strong deterrence with clear consequences.")
        elif strategy == Strategy.ESCALATION_DIPLOMACY:
            reasoning_parts.append("Expanded Strikes + Diplomacy shows resolve while keeping diplomatic channels open.")
        else:
            reasoning_parts.append("Expanded Strikes + Ultimatum applies maximum pressure for immediate compliance.")
        
        # Game state considerations
        if game_state.regime_cohesion < 0.4:
            reasoning_parts.append("Low regime cohesion increases unpredictability and escalation risk.")
        
        if game_state.nuclear_progress > 0.7:
            reasoning_parts.append("High nuclear progress creates urgency for decisive action.")
        
        if game_state.economic_stress > 0.8:
            reasoning_parts.append("Severe economic stress may force Iran toward desperate measures.")
        
        # Risk considerations
        max_risk = max(risks.values())
        if max_risk > 0.3:
            reasoning_parts.append("However, significant risks remain and should be carefully monitored.")
        
        return " ".join(reasoning_parts)
    
    def _compute_certainty(self, strategy_utilities: Dict[Strategy, np.ndarray]) -> float:
        """Compute certainty level in strategy recommendation"""
        
        if len(strategy_utilities) < 2:
            return 0.5
        
        # Compute overlap between top strategies
        utilities_list = sorted(strategy_utilities.values(), key=np.mean, reverse=True)
        
        if len(utilities_list) >= 2:
            top_util = utilities_list[0]
            second_util = utilities_list[1]
            
            # Probability that top strategy is actually better
            prob_better = np.mean(top_util > second_util)
            return prob_better
        
        return 0.8
    
    def counterfactual_analysis(self, 
                               base_game_state: GameState,
                               counterfactual_changes: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform counterfactual analysis: "What if X were different?"
        
        Args:
            base_game_state: Current game state
            counterfactual_changes: Dictionary of parameter changes to test
            
        Returns:
            Analysis results comparing base case to counterfactual
        """
        
        # Get base case recommendation
        base_recommendation = self.recommend_strategy(base_game_state)
        
        # Create counterfactual game state
        cf_state_dict = asdict(base_game_state)
        cf_state_dict.update(counterfactual_changes)
        cf_game_state = GameState(**cf_state_dict)
        
        # Get counterfactual recommendation
        cf_recommendation = self.recommend_strategy(cf_game_state)
        
        return {
            "base_case": {
                "strategy": base_recommendation.recommended_strategy,
                "expected_utility": base_recommendation.expected_utility,
                "certainty": base_recommendation.certainty_level
            },
            "counterfactual": {
                "strategy": cf_recommendation.recommended_strategy,
                "expected_utility": cf_recommendation.expected_utility,
                "certainty": cf_recommendation.certainty_level
            },
            "changes": counterfactual_changes,
            "utility_difference": cf_recommendation.expected_utility - base_recommendation.expected_utility,
            "strategy_changed": base_recommendation.recommended_strategy != cf_recommendation.recommended_strategy
        }
    
    def scenario_planning(self, 
                         game_states: List[GameState],
                         scenario_names: List[str]) -> Dict[str, StrategyRecommendation]:
        """
        Analyze multiple scenarios for robust strategy planning.
        
        Args:
            game_states: List of different game state scenarios
            scenario_names: Names for each scenario
            
        Returns:
            Dictionary mapping scenario names to strategy recommendations
        """
        
        results = {}
        
        for state, name in zip(game_states, scenario_names):
            recommendation = self.recommend_strategy(state)
            results[name] = recommendation
        
        return results
    
    def export_analysis(self, filepath: str) -> None:
        """Export complete Bayesian analysis to file"""
        
        analysis_data = {
            "belief_history": [asdict(update) for update in self.belief_history],
            "expert_priors": self.expert_priors,
            "model_diagnostics": self.mcmc_model.convergence_diagnostics() if self.mcmc_model.trace else {},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{filepath}_bayesian_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"Bayesian analysis exported to {filepath}_bayesian_analysis.json")


def create_default_engine() -> BayesianInferenceEngine:
    """Create default Bayesian inference engine"""
    mcmc_model = BayesianGameModel()
    return BayesianInferenceEngine(mcmc_model)


if __name__ == "__main__":
    # Example usage
    engine = create_default_engine()
    
    # Create example game state
    game_state = GameState(
        regime_cohesion=0.4,
        economic_stress=0.9,
        proxy_support=0.2,
        oil_price=97.0,
        external_support=0.3,
        nuclear_progress=0.7
    )
    
    print("Generating strategy recommendation...")
    recommendation = engine.recommend_strategy(game_state)
    
    print(f"Recommended Strategy: {recommendation.recommended_strategy.value}")
    print(f"Expected Utility: {recommendation.expected_utility:.3f}")
    print(f"Confidence Interval: ({recommendation.utility_confidence_interval[0]:.3f}, {recommendation.utility_confidence_interval[1]:.3f})")
    print(f"Certainty Level: {recommendation.certainty_level:.3f}")
    print(f"Reasoning: {recommendation.reasoning}")
    
    print("\nRisk Assessment:")
    for risk_type, risk_level in recommendation.risk_assessment.items():
        print(f"  {risk_type}: {risk_level:.3f}")