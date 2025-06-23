"""
Utility functions for game theory calculations and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from backend.models.players import GameTheoryModel, GameVariables, Outcome


def run_monte_carlo_simulation(
    model: GameTheoryModel, 
    strategy: str, 
    n_simulations: int = 1000,
    variable_uncertainty: float = 0.1
) -> Dict:
    """
    Run Monte Carlo simulation varying game variables within uncertainty bounds.
    
    Args:
        model: GameTheoryModel instance
        strategy: US strategy key
        n_simulations: Number of simulation runs
        variable_uncertainty: Uncertainty range for variables (0-1)
    
    Returns:
        Dictionary with simulation results
    """
    results = {
        'outcomes': [],
        'utilities': {'USA': [], 'Iran': [], 'Israel': []},
        'variables': []
    }
    
    # Store original variables
    original_vars = model.variables
    
    for _ in range(n_simulations):
        # Add random variation to variables
        varied_vars = GameVariables(
            regime_cohesion=np.clip(
                original_vars.regime_cohesion + np.random.normal(0, variable_uncertainty), 
                0, 1
            ),
            economic_stress=np.clip(
                original_vars.economic_stress + np.random.normal(0, variable_uncertainty), 
                0, 1
            ),
            proxy_support=np.clip(
                original_vars.proxy_support + np.random.normal(0, variable_uncertainty), 
                0, 1
            ),
            oil_price=max(
                original_vars.oil_price + np.random.normal(0, original_vars.oil_price * variable_uncertainty),
                30  # Minimum oil price
            ),
            external_support=np.clip(
                original_vars.external_support + np.random.normal(0, variable_uncertainty), 
                0, 1
            ),
            nuclear_progress=np.clip(
                original_vars.nuclear_progress + np.random.normal(0, variable_uncertainty), 
                0, 1
            )
        )
        
        # Update model with varied variables
        model.variables = varied_vars
        
        # Calculate probabilities and sample outcome
        probs = model.get_outcome_probabilities(strategy)
        outcome = np.random.choice(list(probs.keys()), p=list(probs.values()))
        
        # Calculate utilities
        utilities = model.get_expected_utilities(strategy)
        
        # Store results
        results['outcomes'].append(outcome)
        for player in utilities:
            results['utilities'][player].append(utilities[player])
        results['variables'].append(varied_vars)
    
    # Restore original variables
    model.variables = original_vars
    
    # Calculate summary statistics
    outcome_counts = pd.Series(results['outcomes']).value_counts(normalize=True).to_dict()
    
    results['summary'] = {
        'outcome_probabilities': outcome_counts,
        'mean_utilities': {player: np.mean(utils) for player, utils in results['utilities'].items()},
        'std_utilities': {player: np.std(utils) for player, utils in results['utilities'].items()}
    }
    
    return results


def calculate_strategy_rankings(model: GameTheoryModel) -> pd.DataFrame:
    """
    Calculate and rank all US strategies by expected outcomes.
    
    Returns:
        DataFrame with strategy rankings and metrics
    """
    strategy_data = []
    
    for strategy_name, strategy_params in model.strategies.items():
        # Get outcome probabilities
        probs = model.get_outcome_probabilities(strategy_name)
        
        # Get expected utilities
        utilities = model.get_expected_utilities(strategy_name)
        
        # Calculate risk metrics
        war_risk = probs[Outcome.FULL_WAR] + probs[Outcome.NUCLEAR_BREAKOUT]
        success_prob = probs[Outcome.DEAL] + probs[Outcome.LIMITED_RETALIATION]
        
        # Calculate entropy (uncertainty measure)
        prob_values = list(probs.values())
        uncertainty = entropy(prob_values, base=2)
        
        strategy_data.append({
            'strategy': strategy_name,
            'military_posture': strategy_params[0].value,
            'diplomatic_posture': strategy_params[1].value,
            'usa_utility': utilities['USA'],
            'iran_utility': utilities['Iran'],
            'israel_utility': utilities['Israel'],
            'success_probability': success_prob,
            'war_risk': war_risk,
            'uncertainty': uncertainty,
            'deal_prob': probs[Outcome.DEAL],
            'limited_retaliation_prob': probs[Outcome.LIMITED_RETALIATION],
            'frozen_conflict_prob': probs[Outcome.FROZEN_CONFLICT],
            'full_war_prob': probs[Outcome.FULL_WAR],
            'nuclear_breakout_prob': probs[Outcome.NUCLEAR_BREAKOUT]
        })
    
    df = pd.DataFrame(strategy_data)
    
    # Rank strategies by US utility (primary) and war risk (secondary)
    df['usa_rank'] = df['usa_utility'].rank(ascending=False)
    df['risk_rank'] = df['war_risk'].rank(ascending=True)  # Lower risk is better
    df['overall_score'] = df['usa_rank'] + df['risk_rank']
    df = df.sort_values('overall_score')
    
    return df


def sensitivity_analysis(
    model: GameTheoryModel, 
    strategy: str, 
    variable_name: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    steps: int = 20
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on a specific variable.
    
    Args:
        model: GameTheoryModel instance
        strategy: US strategy to analyze
        variable_name: Name of variable to vary
        min_val: Minimum value for the variable
        max_val: Maximum value for the variable
        steps: Number of steps between min and max
    
    Returns:
        DataFrame with sensitivity analysis results
    """
    # Store original variable value
    original_value = getattr(model.variables, variable_name)
    
    # Create range of values to test
    test_values = np.linspace(min_val, max_val, steps)
    
    results = []
    
    for value in test_values:
        # Update the specific variable
        setattr(model.variables, variable_name, value)
        
        # Calculate probabilities and utilities
        probs = model.get_outcome_probabilities(strategy)
        utilities = model.get_expected_utilities(strategy)
        
        # Store results
        result = {
            variable_name: value,
            'usa_utility': utilities['USA'],
            'iran_utility': utilities['Iran'],
            'israel_utility': utilities['Israel'],
            'war_risk': probs[Outcome.FULL_WAR] + probs[Outcome.NUCLEAR_BREAKOUT],
            'success_prob': probs[Outcome.DEAL] + probs[Outcome.LIMITED_RETALIATION]
        }
        
        # Add individual outcome probabilities
        for outcome in probs:
            result[f'{outcome.name.lower()}_prob'] = probs[outcome]
        
        results.append(result)
    
    # Restore original value
    setattr(model.variables, variable_name, original_value)
    
    return pd.DataFrame(results)


def calculate_escalation_ladder_position(variables: GameVariables) -> Dict[str, float]:
    """
    Calculate current position on escalation ladder based on game variables.
    
    Returns:
        Dictionary with escalation metrics
    """
    # Defcon-style scale (5=normal, 1=nuclear)
    base_defcon = 5.0
    
    # Factors that increase escalation
    escalation_factors = {
        'economic_stress': variables.economic_stress * 0.8,
        'regime_weakness': (1 - variables.regime_cohesion) * 0.6,
        'proxy_collapse': (1 - variables.proxy_support) * 0.4,
        'nuclear_progress': variables.nuclear_progress * 0.7,
        'oil_crisis': max(0, (variables.oil_price - 90) / 50) * 0.5
    }
    
    total_escalation = sum(escalation_factors.values())
    current_defcon = max(1.0, base_defcon - total_escalation)
    
    return {
        'defcon_level': current_defcon,
        'escalation_factors': escalation_factors,
        'total_escalation': total_escalation,
        'crisis_severity': min(1.0, total_escalation / 2.0)  # Normalized 0-1
    }


def calculate_market_impact_metrics(variables: GameVariables) -> Dict[str, float]:
    """
    Calculate market impact metrics based on game variables.
    
    Returns:
        Dictionary with market indicators
    """
    # VIX calculation (simplified)
    base_vix = 20
    volatility_factors = [
        variables.economic_stress * 25,
        (1 - variables.regime_cohesion) * 20,
        (variables.oil_price - 80) / 4,  # Oil price impact
        variables.nuclear_progress * 15
    ]
    estimated_vix = base_vix + sum(volatility_factors)
    
    # Gold price impact (simplified)
    base_gold = 2000
    safe_haven_demand = (variables.economic_stress + (1 - variables.regime_cohesion)) * 400
    estimated_gold = base_gold + safe_haven_demand
    
    # Iranian Rial impact
    base_rial_rate = 42000  # Historical normal
    sanctions_multiplier = 1 + (variables.economic_stress * 12)
    estimated_rial = base_rial_rate * sanctions_multiplier
    
    return {
        'vix_estimate': min(100, estimated_vix),
        'gold_price_estimate': estimated_gold,
        'rial_rate_estimate': estimated_rial,
        'oil_price': variables.oil_price,
        'market_stress_index': min(1.0, estimated_vix / 100)
    }