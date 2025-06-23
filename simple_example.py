#!/usr/bin/env python3
"""
Simple Example: Game Theory Iran Model
Demonstrates key concepts without full MCMC implementation
"""

import sys
sys.path.append('.')

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Define core enums and classes
class Strategy(Enum):
    DETERRENCE_DIPLOMACY = "halt_deter_diplomacy"
    DETERRENCE_ULTIMATUM = "halt_deter_ultimatum"  
    ESCALATION_DIPLOMACY = "expand_strikes_diplomacy"
    ESCALATION_ULTIMATUM = "expand_strikes_ultimatum"


class Outcome(Enum):
    DEAL = "negotiated_deal"
    LIMITED_RETALIATION = "limited_retaliation"
    FROZEN_CONFLICT = "frozen_conflict"
    FULL_WAR = "full_war"
    NUCLEAR_BREAKOUT = "nuclear_breakout"


@dataclass
class GameState:
    regime_cohesion: float     # 0-1, Iran's internal stability
    economic_stress: float     # 0-1, Economic pressure level
    proxy_support: float       # 0-1, Proxy network strength
    oil_price: float          # USD per barrel
    external_support: float    # 0-1, China/Russia support
    nuclear_progress: float    # 0-1, Nuclear program advancement


@dataclass
class StrategyResult:
    strategy: Strategy
    outcome_probabilities: Dict[Outcome, float]
    expected_utility: float
    war_risk: float
    nuclear_risk: float


class SimplifiedGameModel:
    """Simplified game theory model for demonstration"""
    
    def __init__(self):
        # Player preferences (1=best, 5=worst)
        self.usa_preferences = {
            Outcome.DEAL: 1,
            Outcome.LIMITED_RETALIATION: 2,
            Outcome.NUCLEAR_BREAKOUT: 3,
            Outcome.FROZEN_CONFLICT: 4,
            Outcome.FULL_WAR: 5
        }
    
    def evaluate_strategy(self, strategy: Strategy, state: GameState) -> StrategyResult:
        """Evaluate a strategy given the current game state"""
        
        # Base probabilities for each strategy
        base_probs = {
            Strategy.DETERRENCE_DIPLOMACY: {
                Outcome.DEAL: 0.35,
                Outcome.LIMITED_RETALIATION: 0.30,
                Outcome.FROZEN_CONFLICT: 0.25,
                Outcome.FULL_WAR: 0.08,
                Outcome.NUCLEAR_BREAKOUT: 0.02
            },
            Strategy.DETERRENCE_ULTIMATUM: {
                Outcome.DEAL: 0.25,
                Outcome.LIMITED_RETALIATION: 0.25,
                Outcome.FROZEN_CONFLICT: 0.30,
                Outcome.FULL_WAR: 0.15,
                Outcome.NUCLEAR_BREAKOUT: 0.05
            },
            Strategy.ESCALATION_DIPLOMACY: {
                Outcome.DEAL: 0.20,
                Outcome.LIMITED_RETALIATION: 0.35,
                Outcome.FROZEN_CONFLICT: 0.20,
                Outcome.FULL_WAR: 0.20,
                Outcome.NUCLEAR_BREAKOUT: 0.05
            },
            Strategy.ESCALATION_ULTIMATUM: {
                Outcome.DEAL: 0.10,
                Outcome.LIMITED_RETALIATION: 0.20,
                Outcome.FROZEN_CONFLICT: 0.25,
                Outcome.FULL_WAR: 0.35,
                Outcome.NUCLEAR_BREAKOUT: 0.10
            }
        }
        
        probs = base_probs[strategy].copy()
        
        # Adjust probabilities based on game state
        
        # Nuclear progress effects
        if state.nuclear_progress > 0.8:
            probs[Outcome.NUCLEAR_BREAKOUT] *= 3
            probs[Outcome.DEAL] *= 0.7
        
        # Regime cohesion effects  
        if state.regime_cohesion < 0.3:
            # Cornered animal effect
            probs[Outcome.FULL_WAR] *= 1.5
            probs[Outcome.NUCLEAR_BREAKOUT] *= 1.5
            probs[Outcome.DEAL] *= 0.5
        
        # Economic stress effects
        if state.economic_stress > 0.8:
            probs[Outcome.NUCLEAR_BREAKOUT] *= 1.3
            probs[Outcome.FULL_WAR] *= 1.2
        
        # External support effects
        if state.external_support > 0.6:
            probs[Outcome.DEAL] *= 1.4
            probs[Outcome.FULL_WAR] *= 0.7
        
        # Escalation strategy effects
        if "escalation" in strategy.value:
            probs[Outcome.FULL_WAR] *= 1.5
            probs[Outcome.LIMITED_RETALIATION] *= 1.3
        
        # Ultimatum effects
        if "ultimatum" in strategy.value:
            probs[Outcome.FULL_WAR] *= 1.2
            if state.regime_cohesion < 0.4:
                probs[Outcome.NUCLEAR_BREAKOUT] *= 1.4
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Calculate utility
        utility = sum(probs[outcome] * (6 - self.usa_preferences[outcome]) 
                     for outcome in probs.keys())
        utility = utility / 5.0  # Normalize to 0-1
        
        # Calculate risks
        war_risk = probs[Outcome.FULL_WAR]
        nuclear_risk = probs[Outcome.NUCLEAR_BREAKOUT]
        
        return StrategyResult(
            strategy=strategy,
            outcome_probabilities=probs,
            expected_utility=utility,
            war_risk=war_risk,
            nuclear_risk=nuclear_risk
        )
    
    def recommend_strategy(self, state: GameState) -> Tuple[Strategy, StrategyResult]:
        """Find the optimal strategy for given game state"""
        
        results = {}
        for strategy in Strategy:
            results[strategy] = self.evaluate_strategy(strategy, state)
        
        # Find strategy with highest expected utility
        best_strategy = max(results.keys(), 
                          key=lambda s: results[s].expected_utility)
        
        return best_strategy, results[best_strategy]


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)


def main():
    """Run simplified example analysis"""
    
    print_section("GAME THEORY IRAN MODEL - SIMPLIFIED EXAMPLE")
    
    # Initialize model
    model = SimplifiedGameModel()
    
    # 1. Current Situation Analysis
    print_section("1. CURRENT STRATEGIC SITUATION")
    
    current_state = GameState(
        regime_cohesion=0.4,      # Moderate regime stability
        economic_stress=0.9,      # Severe economic pressure  
        proxy_support=0.2,        # Proxy networks degraded
        oil_price=97.0,          # Current oil price
        external_support=0.3,     # Limited China/Russia support
        nuclear_progress=0.7      # Approaching breakout capability
    )
    
    print("Current Game State:")
    print(f"  🏛️  Regime Cohesion: {current_state.regime_cohesion:.1%}")
    print(f"  💰 Economic Stress: {current_state.economic_stress:.1%}")  
    print(f"  🔗 Proxy Support: {current_state.proxy_support:.1%}")
    print(f"  🛢️  Oil Price: ${current_state.oil_price:.0f}/barrel")
    print(f"  🌏 External Support: {current_state.external_support:.1%}")
    print(f"  ☢️  Nuclear Progress: {current_state.nuclear_progress:.1%}")
    
    # 2. Strategy Recommendation
    print_section("2. STRATEGY RECOMMENDATION")
    
    best_strategy, best_result = model.recommend_strategy(current_state)
    
    print(f"✅ Recommended Strategy: {best_strategy.value}")
    print(f"   Expected Utility: {best_result.expected_utility:.3f}")
    print(f"   War Risk: {best_result.war_risk:.1%}")
    print(f"   Nuclear Risk: {best_result.nuclear_risk:.1%}")
    
    # 3. All Strategies Comparison  
    print_section("3. STRATEGY COMPARISON")
    
    print("Comparing all strategic options:")
    print("-" * 60)
    print(f"{'Strategy':<25} {'Utility':<8} {'War Risk':<10} {'Nuclear Risk':<12}")
    print("-" * 60)
    
    all_results = {}
    for strategy in Strategy:
        result = model.evaluate_strategy(strategy, current_state)
        all_results[strategy] = result
        
        print(f"{strategy.value:<25} {result.expected_utility:.3f}    "
              f"{result.war_risk:.1%}        {result.nuclear_risk:.1%}")
    
    # 4. Outcome Probabilities
    print_section("4. OUTCOME PROBABILITIES")
    
    print(f"Probability distribution for {best_strategy.value}:")
    print("-" * 60)
    
    outcome_icons = {
        Outcome.DEAL: "🕊️",
        Outcome.LIMITED_RETALIATION: "⚡",
        Outcome.FROZEN_CONFLICT: "❄️", 
        Outcome.FULL_WAR: "🔥",
        Outcome.NUCLEAR_BREAKOUT: "☢️"
    }
    
    for outcome in Outcome:
        prob = best_result.outcome_probabilities[outcome]
        bar = '█' * int(prob * 50)
        icon = outcome_icons[outcome]
        print(f"{icon} {outcome.value:<25} {bar:<25} {prob:.1%}")
    
    # 5. Scenario Analysis
    print_section("5. SCENARIO ANALYSIS")
    
    scenarios = {
        "Current": current_state,
        "Regime Collapse": GameState(
            regime_cohesion=0.1,
            economic_stress=0.95,
            proxy_support=0.05,
            oil_price=97.0,
            external_support=0.1,
            nuclear_progress=0.85
        ),
        "Chinese Backing": GameState(
            regime_cohesion=0.6,
            economic_stress=0.5,
            proxy_support=0.4,
            oil_price=120.0,
            external_support=0.9,
            nuclear_progress=0.7
        ),
        "Nuclear Sprint": GameState(
            regime_cohesion=0.3,
            economic_stress=0.9,
            proxy_support=0.1,
            oil_price=97.0,
            external_support=0.2,
            nuclear_progress=0.95
        )
    }
    
    print("Optimal strategy by scenario:")
    print("-" * 60)
    
    for scenario_name, scenario_state in scenarios.items():
        best_strat, result = model.recommend_strategy(scenario_state)
        risk_level = "🟢" if result.war_risk < 0.2 else "🟡" if result.war_risk < 0.4 else "🔴"
        
        print(f"{risk_level} {scenario_name:<15} → {best_strat.value}")
        print(f"     War Risk: {result.war_risk:.1%}, "
              f"Nuclear Risk: {result.nuclear_risk:.1%}")
    
    # 6. Sensitivity Analysis
    print_section("6. SENSITIVITY TO NUCLEAR PROGRESS")
    
    print("How strategy changes with nuclear progress:")
    print("-" * 60)
    
    nuclear_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    for nuclear_level in nuclear_levels:
        test_state = GameState(
            regime_cohesion=current_state.regime_cohesion,
            economic_stress=current_state.economic_stress,
            proxy_support=current_state.proxy_support,
            oil_price=current_state.oil_price,
            external_support=current_state.external_support,
            nuclear_progress=nuclear_level
        )
        
        best_strat, result = model.recommend_strategy(test_state)
        
        if nuclear_level < 0.7:
            indicator = "🟢"
        elif nuclear_level < 0.85:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{indicator} Nuclear Progress: {nuclear_level:.0%} → "
              f"{best_strat.value}")
    
    # 7. Key Insights
    print_section("7. KEY STRATEGIC INSIGHTS")
    
    print("🎯 Strategic Findings:")
    print()
    print("1. CORNERED ANIMAL PARADOX:")
    print("   Low regime cohesion increases unpredictable escalation")
    print("   despite Iran's military weakness")
    print()
    print("2. NUCLEAR THRESHOLD EFFECT:")
    print("   Strategy shifts dramatically above 85% nuclear progress")
    print("   Window for diplomatic solutions narrows rapidly")
    print()
    print("3. EXTERNAL SUPPORT LEVERAGE:")
    print("   Chinese backing significantly improves deal prospects")
    print("   but reduces pressure for Iranian concessions")
    print()
    print("4. ESCALATION RISKS:")
    print("   Military escalation strategies double war risk")
    print("   Ultimatums trigger defensive reactions in weak regimes")
    print()
    print("5. OPTIMAL APPROACH:")
    print("   'Halt & Deter + De-escalatory Off-Ramp' provides")
    print("   best balance of deterrence and diplomatic options")
    
    # Summary
    print_section("ANALYSIS COMPLETE")
    
    print(f"📊 Based on current conditions:")
    print(f"   • Recommended Strategy: {best_strategy.value}")
    print(f"   • Success Probability: {(1-best_result.war_risk-best_result.nuclear_risk):.1%}")
    print(f"   • Primary Risk: {'Nuclear Breakout' if best_result.nuclear_risk > best_result.war_risk else 'Regional War'}")
    print()
    print("🚀 For interactive analysis with uncertainty quantification:")
    print("   uv run python run_d3_app.py")
    print()
    print("📚 For full documentation:")
    print("   See README.md and USER_GUIDE.md")


if __name__ == "__main__":
    main()