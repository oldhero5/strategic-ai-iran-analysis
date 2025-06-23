"""
Player classes for the Iran-Israel-US game theory model.
Based on the research analysis of strategic preferences and constraints.
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class Outcome(Enum):
    """Possible short-term outcomes from the strategic game."""
    DEAL = "Iranian Capitulation & Verifiable Deal"
    LIMITED_RETALIATION = "Limited Iranian Retaliation & De-escalation"  
    FROZEN_CONFLICT = "Protracted Low-Intensity Conflict"
    FULL_WAR = "Full-Scale Regional War"
    NUCLEAR_BREAKOUT = "Iranian Nuclear Breakout"


class MilitaryPosture(Enum):
    """US Military posture options."""
    HALT_DETER = "Halt & Deter"
    EXPAND_STRIKES = "Expand Strikes"


class DiplomaticPosture(Enum):
    """US Diplomatic posture options."""
    DEESCALATORY_OFFRAMP = "De-escalatory Off-Ramp"
    COERCIVE_ULTIMATUM = "Coercive Ultimatum"


@dataclass
class GameVariables:
    """Key observable variables that affect the game dynamics."""
    regime_cohesion: float = 0.4  # Iranian regime internal unity (0-1)
    economic_stress: float = 0.9  # Iranian economic distress level (0-1)
    proxy_support: float = 0.1    # Support from proxy network (0-1)
    oil_price: float = 97.0       # Brent crude oil price (USD/barrel)
    external_support: float = 0.2 # China/Russia support level (0-1)
    nuclear_progress: float = 0.7 # Iranian nuclear program advancement (0-1)


class Player:
    """Base class for strategic players in the game."""
    
    def __init__(self, name: str, preferences: Dict[Outcome, int]):
        self.name = name
        self.preferences = preferences  # Ordinal ranking: 1=best, 5=worst
    
    def get_utility(self, outcome: Outcome) -> int:
        """Get utility ranking for a given outcome (lower is better)."""
        return self.preferences[outcome]
    
    def rank_outcomes(self) -> List[Outcome]:
        """Return outcomes ranked by preference (best to worst)."""
        return sorted(self.preferences.keys(), key=lambda x: self.preferences[x])


class USA(Player):
    """United States player with Trump Administration preferences."""
    
    def __init__(self):
        # From research: A > B > E > C > D (1=best, 5=worst)
        preferences = {
            Outcome.DEAL: 1,                    # Achieves core objectives without war costs
            Outcome.LIMITED_RETALIATION: 2,     # Negotiated de-escalation
            Outcome.NUCLEAR_BREAKOUT: 3,        # Policy failure but manageable
            Outcome.FROZEN_CONFLICT: 4,         # Grinding, costly stalemate
            Outcome.FULL_WAR: 5                 # Catastrophic costs and risks
        }
        super().__init__("USA", preferences)
        
        self.objectives = [
            "Prevent Iranian nuclear weapon",
            "Maintain regional stability", 
            "Avoid costly long-term war",
            "Reassert Middle East leadership"
        ]
        
        self.constraints = [
            "Domestic war fatigue",
            "Alliance management with Israel",
            "Economic/energy market concerns",
            "Great power competition focus"
        ]


class Iran(Player):
    """Islamic Republic of Iran player focused on regime survival."""
    
    def __init__(self):
        # From research: E > B > A > C > D (1=best, 5=worst)
        preferences = {
            Outcome.NUCLEAR_BREAKOUT: 1,        # Ultimate security guarantee
            Outcome.LIMITED_RETALIATION: 2,     # Face-saving survival path
            Outcome.DEAL: 3,                    # Humiliating but survivable
            Outcome.FROZEN_CONFLICT: 4,         # Slow degradation
            Outcome.FULL_WAR: 5                 # Regime destruction
        }
        super().__init__("Iran", preferences)
        
        self.objectives = [
            "Regime survival (paramount)",
            "Preserve nuclear program",
            "Maintain deterrence credibility",
            "Avoid national humiliation"
        ]
        
        self.constraints = [
            "Economic collapse under sanctions",
            "Military inferiority to US/Israel",
            "Degraded proxy network",
            "Internal legitimacy crisis"
        ]


class Israel(Player):
    """State of Israel player with Netanyahu government preferences."""
    
    def __init__(self):
        # From research: E > A > C > B > D (1=best, 5=worst)
        # Note: E is preferred by hawks as ultimate justification for total war
        preferences = {
            Outcome.NUCLEAR_BREAKOUT: 1,        # Justifies total war with US support
            Outcome.DEAL: 2,                    # Eliminates existential threat
            Outcome.FROZEN_CONFLICT: 3,         # Can continue degrading Iran
            Outcome.LIMITED_RETALIATION: 4,     # Leaves threats intact
            Outcome.FULL_WAR: 5                 # Unsustainable costs alone
        }
        super().__init__("Israel", preferences)
        
        self.objectives = [
            "Eliminate Iranian existential threat",
            "Achieve regional security",
            "Potential regime change in Tehran"
        ]
        
        self.constraints = [
            "Cannot destroy hardened targets alone",
            "Dependent on US support",
            "Risk of military overstretch",
            "Severe economic costs in prolonged war"
        ]


class GameTheoryModel:
    """Main game theory model coordinating all players and strategies."""
    
    def __init__(self, variables: GameVariables = None):
        self.usa = USA()
        self.iran = Iran()
        self.israel = Israel()
        self.variables = variables or GameVariables()
        
        # US Strategic combinations
        self.strategies = {
            "deterrence_diplomacy": (MilitaryPosture.HALT_DETER, DiplomaticPosture.DEESCALATORY_OFFRAMP),
            "deterrence_ultimatum": (MilitaryPosture.HALT_DETER, DiplomaticPosture.COERCIVE_ULTIMATUM),
            "escalation_diplomacy": (MilitaryPosture.EXPAND_STRIKES, DiplomaticPosture.DEESCALATORY_OFFRAMP),
            "escalation_ultimatum": (MilitaryPosture.EXPAND_STRIKES, DiplomaticPosture.COERCIVE_ULTIMATUM)
        }
    
    def get_outcome_probabilities(self, strategy_key: str) -> Dict[Outcome, float]:
        """Calculate outcome probabilities for a given US strategy."""
        military, diplomatic = self.strategies[strategy_key]
        
        # Base probabilities influenced by strategy choice and game variables
        if strategy_key == "deterrence_diplomacy":
            # Optimal strategy from research - highest chance of good outcomes
            base_probs = {
                Outcome.DEAL: 0.4,
                Outcome.LIMITED_RETALIATION: 0.35,
                Outcome.FROZEN_CONFLICT: 0.15,
                Outcome.FULL_WAR: 0.05,
                Outcome.NUCLEAR_BREAKOUT: 0.05
            }
        elif strategy_key == "deterrence_ultimatum":
            # Moderate risk - no off-ramp increases conflict probability
            base_probs = {
                Outcome.DEAL: 0.2,
                Outcome.LIMITED_RETALIATION: 0.25,
                Outcome.FROZEN_CONFLICT: 0.35,
                Outcome.FULL_WAR: 0.15,
                Outcome.NUCLEAR_BREAKOUT: 0.05
            }
        elif strategy_key == "escalation_diplomacy":
            # Mixed signals increase miscalculation risk
            base_probs = {
                Outcome.DEAL: 0.15,
                Outcome.LIMITED_RETALIATION: 0.2,
                Outcome.FROZEN_CONFLICT: 0.25,
                Outcome.FULL_WAR: 0.3,
                Outcome.NUCLEAR_BREAKOUT: 0.1
            }
        else:  # escalation_ultimatum
            # Highest risk strategy - likely triggers desperation
            base_probs = {
                Outcome.DEAL: 0.1,
                Outcome.LIMITED_RETALIATION: 0.1,
                Outcome.FROZEN_CONFLICT: 0.2,
                Outcome.FULL_WAR: 0.45,
                Outcome.NUCLEAR_BREAKOUT: 0.15
            }
        
        # Adjust probabilities based on game variables
        adjusted_probs = self._adjust_probabilities_for_variables(base_probs)
        
        return adjusted_probs
    
    def _adjust_probabilities_for_variables(self, base_probs: Dict[Outcome, float]) -> Dict[Outcome, float]:
        """Adjust outcome probabilities based on current game state variables."""
        adjusted = base_probs.copy()
        
        # Higher economic stress increases desperation (nuclear breakout, full war)
        stress_factor = self.variables.economic_stress
        adjusted[Outcome.NUCLEAR_BREAKOUT] *= (1 + stress_factor * 0.5)
        adjusted[Outcome.FULL_WAR] *= (1 + stress_factor * 0.3)
        
        # Lower regime cohesion increases unpredictability
        cohesion_factor = 1 - self.variables.regime_cohesion
        adjusted[Outcome.NUCLEAR_BREAKOUT] *= (1 + cohesion_factor * 0.4)
        adjusted[Outcome.FULL_WAR] *= (1 + cohesion_factor * 0.2)
        
        # Higher oil prices increase economic pressure for all parties
        if self.variables.oil_price > 100:
            oil_factor = (self.variables.oil_price - 100) / 100
            adjusted[Outcome.DEAL] *= (1 + oil_factor * 0.3)  # Pressure for resolution
            adjusted[Outcome.FROZEN_CONFLICT] *= (1 - oil_factor * 0.2)  # Less sustainable
        
        # Lower proxy support makes Iran more vulnerable and desperate
        proxy_weakness = 1 - self.variables.proxy_support
        adjusted[Outcome.NUCLEAR_BREAKOUT] *= (1 + proxy_weakness * 0.3)
        adjusted[Outcome.DEAL] *= (1 + proxy_weakness * 0.2)  # More likely to negotiate
        
        # Normalize probabilities to sum to 1.0
        total = sum(adjusted.values())
        for outcome in adjusted:
            adjusted[outcome] /= total
            
        return adjusted
    
    def get_expected_utilities(self, strategy_key: str) -> Dict[str, float]:
        """Calculate expected utility for each player given a US strategy."""
        probs = self.get_outcome_probabilities(strategy_key)
        
        utilities = {}
        for player_name, player in [("USA", self.usa), ("Iran", self.iran), ("Israel", self.israel)]:
            expected_utility = sum(
                probs[outcome] * (6 - player.get_utility(outcome))  # Convert to positive scale
                for outcome in probs
            )
            utilities[player_name] = expected_utility
            
        return utilities