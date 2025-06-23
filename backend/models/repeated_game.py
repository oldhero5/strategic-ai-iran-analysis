"""
Repeated Game Theory Model for Iran Nuclear Crisis
Analyzes how strategies evolve over multiple rounds of interaction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict

from .mcmc_model import Strategy, Outcome, GameState
from .robust_gametheory import RobustGameTheoryModel, StrategyAnalysis


class GamePhase(Enum):
    """Phases of the repeated game"""
    INITIAL_POSTURING = "initial_posturing"
    NEGOTIATION = "negotiation"
    ESCALATION = "escalation"
    CRISIS = "crisis"
    RESOLUTION = "resolution"
    POST_CONFLICT = "post_conflict"


@dataclass
class GameHistory:
    """Track history of moves and outcomes"""
    round: int
    phase: GamePhase
    us_strategy: Strategy
    iran_strategy: Strategy
    israel_action: str
    outcome: Outcome
    game_state: GameState
    payoffs: Dict[str, float]


@dataclass
class ReputationState:
    """Track reputation and credibility"""
    us_credibility: float = 0.7
    iran_credibility: float = 0.5
    israel_restraint: float = 0.4
    international_support_us: float = 0.6
    international_support_iran: float = 0.3
    
    def update(self, history: GameHistory):
        """Update reputation based on actions"""
        if history.us_strategy in [Strategy.ESCALATION_ULTIMATUM, Strategy.ESCALATION_MILITARY]:
            if history.outcome == Outcome.COMPLIANCE_DELAYED:
                self.us_credibility *= 0.9
            elif history.outcome == Outcome.COMPLIANCE_FULL:
                self.us_credibility = min(1.0, self.us_credibility * 1.1)
        
        if history.iran_strategy == Strategy.CONCILIATORY_MODERATE:
            self.iran_credibility = min(1.0, self.iran_credibility * 1.05)
        elif history.iran_strategy == Strategy.ESCALATION_NUCLEAR:
            self.international_support_iran *= 0.8


@dataclass
class DecisionNode:
    """Node in decision tree"""
    node_id: str
    phase: GamePhase
    player: str
    state: GameState
    reputation: ReputationState
    available_strategies: List[Strategy]
    expected_values: Dict[Strategy, float] = field(default_factory=dict)
    children: List['DecisionNode'] = field(default_factory=list)
    parent: Optional['DecisionNode'] = None
    probability: float = 1.0
    history: List[GameHistory] = field(default_factory=list)


class RepeatedGameModel:
    """
    Models Iran nuclear crisis as repeated game with:
    - Multiple rounds of interaction
    - Reputation effects
    - Learning and adaptation
    - Decision tree visualization
    - Path-dependent outcomes
    """
    
    def __init__(self, base_model: RobustGameTheoryModel):
        self.base_model = base_model
        self.discount_factor = 0.9  # Future payoff discount
        self.reputation_weight = 0.3  # Weight of reputation in decisions
        self.learning_rate = 0.1
        
        # Strategy transition probabilities based on outcomes
        self.transition_probs = self._initialize_transition_probs()
        
        # Decision tree
        self.decision_tree = None
        self.current_node = None
        
    def _initialize_transition_probs(self) -> Dict:
        """Initialize strategy transition probabilities"""
        return {
            # US transitions based on actual Strategy enum values
            "US": {
                Strategy.DETERRENCE_DIPLOMACY: {
                    Outcome.DEAL: {
                        Strategy.DETERRENCE_DIPLOMACY: 0.8,
                        Strategy.DETERRENCE_ULTIMATUM: 0.2
                    },
                    Outcome.LIMITED_RETALIATION: {
                        Strategy.DETERRENCE_DIPLOMACY: 0.3,
                        Strategy.ESCALATION_DIPLOMACY: 0.5,
                        Strategy.ESCALATION_ULTIMATUM: 0.2
                    },
                    Outcome.NUCLEAR_BREAKOUT: {
                        Strategy.ESCALATION_DIPLOMACY: 0.4,
                        Strategy.ESCALATION_ULTIMATUM: 0.6
                    }
                },
                Strategy.ESCALATION_ULTIMATUM: {
                    Outcome.DEAL: {
                        Strategy.DETERRENCE_DIPLOMACY: 0.6,
                        Strategy.DETERRENCE_ULTIMATUM: 0.4
                    },
                    Outcome.FULL_WAR: {
                        Strategy.ESCALATION_ULTIMATUM: 0.7,
                        Strategy.ESCALATION_DIPLOMACY: 0.3
                    }
                }
            },
            # Iran transitions (simplified as Iran strategies not in enum)
            "Iran": {
                "cooperate": {
                    Outcome.DEAL: {
                        "cooperate": 0.9,
                        "resist": 0.1
                    },
                    Outcome.LIMITED_RETALIATION: {
                        "resist": 0.6,
                        "escalate": 0.4
                    }
                },
                "escalate": {
                    Outcome.FULL_WAR: {
                        "escalate": 0.6,
                        "resist": 0.4
                    },
                    Outcome.NUCLEAR_BREAKOUT: {
                        "escalate": 0.8,
                        "cooperate": 0.2
                    }
                }
            }
        }
    
    def simulate_repeated_game(self, 
                             initial_state: GameState,
                             max_rounds: int = 10,
                             convergence_threshold: float = 0.01) -> List[GameHistory]:
        """
        Simulate repeated game with adaptive strategies
        
        Args:
            initial_state: Starting game state
            max_rounds: Maximum rounds to simulate
            convergence_threshold: Stop if strategies stabilize
            
        Returns:
            Game history over all rounds
        """
        
        history = []
        current_state = initial_state
        reputation = ReputationState()
        
        # Initialize decision tree
        self.decision_tree = self._create_initial_node(current_state, reputation)
        self.current_node = self.decision_tree
        
        # Track strategy frequencies for convergence
        us_strategy_freq = defaultdict(int)
        iran_strategy_freq = defaultdict(int)
        
        for round_num in range(max_rounds):
            # Determine current phase
            phase = self._determine_phase(history, current_state)
            
            # US decision
            us_strategy = self._select_strategy_repeated(
                "US", current_state, reputation, history, phase
            )
            us_strategy_freq[us_strategy] += 1
            
            # Iran decision
            iran_strategy = self._select_strategy_repeated(
                "Iran", current_state, reputation, history, phase
            )
            # Convert string to a placeholder enum value for consistency
            if isinstance(iran_strategy, str):
                iran_strategy_str = iran_strategy
            else:
                iran_strategy_str = iran_strategy.value if hasattr(iran_strategy, 'value') else str(iran_strategy)
            
            # Israel decision (simplified)
            israel_action = self._determine_israel_action(
                current_state, us_strategy, iran_strategy, reputation
            )
            
            # Determine outcome
            outcome = self._simulate_outcome(
                us_strategy, iran_strategy, israel_action, current_state, reputation
            )
            
            # Calculate payoffs
            payoffs = self._calculate_payoffs(
                us_strategy, iran_strategy, outcome, current_state, reputation
            )
            
            # Record history
            round_history = GameHistory(
                round=round_num,
                phase=phase,
                us_strategy=us_strategy,
                iran_strategy=iran_strategy,
                israel_action=israel_action,
                outcome=outcome,
                game_state=current_state,
                payoffs=payoffs
            )
            history.append(round_history)
            
            # Update decision tree
            self._update_decision_tree(round_history)
            
            # Update state and reputation
            current_state = self._update_game_state(current_state, outcome, phase)
            reputation.update(round_history)
            
            # Check for convergence
            if round_num > 5:
                if self._check_convergence(history[-5:], convergence_threshold):
                    print(f"Game converged after {round_num + 1} rounds")
                    break
            
            # Check for terminal states
            if self._is_terminal_state(outcome, current_state):
                print(f"Terminal state reached: {outcome.value}")
                break
        
        return history
    
    def _determine_phase(self, history: List[GameHistory], state: GameState) -> GamePhase:
        """Determine current game phase based on history and state"""
        
        if not history:
            return GamePhase.INITIAL_POSTURING
        
        recent_outcomes = [h.outcome for h in history[-3:]]
        
        # Crisis detection
        if (Outcome.REGIONAL_CONFLICT in recent_outcomes or 
            state.nuclear_progress > 0.85):
            return GamePhase.CRISIS
        
        # Resolution detection
        if (Outcome.COMPLIANCE_FULL in recent_outcomes and
            len(recent_outcomes) >= 2 and 
            recent_outcomes[-2] == Outcome.COMPLIANCE_FULL):
            return GamePhase.RESOLUTION
        
        # Escalation detection
        escalation_outcomes = [
            Outcome.SANCTIONS_ENHANCED,
            Outcome.INTERNATIONAL_ISOLATION,
            Outcome.COMPLIANCE_DELAYED
        ]
        if sum(1 for o in recent_outcomes if o in escalation_outcomes) >= 2:
            return GamePhase.ESCALATION
        
        # Negotiation
        if len(history) > 2:
            return GamePhase.NEGOTIATION
        
        return GamePhase.INITIAL_POSTURING
    
    def _select_strategy_repeated(self,
                                player: str,
                                state: GameState,
                                reputation: ReputationState,
                                history: List[GameHistory],
                                phase: GamePhase) -> Strategy:
        """Select strategy considering history and reputation"""
        
        # Get base recommendation
        base_recommendation = self.base_model.bayesian_engine.recommend_strategy(state)
        
        if player == "US":
            available_strategies = [
                Strategy.DETERRENCE_DIPLOMACY,
                Strategy.DETERRENCE_ULTIMATUM,
                Strategy.ESCALATION_DIPLOMACY,
                Strategy.ESCALATION_ULTIMATUM
            ]
        else:  # Iran - using string strategies since not in enum
            available_strategies = ["cooperate", "resist", "escalate"]
        
        # Calculate expected values with reputation
        strategy_values = {}
        
        for strategy in available_strategies:
            # Base value from robust analysis
            base_value = self._get_base_strategy_value(strategy, state, player)
            
            # Reputation adjustment
            rep_adjustment = self._calculate_reputation_adjustment(
                strategy, player, reputation, phase
            )
            
            # Historical performance adjustment
            hist_adjustment = self._calculate_historical_adjustment(
                strategy, player, history
            )
            
            # Future value consideration
            future_value = self._estimate_future_value(
                strategy, player, state, history
            )
            
            total_value = (
                base_value + 
                self.reputation_weight * rep_adjustment +
                self.learning_rate * hist_adjustment +
                self.discount_factor * future_value
            )
            
            strategy_values[strategy] = total_value
        
        # Add noise for exploration
        if phase in [GamePhase.INITIAL_POSTURING, GamePhase.NEGOTIATION]:
            for s in strategy_values:
                strategy_values[s] += np.random.normal(0, 0.1)
        
        # Select best strategy
        best_strategy = max(strategy_values.items(), key=lambda x: x[1])[0]
        
        # Apply transition probabilities if history exists
        if history and player in self.transition_probs:
            last_outcome = history[-1].outcome
            last_strategy = (history[-1].us_strategy if player == "US" 
                           else history[-1].iran_strategy)
            
            if (last_strategy in self.transition_probs[player] and
                last_outcome in self.transition_probs[player][last_strategy]):
                
                transitions = self.transition_probs[player][last_strategy][last_outcome]
                
                # Weighted random choice based on transitions
                if np.random.random() < 0.7:  # 70% follow transitions
                    strategies = list(transitions.keys())
                    probs = list(transitions.values())
                    
                    # Filter to available strategies
                    valid_indices = [i for i, s in enumerate(strategies) 
                                   if s in available_strategies]
                    if valid_indices:
                        strategies = [strategies[i] for i in valid_indices]
                        probs = [probs[i] for i in valid_indices]
                        probs = np.array(probs) / sum(probs)
                        
                        best_strategy = np.random.choice(strategies, p=probs)
        
        return best_strategy
    
    def _get_base_strategy_value(self, 
                               strategy: Strategy, 
                               state: GameState,
                               player: str) -> float:
        """Get base strategy value from robust analysis"""
        
        analysis = self.base_model.analyze_strategy_robustly(strategy, state, n_samples=100)
        
        if player == "US":
            return analysis.expected_utility.mean
        else:
            # For Iran, invert US utility as approximation
            return -analysis.expected_utility.mean * 0.8
    
    def _calculate_reputation_adjustment(self,
                                      strategy: Strategy,
                                      player: str,
                                      reputation: ReputationState,
                                      phase: GamePhase) -> float:
        """Adjust strategy value based on reputation"""
        
        adjustment = 0.0
        
        if player == "US":
            if strategy in [Strategy.ESCALATION_ULTIMATUM, Strategy.ESCALATION_MILITARY]:
                adjustment += reputation.us_credibility * 0.3
                adjustment -= (1 - reputation.international_support_us) * 0.2
            elif strategy in [Strategy.CONCILIATORY_MODERATE, Strategy.CONCILIATORY_INCENTIVES]:
                adjustment += reputation.international_support_us * 0.2
        
        else:  # Iran
            if strategy == Strategy.ESCALATION_NUCLEAR:
                adjustment -= (1 - reputation.iran_credibility) * 0.4
                adjustment -= reputation.international_support_us * 0.3
            elif strategy in [Strategy.CONCILIATORY_MODERATE, Strategy.MODERATE_LIMITED]:
                adjustment += reputation.international_support_iran * 0.3
        
        # Phase adjustments
        if phase == GamePhase.CRISIS:
            if "ESCALATION" in strategy.value:
                adjustment *= 0.5  # De-escalation bias in crisis
        elif phase == GamePhase.RESOLUTION:
            if "CONCILIATORY" in strategy.value:
                adjustment *= 1.5  # Cooperation bias in resolution
        
        return adjustment
    
    def _calculate_historical_adjustment(self,
                                      strategy: Strategy,
                                      player: str,
                                      history: List[GameHistory]) -> float:
        """Adjust based on historical performance"""
        
        if not history:
            return 0.0
        
        # Look at last 5 rounds
        recent_history = history[-5:]
        
        # Calculate success rate for this strategy
        successes = 0
        uses = 0
        
        for h in recent_history:
            if player == "US" and h.us_strategy == strategy:
                uses += 1
                if h.outcome in [Outcome.COMPLIANCE_FULL, Outcome.COMPLIANCE_PARTIAL]:
                    successes += 1
            elif player == "Iran" and h.iran_strategy == strategy:
                uses += 1
                if h.outcome not in [Outcome.REGIONAL_CONFLICT, Outcome.INTERNATIONAL_ISOLATION]:
                    successes += 1
        
        if uses == 0:
            return 0.0  # No data
        
        success_rate = successes / uses
        return (success_rate - 0.5) * 0.5  # Center around 0.5
    
    def _estimate_future_value(self,
                             strategy: Strategy,
                             player: str,
                             state: GameState,
                             history: List[GameHistory]) -> float:
        """Estimate future value of strategy choice"""
        
        # Simplified future value based on likely transitions
        future_value = 0.0
        
        # Check if strategy leads to favorable long-term positions
        if player == "US":
            if strategy in [Strategy.CONCILIATORY_INCENTIVES, Strategy.MODERATE_MONITORING]:
                if state.regime_cohesion < 0.5:
                    future_value += 0.3  # Likely to succeed with weak regime
            elif strategy == Strategy.ESCALATION_MILITARY:
                future_value -= 0.4  # Long-term costs
        
        else:  # Iran
            if strategy == Strategy.ESCALATION_NUCLEAR:
                if state.nuclear_progress > 0.8:
                    future_value += 0.2  # Close to breakout
                else:
                    future_value -= 0.3  # Premature escalation
            elif strategy == Strategy.CONCILIATORY_MODERATE:
                if state.economic_stress > 0.7:
                    future_value += 0.3  # Economic relief priority
        
        return future_value
    
    def _determine_israel_action(self,
                               state: GameState,
                               us_strategy: Strategy,
                               iran_strategy: Strategy,
                               reputation: ReputationState) -> str:
        """Determine Israel's action (simplified)"""
        
        if state.nuclear_progress > 0.85:
            if reputation.israel_restraint < 0.3:
                return "military_strike"
            else:
                return "prepare_strike"
        
        if iran_strategy == Strategy.ESCALATION_NUCLEAR:
            if us_strategy in [Strategy.ESCALATION_MILITARY, Strategy.ESCALATION_ULTIMATUM]:
                return "support_us"
            else:
                return "independent_pressure"
        
        if us_strategy in [Strategy.CONCILIATORY_MODERATE, Strategy.CONCILIATORY_INCENTIVES]:
            return "diplomatic_protest"
        
        return "monitor"
    
    def _simulate_outcome(self,
                        us_strategy: Strategy,
                        iran_strategy: Strategy,
                        israel_action: str,
                        state: GameState,
                        reputation: ReputationState) -> Outcome:
        """Simulate outcome based on strategies and state"""
        
        # Use base model's outcome probabilities
        analysis = self.base_model.analyze_strategy_robustly(us_strategy, state, n_samples=50)
        
        # Get outcome probabilities
        outcome_probs = {}
        for robust_outcome in analysis.outcomes:
            outcome_probs[robust_outcome.outcome] = robust_outcome.probability.mean
        
        # Adjust probabilities based on Iran's strategy
        if iran_strategy == Strategy.CONCILIATORY_MODERATE:
            if Outcome.COMPLIANCE_FULL in outcome_probs:
                outcome_probs[Outcome.COMPLIANCE_FULL] *= 1.5
            if Outcome.NON_COMPLIANCE in outcome_probs:
                outcome_probs[Outcome.NON_COMPLIANCE] *= 0.5
        
        elif iran_strategy == Strategy.ESCALATION_NUCLEAR:
            if Outcome.REGIONAL_CONFLICT in outcome_probs:
                outcome_probs[Outcome.REGIONAL_CONFLICT] *= 2.0
            if Outcome.COMPLIANCE_FULL in outcome_probs:
                outcome_probs[Outcome.COMPLIANCE_FULL] *= 0.1
        
        # Israel action effects
        if israel_action == "military_strike":
            outcome_probs[Outcome.REGIONAL_CONFLICT] = outcome_probs.get(Outcome.REGIONAL_CONFLICT, 0.1) * 3.0
        
        # Normalize probabilities
        total_prob = sum(outcome_probs.values())
        if total_prob > 0:
            outcome_probs = {k: v/total_prob for k, v in outcome_probs.items()}
        else:
            # Fallback
            outcome_probs = {Outcome.COMPLIANCE_DELAYED: 1.0}
        
        # Sample outcome
        outcomes = list(outcome_probs.keys())
        probs = list(outcome_probs.values())
        
        return np.random.choice(outcomes, p=probs)
    
    def _calculate_payoffs(self,
                         us_strategy: Strategy,
                         iran_strategy: Strategy,
                         outcome: Outcome,
                         state: GameState,
                         reputation: ReputationState) -> Dict[str, float]:
        """Calculate payoffs for all players"""
        
        payoffs = {
            "US": 0.0,
            "Iran": 0.0,
            "Israel": 0.0
        }
        
        # Base payoffs from outcome
        outcome_payoffs = {
            Outcome.COMPLIANCE_FULL: {"US": 1.0, "Iran": 0.3, "Israel": 0.9},
            Outcome.COMPLIANCE_PARTIAL: {"US": 0.5, "Iran": 0.4, "Israel": 0.4},
            Outcome.COMPLIANCE_DELAYED: {"US": 0.3, "Iran": 0.5, "Israel": 0.2},
            Outcome.NON_COMPLIANCE: {"US": -0.3, "Iran": 0.6, "Israel": -0.4},
            Outcome.SANCTIONS_ENHANCED: {"US": 0.1, "Iran": -0.5, "Israel": 0.1},
            Outcome.INTERNATIONAL_ISOLATION: {"US": 0.4, "Iran": -0.7, "Israel": 0.3},
            Outcome.REGIONAL_CONFLICT: {"US": -0.8, "Iran": -0.6, "Israel": -0.7},
            Outcome.REGIME_CHANGE: {"US": 0.6, "Iran": -1.0, "Israel": 0.7}
        }
        
        if outcome in outcome_payoffs:
            for player in payoffs:
                payoffs[player] = outcome_payoffs[outcome][player]
        
        # Adjust for strategies
        if us_strategy == Strategy.ESCALATION_MILITARY:
            payoffs["US"] -= 0.3  # Military cost
        
        if iran_strategy == Strategy.ESCALATION_NUCLEAR:
            payoffs["Iran"] -= 0.2  # International cost
            payoffs["Israel"] -= 0.3  # Security threat
        
        # Reputation effects
        payoffs["US"] += reputation.us_credibility * 0.1
        payoffs["Iran"] += reputation.international_support_iran * 0.1
        
        return payoffs
    
    def _update_game_state(self, 
                         current_state: GameState,
                         outcome: Outcome,
                         phase: GamePhase) -> GameState:
        """Update game state based on outcome"""
        
        new_state = GameState(
            regime_cohesion=current_state.regime_cohesion,
            economic_stress=current_state.economic_stress,
            proxy_support=current_state.proxy_support,
            oil_price=current_state.oil_price,
            external_support=current_state.external_support,
            nuclear_progress=current_state.nuclear_progress
        )
        
        # Outcome effects
        if outcome == Outcome.COMPLIANCE_FULL:
            new_state.nuclear_progress *= 0.5
            new_state.economic_stress *= 0.8
        
        elif outcome == Outcome.SANCTIONS_ENHANCED:
            new_state.economic_stress = min(1.0, new_state.economic_stress * 1.2)
            new_state.regime_cohesion *= 0.95
        
        elif outcome == Outcome.REGIONAL_CONFLICT:
            new_state.regime_cohesion *= 0.8
            new_state.economic_stress = min(1.0, new_state.economic_stress * 1.3)
            new_state.proxy_support *= 0.7
        
        elif outcome == Outcome.REGIME_CHANGE:
            new_state.regime_cohesion = 0.1
            new_state.nuclear_progress = 0.1
        
        # Phase effects
        if phase == GamePhase.ESCALATION:
            new_state.nuclear_progress = min(1.0, new_state.nuclear_progress * 1.1)
        
        elif phase == GamePhase.RESOLUTION:
            new_state.economic_stress *= 0.95
        
        # Time progression
        new_state.nuclear_progress = min(1.0, new_state.nuclear_progress + 0.02)
        
        return new_state
    
    def _check_convergence(self, 
                         recent_history: List[GameHistory],
                         threshold: float) -> bool:
        """Check if strategies have converged to equilibrium"""
        
        if len(recent_history) < 3:
            return False
        
        # Check if strategies are repeating
        us_strategies = [h.us_strategy for h in recent_history]
        iran_strategies = [h.iran_strategy for h in recent_history]
        
        # Simple check: same strategies in last 3 rounds
        if (len(set(us_strategies[-3:])) == 1 and 
            len(set(iran_strategies[-3:])) == 1):
            return True
        
        # Check payoff stability
        payoff_variance = np.var([h.payoffs["US"] for h in recent_history])
        
        return payoff_variance < threshold
    
    def _is_terminal_state(self, outcome: Outcome, state: GameState) -> bool:
        """Check if game reached terminal state"""
        
        terminal_outcomes = [
            Outcome.REGIONAL_CONFLICT,
            Outcome.REGIME_CHANGE
        ]
        
        if outcome in terminal_outcomes:
            return True
        
        if state.nuclear_progress >= 0.95:
            return True
        
        if state.regime_cohesion <= 0.1:
            return True
        
        return False
    
    def _create_initial_node(self, 
                           state: GameState,
                           reputation: ReputationState) -> DecisionNode:
        """Create initial decision tree node"""
        
        return DecisionNode(
            node_id="root",
            phase=GamePhase.INITIAL_POSTURING,
            player="US",
            state=state,
            reputation=reputation,
            available_strategies=list(Strategy),
            probability=1.0
        )
    
    def _update_decision_tree(self, history: GameHistory):
        """Update decision tree with new round"""
        
        # Create new node
        new_node = DecisionNode(
            node_id=f"round_{history.round}",
            phase=history.phase,
            player="US",  # Alternates in full implementation
            state=history.game_state,
            reputation=ReputationState(),  # Would track actual reputation
            available_strategies=list(Strategy),
            parent=self.current_node,
            history=self.current_node.history + [history]
        )
        
        # Add expected values
        for strategy in new_node.available_strategies:
            value = self._get_base_strategy_value(
                strategy, history.game_state, "US"
            )
            new_node.expected_values[strategy] = value
        
        # Update tree
        self.current_node.children.append(new_node)
        self.current_node = new_node
    
    def export_decision_tree(self, filename: str = "decision_tree"):
        """Export decision tree for visualization"""
        
        if not self.decision_tree:
            print("No decision tree to export")
            return
        
        # Create networkx graph
        G = nx.DiGraph()
        
        def add_node_recursive(node: DecisionNode, parent_id: Optional[str] = None):
            node_label = f"{node.phase.value}\\n{node.player}"
            
            if node.history:
                last_history = node.history[-1]
                node_label += f"\\nUS: {last_history.us_strategy.value[:15]}..."
                node_label += f"\\nOutcome: {last_history.outcome.value[:15]}..."
            
            G.add_node(node.node_id, label=node_label)
            
            if parent_id:
                G.add_edge(parent_id, node.node_id)
            
            for child in node.children:
                add_node_recursive(child, node.node_id)
        
        add_node_recursive(self.decision_tree)
        
        # Export to DOT format
        import json
        
        dot_data = {
            "nodes": [
                {
                    "id": n,
                    "label": G.nodes[n].get("label", n)
                }
                for n in G.nodes()
            ],
            "edges": [
                {
                    "from": e[0],
                    "to": e[1]
                }
                for e in G.edges()
            ]
        }
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(dot_data, f, indent=2)
        
        print(f"Decision tree exported to {filename}.json")
    
    def analyze_equilibrium_paths(self, 
                                initial_state: GameState,
                                n_simulations: int = 100) -> Dict[str, Any]:
        """Analyze equilibrium strategies through repeated simulations"""
        
        equilibrium_strategies = defaultdict(int)
        final_outcomes = defaultdict(int)
        convergence_rounds = []
        
        for i in range(n_simulations):
            # Run simulation
            history = self.simulate_repeated_game(
                initial_state, 
                max_rounds=20,
                convergence_threshold=0.01
            )
            
            if history:
                # Record final strategies
                final_round = history[-1]
                strategy_pair = (
                    final_round.us_strategy.value,
                    final_round.iran_strategy.value
                )
                equilibrium_strategies[strategy_pair] += 1
                
                # Record final outcome
                final_outcomes[final_round.outcome.value] += 1
                
                # Record convergence time
                convergence_rounds.append(len(history))
        
        # Analyze results
        total_sims = sum(equilibrium_strategies.values())
        
        results = {
            "equilibrium_strategies": {
                k: v/total_sims for k, v in equilibrium_strategies.items()
            },
            "final_outcomes": {
                k: v/n_simulations for k, v in final_outcomes.items()
            },
            "avg_convergence_rounds": np.mean(convergence_rounds),
            "convergence_std": np.std(convergence_rounds)
        }
        
        # Find dominant equilibrium
        dominant_eq = max(equilibrium_strategies.items(), key=lambda x: x[1])
        results["dominant_equilibrium"] = {
            "strategies": dominant_eq[0],
            "frequency": dominant_eq[1] / total_sims
        }
        
        return results