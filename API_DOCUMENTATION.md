# API Documentation: Game Theory Iran Model

## Core Classes

### `BayesianGameModel`
Main MCMC model for strategic game theory analysis.

```python
class BayesianGameModel:
    def __init__(self, random_seed: int = 42)
    def build_model(self, observed_data: Optional[pd.DataFrame] = None) -> pm.Model
    def sample_posterior(self, draws: int = 2000, tune: int = 1000, 
                        chains: int = 4, target_accept: float = 0.95) -> az.InferenceData
    def analyze_strategies(self) -> Dict[Strategy, Dict[Outcome, Tuple[float, float, float]]]
    def convergence_diagnostics(self) -> Dict[str, float]
    def get_optimal_strategy(self, usa_preferences: Dict[Outcome, float]) -> Tuple[Strategy, float]
```

#### Methods

##### `__init__(random_seed: int = 42)`
Initialize the Bayesian game model.
- **Parameters:**
  - `random_seed`: Random seed for reproducibility

##### `build_model(observed_data: Optional[pd.DataFrame] = None) -> pm.Model`
Build the hierarchical Bayesian model structure.
- **Parameters:**
  - `observed_data`: Optional historical data for likelihood
- **Returns:** PyMC model object

##### `sample_posterior(...) -> az.InferenceData`
Sample from the posterior distribution using NUTS.
- **Parameters:**
  - `draws`: Number of posterior samples per chain
  - `tune`: Number of tuning steps
  - `chains`: Number of parallel MCMC chains
  - `target_accept`: Target acceptance probability
- **Returns:** ArviZ InferenceData with posterior samples

##### `analyze_strategies() -> Dict[Strategy, Dict[Outcome, Tuple[float, float, float]]]`
Analyze all strategies with uncertainty quantification.
- **Returns:** Dictionary mapping strategies to outcomes with (mean, lower_ci, upper_ci)

### `BayesianInferenceEngine`
Engine for Bayesian inference and strategy optimization.

```python
class BayesianInferenceEngine:
    def __init__(self, mcmc_model: Optional[BayesianGameModel] = None)
    def update_beliefs(self, evidence: Dict[str, Any], 
                      evidence_reliability: float = 0.8) -> BayesianUpdate
    def recommend_strategy(self, game_state: GameState,
                          usa_preferences: Optional[Dict[Outcome, float]] = None) -> StrategyRecommendation
    def counterfactual_analysis(self, base_game_state: GameState,
                               counterfactual_changes: Dict[str, float]) -> Dict[str, Any]
```

#### Methods

##### `update_beliefs(evidence: Dict[str, Any], evidence_reliability: float = 0.8) -> BayesianUpdate`
Update beliefs based on new evidence using Bayes' theorem.
- **Parameters:**
  - `evidence`: Dictionary of observed evidence
  - `evidence_reliability`: Reliability score for evidence (0-1)
- **Returns:** BayesianUpdate with prior, posterior, and KL divergence

##### `recommend_strategy(...) -> StrategyRecommendation`
Generate optimal strategy recommendation with uncertainty.
- **Parameters:**
  - `game_state`: Current state of the game
  - `usa_preferences`: Utility function (optional)
- **Returns:** StrategyRecommendation with confidence intervals

### `RobustGameTheoryModel`
Enhanced model with uncertainty quantification.

```python
class RobustGameTheoryModel:
    def __init__(self, mcmc_samples: int = 2000, random_seed: int = 42)
    def analyze_strategy_robustly(self, strategy: Strategy,
                                 base_game_state: Optional[GameState] = None,
                                 n_samples: int = 1000) -> StrategyAnalysis
    def compare_strategies_robust(self, strategies: List[Strategy],
                                 base_game_state: Optional[GameState] = None) -> Dict[Strategy, StrategyAnalysis]
    def sensitivity_analysis_robust(self, parameter: str,
                                   parameter_range: Tuple[float, float],
                                   n_points: int = 10) -> Dict[str, Any]
```

## Data Classes

### `GameState`
Current state of the strategic game.

```python
@dataclass
class GameState:
    regime_cohesion: float     # 0-1, Iran's internal stability
    economic_stress: float     # 0-1, Economic pressure level
    proxy_support: float       # 0-1, Proxy network strength
    oil_price: float          # 50-150, USD per barrel
    external_support: float    # 0-1, China/Russia support
    nuclear_progress: float    # 0-1, Nuclear program advancement
```

### `StrategyRecommendation`
AI-generated strategy recommendation.

```python
@dataclass
class StrategyRecommendation:
    recommended_strategy: Strategy
    expected_utility: float
    utility_confidence_interval: Tuple[float, float]
    risk_assessment: Dict[str, float]
    alternative_strategies: List[Tuple[Strategy, float]]
    reasoning: str
    certainty_level: float  # 0-1, confidence in recommendation
```

### `UncertaintyBounds`
Uncertainty quantification for any metric.

```python
@dataclass
class UncertaintyBounds:
    mean: float              # Point estimate
    std: float               # Standard deviation
    ci_lower: float          # 2.5th percentile
    ci_upper: float          # 97.5th percentile
    percentile_25: float     # First quartile
    percentile_75: float     # Third quartile
```

### `StrategyAnalysis`
Complete analysis results with uncertainty.

```python
@dataclass
class StrategyAnalysis:
    strategy: Strategy
    expected_utility: UncertaintyBounds
    outcomes: List[RobustOutcome]
    risk_assessment: Dict[str, UncertaintyBounds]
    robustness_score: float  # 0-1, higher is more robust
    regret_analysis: Dict[str, float]
```

### `BayesianUpdate`
Record of belief update process.

```python
@dataclass
class BayesianUpdate:
    timestamp: datetime
    prior_belief: Dict[str, float]
    evidence: Dict[str, Any]
    posterior_belief: Dict[str, float]
    kl_divergence: float  # Measure of belief change
```

## Enumerations

### `Strategy`
US strategic options.

```python
class Strategy(Enum):
    DETERRENCE_DIPLOMACY = "halt_deter_diplomacy"      # Halt & Deter + De-escalatory Off-Ramp
    DETERRENCE_ULTIMATUM = "halt_deter_ultimatum"      # Halt & Deter + Coercive Ultimatum
    ESCALATION_DIPLOMACY = "expand_strikes_diplomacy"  # Expand Strikes + De-escalatory Off-Ramp
    ESCALATION_ULTIMATUM = "expand_strikes_ultimatum"  # Expand Strikes + Coercive Ultimatum
```

### `Outcome`
Possible game outcomes.

```python
class Outcome(Enum):
    DEAL = "negotiated_deal"                    # Diplomatic resolution
    LIMITED_RETALIATION = "limited_retaliation" # Contained military response
    FROZEN_CONFLICT = "frozen_conflict"         # Prolonged standoff
    FULL_WAR = "full_war"                      # Regional escalation
    NUCLEAR_BREAKOUT = "nuclear_breakout"       # Iran gets nuclear weapons
```

## Utility Classes

### `GameStateSimulator`
Monte Carlo simulator for game states.

```python
class GameStateSimulator:
    def __init__(self, config: SamplingConfig)
    def simulate_game_states(self, base_state: Optional[GameState] = None,
                           correlation_strength: float = 0.7) -> List[GameState]
    def scenario_analysis(self, scenario_definitions: Dict[str, Dict[str, float]],
                         n_samples_per_scenario: int = 1000) -> Dict[str, List[GameState]]
    def sensitivity_sampling(self, parameter_name: str,
                           parameter_range: Tuple[float, float]) -> Dict[str, np.ndarray]
```

### `ModelValidator`
Comprehensive model validation framework.

```python
class ModelValidator:
    def __init__(self, model: BayesianGameModel, 
                 validation_data: Optional[pd.DataFrame] = None)
    def full_validation(self, n_posterior_samples: int = 1000,
                       n_cross_val_folds: int = 5) -> ValidationResults
    def convergence_diagnostics(self) -> Dict[str, float]
    def posterior_predictive_checks(self, n_samples: int = 1000) -> Dict[str, float]
    def generate_validation_report(self, results: ValidationResults,
                                  save_path: Optional[str] = None) -> str
```

## Configuration Classes

### `SamplingConfig`
Configuration for Monte Carlo sampling.

```python
@dataclass
class SamplingConfig:
    n_samples: int = 10000              # Number of samples
    n_bootstrap: int = 1000             # Bootstrap iterations
    confidence_level: float = 0.95      # Confidence level for intervals
    random_seed: Optional[int] = 42     # Random seed
    sampling_method: str = "lhs"        # "random", "lhs", "sobol", "halton"
    antithetic: bool = True            # Use antithetic variates
```

## Usage Examples

### Basic Strategy Analysis

```python
# Initialize model and engine
model = BayesianGameModel()
engine = BayesianInferenceEngine(model)

# Define game state
state = GameState(
    regime_cohesion=0.4,
    economic_stress=0.9,
    proxy_support=0.2,
    oil_price=97.0,
    external_support=0.3,
    nuclear_progress=0.7
)

# Get recommendation
rec = engine.recommend_strategy(state)
print(f"Strategy: {rec.recommended_strategy}")
print(f"Utility: {rec.expected_utility:.3f} "
      f"({rec.utility_confidence_interval[0]:.3f}, "
      f"{rec.utility_confidence_interval[1]:.3f})")
```

### Robust Analysis with Uncertainty

```python
# Create robust model
robust_model = RobustGameTheoryModel()

# Compare all strategies
strategies = list(Strategy)
analyses = robust_model.compare_strategies_robust(strategies, state)

# Find most robust strategy
best_strategy = max(analyses.items(), 
                   key=lambda x: x[1].robustness_score)
print(f"Most robust: {best_strategy[0].value}")
```

### Scenario Testing

```python
# Define scenarios
scenarios = {
    "base_case": state,
    "regime_collapse": GameState(
        regime_cohesion=0.1,
        economic_stress=0.95,
        proxy_support=0.05,
        oil_price=97.0,
        external_support=0.1,
        nuclear_progress=0.85
    )
}

# Analyze each scenario
for name, scenario_state in scenarios.items():
    rec = engine.recommend_strategy(scenario_state)
    print(f"{name}: {rec.recommended_strategy.value}")
```

### Real-time Updates

```python
# Initial assessment
initial_rec = engine.recommend_strategy(state)

# New evidence
evidence = {
    "regime_stability_signal": 0.2,
    "nuclear_activity_detected": True
}

# Update beliefs
update = engine.update_beliefs(evidence, evidence_reliability=0.9)

# New recommendation
new_rec = engine.recommend_strategy(state)

if initial_rec.recommended_strategy != new_rec.recommended_strategy:
    print("Strategy change recommended!")
```

### Model Validation

```python
# Create validator
validator = ModelValidator(model)

# Run validation
results = validator.full_validation()

# Check convergence
if results.convergence_diagnostics['convergence_passed']:
    print("Model converged successfully")

# Generate report
report = validator.generate_validation_report(results, "validation.txt")
```

## Error Handling

### Common Exceptions

```python
try:
    rec = engine.recommend_strategy(state)
except ValueError as e:
    # Invalid game state parameters
    print(f"Invalid state: {e}")
except RuntimeError as e:
    # MCMC sampling failed
    print(f"Sampling error: {e}")
```

### Validation Errors

```python
# Check state validity
def validate_game_state(state: GameState) -> bool:
    """Validate game state parameters."""
    if not 0 <= state.regime_cohesion <= 1:
        raise ValueError("regime_cohesion must be in [0,1]")
    if not 50 <= state.oil_price <= 150:
        raise ValueError("oil_price must be in [50,150]")
    # ... other checks
    return True
```

## Performance Considerations

### Sampling Performance

- **Quick analysis**: 500-1000 draws, 2 chains
- **Standard analysis**: 2000 draws, 4 chains
- **Publication quality**: 5000+ draws, 4+ chains

### Memory Usage

- Each chain requires ~100MB for 1000 draws
- Use fewer chains on memory-constrained systems
- Consider sampling methods: "random" < "lhs" < "sobol"

### Optimization Tips

1. **Cache results**: Store computed strategies
2. **Parallel sampling**: Use multiple chains
3. **Reduce dimensions**: Fix non-critical parameters
4. **Simplify model**: Use fewer outcome states

## Extension Points

### Adding New Strategies

```python
# In Strategy enum
class Strategy(Enum):
    # ... existing strategies
    CYBER_DIPLOMACY = "cyber_warfare_diplomacy"  # New strategy

# Update strategy analysis
def analyze_cyber_strategy(self, game_state: GameState):
    # Custom logic for cyber warfare strategy
    pass
```

### Custom Risk Metrics

```python
def compute_custom_risk(self, strategy: Strategy, 
                       game_state: GameState) -> float:
    """Add custom risk calculation."""
    base_risk = 0.1
    
    # Adjust based on strategy
    if "escalation" in strategy.value:
        base_risk *= 2.0
        
    # Adjust based on state
    if game_state.nuclear_progress > 0.8:
        base_risk *= 1.5
        
    return min(base_risk, 1.0)
```

### New Outcome Types

```python
# Add to Outcome enum
class Outcome(Enum):
    # ... existing outcomes
    CYBER_CONFLICT = "cyber_warfare_only"
    ECONOMIC_COLLAPSE = "iranian_economic_collapse"
```

## Version Information

- **Current Version**: 1.0.0
- **Python Required**: 3.11+
- **Key Dependencies**: 
  - pymc >= 5.9.0
  - arviz >= 0.16.0
  - numpy >= 1.24.0
  - pandas >= 2.0.0