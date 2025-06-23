# User Guide: Game Theory Iran Model

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the Model](#understanding-the-model)
3. [Using the Web Interface](#using-the-web-interface)
4. [Python API Tutorial](#python-api-tutorial)
5. [Interpreting Results](#interpreting-results)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and set up the project**:
```bash
git clone https://github.com/yourusername/game-theory-iran.git
cd game-theory-iran
uv sync
```

3. **Start the application**:
```bash
uv run python run_d3_app.py
```

Navigate to `http://localhost:8000` in your browser.

## Understanding the Model

### The Strategic Situation

This model analyzes the strategic interactions between three key players:

- **USA**: Seeking to prevent Iranian nuclear weapons while avoiding war
- **Iran**: Balancing regime survival with nuclear ambitions
- **Israel**: Pushing for decisive action against Iranian threats

### Game Variables Explained

#### 1. **Regime Cohesion** (0-1)
- **Low (0-0.3)**: Iranian leadership is fractured, unpredictable
- **Medium (0.3-0.7)**: Some internal divisions but functional
- **High (0.7-1)**: United leadership, predictable behavior

#### 2. **Economic Stress** (0-1)
- **Low (0-0.3)**: Economy stable despite sanctions
- **Medium (0.3-0.7)**: Significant pressure but manageable
- **High (0.7-1)**: Near economic collapse

#### 3. **Proxy Support** (0-1)
- **Low (0-0.3)**: Proxy networks degraded/ineffective
- **Medium (0.3-0.7)**: Some proxy capability remains
- **High (0.7-1)**: Full proxy network operational

#### 4. **Oil Price** (50-150 USD)
- Affects global economic pressure and Iran's leverage
- Higher prices = more Iranian revenue and global caution

#### 5. **External Support** (0-1)
- **Low (0-0.3)**: Minimal China/Russia backing
- **Medium (0.3-0.7)**: Some diplomatic/economic support
- **High (0.7-1)**: Full backing including military aid

#### 6. **Nuclear Progress** (0-1)
- **Low (0-0.3)**: Years from weaponization
- **Medium (0.3-0.7)**: Months from breakout
- **High (0.7-1)**: Weeks or less from nuclear weapon

## Using the Web Interface

### Main Dashboard

1. **Control Panel** (Left Side)
   - Adjust game variables using sliders
   - See real-time updates to probabilities
   - Values update automatically

2. **Visualization Area** (Center)
   - **Escalation Heatmap**: Shows war risk for each strategy
   - **Strategy Comparison**: Risk vs reward scatter plot
   - **Outcome Probabilities**: Likelihood of each outcome

3. **Export Options** (Top Right)
   - Click "Export Graphics" to save visualizations
   - Choose individual charts or export all

### Reading the Escalation Heatmap

The heatmap is the most important visualization:

```
                 De-escalatory     Coercive
                  Off-Ramp        Ultimatum
    ┌─────────┬──────────────┬──────────────┐
    │ Halt &  │   15% 🕊️     │   25% ⚠️     │
    │ Deter   │   (Green)    │  (Yellow)    │
    ├─────────┼──────────────┼──────────────┤
    │ Expand  │   35% 🔥     │   65% 💀     │
    │ Strikes │  (Orange)    │   (Red)      │
    └─────────┴──────────────┴──────────────┘
```

- **Green (0-20%)**: Low escalation risk
- **Yellow (20-40%)**: Moderate risk
- **Orange (40-60%)**: High risk
- **Red (60%+)**: Extreme risk

### Interactive Features

1. **Hover Effects**
   - Hover over any visualization element for details
   - See confidence intervals and risk metrics
   - View outcome breakdowns

2. **Real-time Updates**
   - Change sliders to see immediate impact
   - Watch probability distributions shift
   - Observe strategy rankings change

3. **Creative Mode Toggle**
   - Default: Cyberpunk aesthetic with animations
   - Clean Mode: Professional, minimal design
   - Toggle in settings or code

## Python API Tutorial

### Basic Analysis

```python
# Import required modules
from backend.models.mcmc_model import BayesianGameModel, GameState
from backend.models.bayesian_engine import BayesianInferenceEngine

# Initialize the model
model = BayesianGameModel()
engine = BayesianInferenceEngine(model)

# Create a game state
current_situation = GameState(
    regime_cohesion=0.3,    # Weak regime
    economic_stress=0.9,    # Near collapse
    proxy_support=0.1,      # Proxies defeated
    oil_price=95.0,         # Moderate oil prices
    external_support=0.2,   # Limited backing
    nuclear_progress=0.8    # Close to breakout
)

# Get recommendation
rec = engine.recommend_strategy(current_situation)
print(f"Recommended: {rec.recommended_strategy.value}")
print(f"Confidence: {rec.certainty_level:.1%}")
```

### Scenario Comparison

```python
# Define multiple scenarios
scenarios = {
    "current": GameState(
        regime_cohesion=0.4,
        economic_stress=0.9,
        proxy_support=0.2,
        oil_price=97.0,
        external_support=0.3,
        nuclear_progress=0.7
    ),
    "regime_collapse": GameState(
        regime_cohesion=0.1,
        economic_stress=0.95,
        proxy_support=0.05,
        oil_price=97.0,
        external_support=0.1,
        nuclear_progress=0.85
    ),
    "chinese_intervention": GameState(
        regime_cohesion=0.6,
        economic_stress=0.5,
        proxy_support=0.3,
        oil_price=120.0,
        external_support=0.9,
        nuclear_progress=0.7
    )
}

# Analyze each scenario
for name, state in scenarios.items():
    rec = engine.recommend_strategy(state)
    print(f"\n{name.upper()} Scenario:")
    print(f"  Strategy: {rec.recommended_strategy.value}")
    print(f"  War Risk: {rec.risk_assessment['escalation_risk']:.1%}")
```

### Uncertainty Analysis

```python
from backend.models.robust_gametheory import RobustGameTheoryModel

# Create robust model
robust = RobustGameTheoryModel()

# Analyze with uncertainty
from backend.models.mcmc_model import Strategy
analysis = robust.analyze_strategy_robustly(
    strategy=Strategy.DETERRENCE_DIPLOMACY,
    base_game_state=current_situation,
    n_samples=5000  # More samples = better uncertainty estimates
)

# Print results with confidence intervals
print(f"Expected Utility: {analysis.expected_utility.mean:.3f}")
print(f"95% Confidence Interval: [{analysis.expected_utility.ci_lower:.3f}, "
      f"{analysis.expected_utility.ci_upper:.3f}]")

# Check robustness
if analysis.robustness_score > 0.7:
    print("✓ This strategy is robust to parameter uncertainty")
else:
    print("⚠️ This strategy is sensitive to parameter changes")
```

### Real-time Updates

```python
# Initial assessment
initial = engine.recommend_strategy(current_situation)

# New intelligence arrives
new_evidence = {
    "regime_stability_signal": 0.15,  # Regime weaker than thought
    "nuclear_activity_detected": True,
    "chinese_military_movements": True
}

# Update beliefs
update = engine.update_beliefs(
    evidence=new_evidence,
    evidence_reliability=0.9  # High confidence in intel
)

# Get new recommendation
updated = engine.recommend_strategy(current_situation)

# Check if strategy changed
if initial.recommended_strategy != updated.recommended_strategy:
    print(f"STRATEGY CHANGE RECOMMENDED!")
    print(f"Was: {initial.recommended_strategy.value}")
    print(f"Now: {updated.recommended_strategy.value}")
```

## Interpreting Results

### Strategy Recommendations

1. **Deterrence + Diplomacy** ✅
   - Best when: Regime is rational, nuclear progress moderate
   - Risk: May appear weak if not credible
   - Success factors: Clear red lines, viable off-ramps

2. **Deterrence + Ultimatum** ⚠️
   - Best when: Need to signal resolve, regime is cohesive
   - Risk: Can trigger "cornered animal" response
   - Success factors: Credible threats, allied unity

3. **Escalation + Diplomacy** 🔥
   - Best when: Deterrence has failed, need to regain initiative
   - Risk: High escalation potential
   - Success factors: Limited strikes, clear end-state

4. **Escalation + Ultimatum** 💀
   - Best when: Imminent nuclear breakout, no alternatives
   - Risk: Highest war probability
   - Success factors: Overwhelming force, regime collapse

### Understanding Uncertainty

All predictions come with uncertainty bounds:

```
War Probability: 35% (25%-45%)
                 │    └─────┴── 95% Confidence Interval
                 └── Point Estimate
```

- **Narrow intervals** (±5%): High confidence
- **Medium intervals** (±10%): Moderate confidence  
- **Wide intervals** (±20%+): High uncertainty

### Risk Metrics

1. **Escalation Risk**: Probability of uncontrolled escalation
2. **Miscalculation Risk**: Chance of misreading opponent
3. **Third-party Risk**: Allied or enemy intervention
4. **Domestic Risk**: US political constraints

## Advanced Features

### Custom Scenario Testing

```python
# Test specific parameter combinations
from backend.utils.monte_carlo import create_default_simulator

simulator = create_default_simulator()

# Define custom scenario
my_scenario = {
    "regime_cohesion": 0.2,
    "economic_stress": 0.95,
    "proxy_support": 0.0,
    "oil_price": 130.0,
    "external_support": 0.4,
    "nuclear_progress": 0.9
}

# Generate variations
states = simulator.simulate_game_states(
    GameState(**my_scenario)
)

# Analyze across variations
from collections import Counter
recommendations = []
for state in states[:100]:
    rec = engine.recommend_strategy(state)
    recommendations.append(rec.recommended_strategy)

# See strategy distribution
strategy_counts = Counter(recommendations)
for strategy, count in strategy_counts.most_common():
    print(f"{strategy.value}: {count/100:.0%}")
```

### Model Validation

```python
from backend.utils.model_validation import ModelValidator

# Validate the model
validator = ModelValidator(model)
results = validator.full_validation()

# Check convergence
if results.convergence_diagnostics['convergence_passed']:
    print("✓ Model converged successfully")
else:
    print("⚠️ Convergence issues detected")

# Generate full report
report = validator.generate_validation_report(results)
with open("validation_report.txt", "w") as f:
    f.write(report)
```

### Sensitivity Analysis

```python
# Test sensitivity to specific parameters
sensitivity_results = robust.sensitivity_analysis_robust(
    parameter="nuclear_progress",
    parameter_range=(0.5, 0.95),
    n_points=10,
    base_game_state=current_situation
)

# Plot results (if matplotlib installed)
import matplotlib.pyplot as plt

param_values = sensitivity_results["parameter_values"]
for strategy, data in sensitivity_results["strategies"].items():
    plt.plot(param_values, data["expected_utility_mean"], 
             label=strategy.value)
    plt.fill_between(param_values,
                     data["expected_utility_ci_lower"],
                     data["expected_utility_ci_upper"],
                     alpha=0.3)

plt.xlabel("Nuclear Progress")
plt.ylabel("Expected Utility")
plt.legend()
plt.title("Strategy Sensitivity to Nuclear Progress")
plt.show()
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'pymc'
   ```
   Solution: Run `uv sync` to install dependencies

2. **Port Already in Use**
   ```
   Address already in use
   ```
   Solution: Kill existing process or use different port:
   ```bash
   lsof -i :8000
   kill <PID>
   ```

3. **Slow Performance**
   - Reduce MCMC samples: `draws=500` instead of `2000`
   - Use fewer Monte Carlo samples
   - Disable creative visualizations

4. **Memory Issues**
   - Close other applications
   - Reduce chain count: `chains=2` instead of `4`
   - Use sampling method `"random"` instead of `"lhs"`

### Getting Help

1. Check error messages in console
2. Review validation report for model issues
3. Open GitHub issue with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

### Performance Tips

1. **For Quick Analysis**:
   - Use fewer samples (500-1000)
   - Skip validation steps
   - Use point estimates

2. **For Publication**:
   - Use more samples (5000+)
   - Run full validation
   - Include uncertainty bounds

3. **For Real-time Use**:
   - Cache results
   - Pre-compute scenarios
   - Use simplified model

## Next Steps

1. **Explore the research**: Read `research.md` for theoretical background
2. **Customize the model**: Modify priors in `mcmc_model.py`
3. **Add new features**: Extend visualizations or add new strategies
4. **Contribute**: Submit pull requests with improvements

Remember: This model is a decision support tool. Always combine with expert judgment for real decisions.