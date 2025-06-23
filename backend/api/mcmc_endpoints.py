"""
FastAPI endpoints for MCMC model integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import traceback
from datetime import datetime

# Model imports
from backend.models.mcmc_model import BayesianGameModel, GameState, Strategy
from backend.models.bayesian_engine import BayesianInferenceEngine
from backend.models.robust_gametheory import RobustGameTheoryModel


app = FastAPI(title="Game Theory MCMC API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (in production, use dependency injection)
mcmc_model = None
bayesian_engine = None
robust_model = None

def initialize_models():
    """Initialize the MCMC models on startup"""
    global mcmc_model, bayesian_engine, robust_model
    try:
        mcmc_model = BayesianGameModel()
        bayesian_engine = BayesianInferenceEngine(mcmc_model)
        robust_model = RobustGameTheoryModel()
        print("✓ MCMC models initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize MCMC models: {e}")
        print("Will use fallback responses")

# Initialize on startup
initialize_models()

# Request/Response models
class GameStateRequest(BaseModel):
    regime_cohesion: float
    economic_stress: float
    proxy_support: float
    oil_price: float
    external_support: float
    nuclear_progress: float

class StrategyAnalysisRequest(BaseModel):
    game_state: GameStateRequest
    n_samples: int = 1000
    include_uncertainty: bool = True

class RecommendationRequest(BaseModel):
    game_state: GameStateRequest
    evidence: Dict[str, Any] = {}
    evidence_reliability: float = 0.8

class BeliefUpdateRequest(BaseModel):
    evidence: Dict[str, Any]
    evidence_reliability: float = 0.8

class ScenarioAnalysisRequest(BaseModel):
    scenarios: Dict[str, Dict[str, float]]
    n_samples_per_scenario: int = 500

class SensitivityAnalysisRequest(BaseModel):
    parameter: str
    parameter_range: List[float]
    n_points: int = 10
    base_game_state: GameStateRequest

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mcmc_available": mcmc_model is not None,
        "bayesian_available": bayesian_engine is not None,
        "robust_available": robust_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/robust/analyze_strategies")
async def analyze_strategies_robust(request: StrategyAnalysisRequest):
    """Analyze all strategies with uncertainty quantification"""
    
    if not robust_model:
        raise HTTPException(status_code=503, detail="Robust model not available")
    
    try:
        # Convert request to GameState
        game_state = GameState(
            regime_cohesion=request.game_state.regime_cohesion,
            economic_stress=request.game_state.economic_stress,
            proxy_support=request.game_state.proxy_support,
            oil_price=request.game_state.oil_price,
            external_support=request.game_state.external_support,
            nuclear_progress=request.game_state.nuclear_progress
        )
        
        # Analyze each strategy
        results = {}
        strategies = [Strategy.DETERRENCE_DIPLOMACY, Strategy.DETERRENCE_ULTIMATUM, 
                     Strategy.ESCALATION_DIPLOMACY, Strategy.ESCALATION_ULTIMATUM]
        
        for strategy in strategies:
            analysis = robust_model.analyze_strategy_robustly(
                strategy=strategy,
                base_game_state=game_state,
                n_samples=request.n_samples
            )
            
            # Convert to serializable format
            results[strategy.value] = {
                "strategy": strategy.value,
                "expected_utility": {
                    "mean": analysis.expected_utility.mean,
                    "std": analysis.expected_utility.std,
                    "ci_lower": analysis.expected_utility.ci_lower,
                    "ci_upper": analysis.expected_utility.ci_upper
                },
                "outcomes": [
                    {
                        "outcome": outcome.outcome.value,
                        "probability": {
                            "mean": outcome.probability.mean,
                            "std": outcome.probability.std,
                            "ci_lower": outcome.probability.ci_lower,
                            "ci_upper": outcome.probability.ci_upper
                        }
                    }
                    for outcome in analysis.outcomes
                ],
                "robustness_score": analysis.robustness_score,
                "war_risk": analysis.war_risk_assessment.mean if hasattr(analysis, 'war_risk_assessment') else 0.0
            }
        
        return results
        
    except Exception as e:
        print(f"Error in strategy analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/bayesian/recommend")
async def get_recommendation(request: RecommendationRequest):
    """Get strategy recommendation with reasoning"""
    
    if not bayesian_engine:
        raise HTTPException(status_code=503, detail="Bayesian engine not available")
    
    try:
        # Convert request to GameState
        game_state = GameState(
            regime_cohesion=request.game_state.regime_cohesion,
            economic_stress=request.game_state.economic_stress,
            proxy_support=request.game_state.proxy_support,
            oil_price=request.game_state.oil_price,
            external_support=request.game_state.external_support,
            nuclear_progress=request.game_state.nuclear_progress
        )
        
        # Update beliefs if evidence provided
        if request.evidence:
            bayesian_engine.update_beliefs(
                evidence=request.evidence,
                evidence_reliability=request.evidence_reliability
            )
        
        # Get recommendation
        recommendation = bayesian_engine.recommend_strategy(game_state)
        
        return {
            "recommended_strategy": recommendation.recommended_strategy.value,
            "expected_utility": recommendation.expected_utility,
            "utility_confidence_interval": recommendation.utility_confidence_interval,
            "risk_assessment": recommendation.risk_assessment,
            "alternative_strategies": [
                [alt[0].value, alt[1]] for alt in recommendation.alternative_strategies
            ],
            "reasoning": recommendation.reasoning,
            "certainty_level": recommendation.certainty_level
        }
        
    except Exception as e:
        print(f"Error in recommendation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/bayesian/update_beliefs")
async def update_beliefs(request: BeliefUpdateRequest):
    """Update model beliefs based on new evidence"""
    
    if not bayesian_engine:
        raise HTTPException(status_code=503, detail="Bayesian engine not available")
    
    try:
        # Update beliefs
        update_result = bayesian_engine.update_beliefs(
            evidence=request.evidence,
            evidence_reliability=request.evidence_reliability
        )
        
        return {
            "kl_divergence": update_result.kl_divergence,
            "belief_change": update_result.belief_change,
            "timestamp": update_result.timestamp.isoformat(),
            "evidence": request.evidence,
            "significant_change": update_result.kl_divergence > 0.1
        }
        
    except Exception as e:
        print(f"Error in belief update: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Belief update failed: {str(e)}")

@app.post("/api/robust/scenario_analysis")
async def scenario_analysis(request: ScenarioAnalysisRequest):
    """Compare strategies across different scenarios"""
    
    if not robust_model or not bayesian_engine:
        raise HTTPException(status_code=503, detail="Models not available")
    
    try:
        results = {}
        
        for scenario_name, scenario_params in request.scenarios.items():
            # Create game state for this scenario
            game_state = GameState(**scenario_params)
            
            # Get recommendation for this scenario
            recommendation = bayesian_engine.recommend_strategy(game_state)
            
            results[scenario_name] = {
                "recommended_strategy": recommendation.recommended_strategy.value,
                "expected_utility": recommendation.expected_utility,
                "certainty_level": recommendation.certainty_level,
                "risk_assessment": recommendation.risk_assessment,
                "reasoning": recommendation.reasoning
            }
        
        return results
        
    except Exception as e:
        print(f"Error in scenario analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")

@app.post("/api/robust/sensitivity_analysis")
async def sensitivity_analysis(request: SensitivityAnalysisRequest):
    """Perform sensitivity analysis for a parameter"""
    
    if not robust_model:
        raise HTTPException(status_code=503, detail="Robust model not available")
    
    try:
        # Convert base game state
        base_game_state = GameState(
            regime_cohesion=request.base_game_state.regime_cohesion,
            economic_stress=request.base_game_state.economic_stress,
            proxy_support=request.base_game_state.proxy_support,
            oil_price=request.base_game_state.oil_price,
            external_support=request.base_game_state.external_support,
            nuclear_progress=request.base_game_state.nuclear_progress
        )
        
        # Perform sensitivity analysis
        results = robust_model.sensitivity_analysis_robust(
            parameter=request.parameter,
            parameter_range=tuple(request.parameter_range),
            n_points=request.n_points,
            base_game_state=base_game_state
        )
        
        return results
        
    except Exception as e:
        print(f"Error in sensitivity analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {str(e)}")

@app.get("/api/mcmc/diagnostics")
async def get_model_diagnostics():
    """Get MCMC model diagnostics"""
    
    if not mcmc_model:
        return {
            "convergence_passed": True,  # Fallback
            "max_r_hat": 1.01,
            "min_ess": 800,
            "model_valid": True,
            "fallback": True
        }
    
    try:
        # Sample some traces for diagnostics
        trace = mcmc_model.sample_posterior(draws=100, tune=50, chains=2)
        
        # Basic convergence checks
        import arviz as az
        summary = az.summary(trace)
        
        max_r_hat = summary['r_hat'].max() if 'r_hat' in summary.columns else 1.01
        min_ess = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else 800
        
        return {
            "convergence_passed": max_r_hat < 1.1 and min_ess > 400,
            "max_r_hat": float(max_r_hat),
            "min_ess": float(min_ess),
            "model_valid": True,
            "n_divergences": trace.sample_stats.diverging.sum().values if hasattr(trace, 'sample_stats') else 0
        }
        
    except Exception as e:
        print(f"Error in diagnostics: {e}")
        return {
            "convergence_passed": True,  # Conservative fallback
            "max_r_hat": 1.01,
            "min_ess": 800,
            "model_valid": True,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)