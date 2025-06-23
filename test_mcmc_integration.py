#!/usr/bin/env python3
"""
Test script for MCMC integration

This script tests the MCMC model integration by making API calls
and verifying responses.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

API_BASE = "http://localhost:8001"

async def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data['status']}")
                    print(f"   MCMC available: {data['mcmc_available']}")
                    print(f"   Bayesian available: {data['bayesian_available']}")
                    print(f"   Robust available: {data['robust_available']}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

async def test_strategy_analysis():
    """Test strategy analysis with uncertainty"""
    print("\n📊 Testing strategy analysis...")
    
    test_request = {
        "game_state": {
            "regime_cohesion": 0.4,
            "economic_stress": 0.9,
            "proxy_support": 0.1,
            "oil_price": 97.0,
            "external_support": 0.2,
            "nuclear_progress": 0.7
        },
        "n_samples": 100,  # Smaller for testing
        "include_uncertainty": True
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE}/api/robust/analyze_strategies",
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Strategy analysis successful")
                    
                    # Check that we have results for all strategies
                    strategies = ['deterrence_diplomacy', 'deterrence_ultimatum', 
                                'escalation_diplomacy', 'escalation_ultimatum']
                    
                    for strategy in strategies:
                        if strategy in data:
                            utility = data[strategy]['expected_utility']
                            print(f"   {strategy}: utility={utility['mean']:.3f} "
                                  f"(±{utility['std']:.3f})")
                        else:
                            print(f"   ⚠️ Missing strategy: {strategy}")
                    
                    return True
                else:
                    print(f"❌ Strategy analysis failed: {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    return False
        except asyncio.TimeoutError:
            print("❌ Strategy analysis timed out")
            return False
        except Exception as e:
            print(f"❌ Strategy analysis error: {e}")
            return False

async def test_recommendation():
    """Test strategy recommendation"""
    print("\n🎯 Testing strategy recommendation...")
    
    test_request = {
        "game_state": {
            "regime_cohesion": 0.4,
            "economic_stress": 0.9,
            "proxy_support": 0.1,
            "oil_price": 97.0,
            "external_support": 0.2,
            "nuclear_progress": 0.7
        },
        "evidence": {},
        "evidence_reliability": 0.8
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE}/api/bayesian/recommend",
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Recommendation successful")
                    print(f"   Recommended: {data['recommended_strategy']}")
                    print(f"   Confidence: {data['certainty_level']:.3f}")
                    print(f"   Utility: {data['expected_utility']:.3f}")
                    print(f"   Reasoning: {data['reasoning'][:100]}...")
                    return True
                else:
                    print(f"❌ Recommendation failed: {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    return False
        except asyncio.TimeoutError:
            print("❌ Recommendation timed out")
            return False
        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            return False

async def test_diagnostics():
    """Test model diagnostics"""
    print("\n🔧 Testing model diagnostics...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{API_BASE}/api/mcmc/diagnostics",
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Diagnostics successful")
                    print(f"   Convergence passed: {data['convergence_passed']}")
                    print(f"   Max R-hat: {data['max_r_hat']:.3f}")
                    print(f"   Min ESS: {data['min_ess']:.0f}")
                    return True
                else:
                    print(f"❌ Diagnostics failed: {response.status}")
                    return False
        except asyncio.TimeoutError:
            print("❌ Diagnostics timed out")
            return False
        except Exception as e:
            print(f"❌ Diagnostics error: {e}")
            return False

async def main():
    """Run all tests"""
    print("🧪 Testing MCMC Integration")
    print("=" * 50)
    
    tests = [
        test_health_check(),
        test_strategy_analysis(),
        test_recommendation(),
        test_diagnostics()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Count successes
    successes = sum(1 for result in results if result is True)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {successes}/{total} passed")
    
    if successes == total:
        print("🎉 All tests passed! MCMC integration is working.")
        print("\n💡 To use the full system:")
        print("   1. Run: uv run python run_mcmc_server.py")
        print("   2. Run: uv run python run_d3_app.py")
        print("   3. Open: http://localhost:8000")
    else:
        print("⚠️ Some tests failed. Check the MCMC server status.")
        print("   Start server with: uv run python run_mcmc_server.py")
    
    return successes == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        sys.exit(1)