// Game Theory Model - JavaScript Implementation
class GameTheoryModel {
    constructor() {
        this.outcomes = {
            DEAL: 'Iranian Capitulation & Verifiable Deal',
            LIMITED_RETALIATION: 'Limited Iranian Retaliation & De-escalation',
            FROZEN_CONFLICT: 'Protracted Low-Intensity Conflict',
            FULL_WAR: 'Full-Scale Regional War',
            NUCLEAR_BREAKOUT: 'Iranian Nuclear Breakout'
        };

        this.players = {
            USA: { name: 'USA', color: '#4e79a7' },
            IRAN: { name: 'Iran', color: '#f28e2c' },
            ISRAEL: { name: 'Israel', color: '#e15759' }
        };

        this.strategies = {
            deterrence_diplomacy: {
                name: 'Deterrence + Diplomacy',
                military: 'Halt & Deter',
                diplomatic: 'De-escalatory Off-Ramp',
                color: '#4ecdc4',
                icon: '✅'
            },
            deterrence_ultimatum: {
                name: 'Deterrence + Ultimatum',
                military: 'Halt & Deter',
                diplomatic: 'Coercive Ultimatum',
                color: '#ffe66d',
                icon: '⚠️'
            },
            escalation_diplomacy: {
                name: 'Escalation + Diplomacy',
                military: 'Expand Strikes',
                diplomatic: 'De-escalatory Off-Ramp',
                color: '#ff8b94',
                icon: '🚫'
            },
            escalation_ultimatum: {
                name: 'Escalation + Ultimatum',
                military: 'Expand Strikes',
                diplomatic: 'Coercive Ultimatum',
                color: '#ff6b6b',
                icon: '☠️'
            }
        };

        this.variables = {
            regime_cohesion: 0.4,
            economic_stress: 0.9,
            proxy_support: 0.1,
            oil_price: 97.0,
            external_support: 0.2,
            nuclear_progress: 0.7
        };

        // Player preferences (1=best, 5=worst)
        this.preferences = {
            USA: {
                DEAL: 1,
                LIMITED_RETALIATION: 2,
                NUCLEAR_BREAKOUT: 3,
                FROZEN_CONFLICT: 4,
                FULL_WAR: 5
            },
            IRAN: {
                NUCLEAR_BREAKOUT: 1,
                LIMITED_RETALIATION: 2,
                DEAL: 3,
                FROZEN_CONFLICT: 4,
                FULL_WAR: 5
            },
            ISRAEL: {
                NUCLEAR_BREAKOUT: 1,
                DEAL: 2,
                FROZEN_CONFLICT: 3,
                LIMITED_RETALIATION: 4,
                FULL_WAR: 5
            }
        };
    }

    updateVariables(newVariables) {
        this.variables = { ...this.variables, ...newVariables };
    }

    getOutcomeProbabilities(strategyKey) {
        const strategy = this.strategies[strategyKey];
        if (!strategy) return null;

        let baseProbs;
        
        // Base probabilities by strategy
        switch (strategyKey) {
            case 'deterrence_diplomacy':
                baseProbs = {
                    DEAL: 0.4,
                    LIMITED_RETALIATION: 0.35,
                    FROZEN_CONFLICT: 0.15,
                    FULL_WAR: 0.05,
                    NUCLEAR_BREAKOUT: 0.05
                };
                break;
            case 'deterrence_ultimatum':
                baseProbs = {
                    DEAL: 0.2,
                    LIMITED_RETALIATION: 0.25,
                    FROZEN_CONFLICT: 0.35,
                    FULL_WAR: 0.15,
                    NUCLEAR_BREAKOUT: 0.05
                };
                break;
            case 'escalation_diplomacy':
                baseProbs = {
                    DEAL: 0.15,
                    LIMITED_RETALIATION: 0.2,
                    FROZEN_CONFLICT: 0.25,
                    FULL_WAR: 0.3,
                    NUCLEAR_BREAKOUT: 0.1
                };
                break;
            case 'escalation_ultimatum':
                baseProbs = {
                    DEAL: 0.1,
                    LIMITED_RETALIATION: 0.1,
                    FROZEN_CONFLICT: 0.2,
                    FULL_WAR: 0.45,
                    NUCLEAR_BREAKOUT: 0.15
                };
                break;
            default:
                return null;
        }

        // Adjust for current variables
        return this.adjustProbabilitiesForVariables(baseProbs);
    }

    adjustProbabilitiesForVariables(baseProbs) {
        const adjusted = { ...baseProbs };
        const vars = this.variables;

        // Economic stress increases desperation
        const stressFactor = vars.economic_stress;
        adjusted.NUCLEAR_BREAKOUT *= (1 + stressFactor * 0.5);
        adjusted.FULL_WAR *= (1 + stressFactor * 0.3);

        // Regime weakness increases unpredictability
        const cohesionFactor = 1 - vars.regime_cohesion;
        adjusted.NUCLEAR_BREAKOUT *= (1 + cohesionFactor * 0.4);
        adjusted.FULL_WAR *= (1 + cohesionFactor * 0.2);

        // Oil prices affect pressure for resolution
        if (vars.oil_price > 100) {
            const oilFactor = (vars.oil_price - 100) / 100;
            adjusted.DEAL *= (1 + oilFactor * 0.3);
            adjusted.FROZEN_CONFLICT *= (1 - oilFactor * 0.2);
        }

        // Proxy weakness affects Iranian calculations
        const proxyWeakness = 1 - vars.proxy_support;
        adjusted.NUCLEAR_BREAKOUT *= (1 + proxyWeakness * 0.3);
        adjusted.DEAL *= (1 + proxyWeakness * 0.2);

        // Normalize to sum to 1.0
        const total = Object.values(adjusted).reduce((sum, val) => sum + val, 0);
        Object.keys(adjusted).forEach(key => {
            adjusted[key] /= total;
        });

        return adjusted;
    }

    getEscalationProbabilities() {
        const strategies = Object.keys(this.strategies);
        const escalationData = [];

        strategies.forEach(strategy => {
            const probs = this.getOutcomeProbabilities(strategy);
            const escalationRisk = probs.FULL_WAR + probs.NUCLEAR_BREAKOUT;
            
            escalationData.push({
                strategy: strategy,
                strategyName: this.strategies[strategy].name,
                military: this.strategies[strategy].military,
                diplomatic: this.strategies[strategy].diplomatic,
                escalationProbability: escalationRisk,
                warRisk: probs.FULL_WAR,
                nuclearRisk: probs.NUCLEAR_BREAKOUT,
                successProb: probs.DEAL + probs.LIMITED_RETALIATION,
                color: this.strategies[strategy].color,
                icon: this.strategies[strategy].icon
            });
        });

        return escalationData;
    }

    getStrategyRankings() {
        const strategies = Object.keys(this.strategies);
        const rankings = [];

        strategies.forEach(strategy => {
            const probs = this.getOutcomeProbabilities(strategy);
            
            // Calculate utilities (converted to positive scale)
            const utilities = {};
            Object.keys(this.players).forEach(player => {
                let utility = 0;
                Object.keys(probs).forEach(outcome => {
                    const preference = this.preferences[player][outcome];
                    utility += probs[outcome] * (6 - preference);
                });
                utilities[player] = utility;
            });

            const warRisk = probs.FULL_WAR + probs.NUCLEAR_BREAKOUT;
            const successProb = probs.DEAL + probs.LIMITED_RETALIATION;

            rankings.push({
                strategy: strategy,
                strategyName: this.strategies[strategy].name,
                military: this.strategies[strategy].military,
                diplomatic: this.strategies[strategy].diplomatic,
                utilities: utilities,
                warRisk: warRisk,
                successProb: successProb,
                uncertainty: this.calculateEntropy(Object.values(probs)),
                color: this.strategies[strategy].color,
                icon: this.strategies[strategy].icon,
                probabilities: probs
            });
        });

        // Sort by USA utility (descending) then by war risk (ascending)
        rankings.sort((a, b) => {
            if (Math.abs(a.utilities.USA - b.utilities.USA) < 0.1) {
                return a.warRisk - b.warRisk;
            }
            return b.utilities.USA - a.utilities.USA;
        });

        return rankings;
    }

    calculateEntropy(probabilities) {
        return -probabilities.reduce((entropy, p) => {
            return p > 0 ? entropy + p * Math.log2(p) : entropy;
        }, 0);
    }

    getDefconLevel() {
        const vars = this.variables;
        const baseDefcon = 5.0;
        
        const escalationFactors = {
            economic_stress: vars.economic_stress * 0.8,
            regime_weakness: (1 - vars.regime_cohesion) * 0.6,
            proxy_collapse: (1 - vars.proxy_support) * 0.4,
            nuclear_progress: vars.nuclear_progress * 0.7,
            oil_crisis: Math.max(0, (vars.oil_price - 90) / 50) * 0.5
        };

        const totalEscalation = Object.values(escalationFactors).reduce((sum, val) => sum + val, 0);
        const currentDefcon = Math.max(1.0, baseDefcon - totalEscalation);

        return {
            level: currentDefcon,
            factors: escalationFactors,
            totalEscalation: totalEscalation,
            severity: Math.min(1.0, totalEscalation / 2.0)
        };
    }

    getMarketIndicators() {
        const vars = this.variables;
        
        // VIX calculation
        const baseVix = 20;
        const volatilityFactors = [
            vars.economic_stress * 25,
            (1 - vars.regime_cohesion) * 20,
            (vars.oil_price - 80) / 4,
            vars.nuclear_progress * 15
        ];
        const vixEstimate = Math.min(100, baseVix + volatilityFactors.reduce((sum, val) => sum + val, 0));

        // Gold price
        const baseGold = 2000;
        const safeHavenDemand = (vars.economic_stress + (1 - vars.regime_cohesion)) * 400;
        const goldEstimate = baseGold + safeHavenDemand;

        // Iranian Rial
        const baseRialRate = 42000;
        const sanctionsMultiplier = 1 + (vars.economic_stress * 12);
        const rialEstimate = baseRialRate * sanctionsMultiplier;

        return {
            vix: vixEstimate,
            gold: goldEstimate,
            rial: rialEstimate,
            oil: vars.oil_price,
            stressIndex: Math.min(1.0, vixEstimate / 100)
        };
    }

    // Sensitivity analysis for a specific variable
    sensitivityAnalysis(strategy, variable, steps = 20) {
        const originalValue = this.variables[variable];
        const results = [];
        
        let minVal = 0;
        let maxVal = 1;
        if (variable === 'oil_price') {
            minVal = 50;
            maxVal = 150;
        }

        for (let i = 0; i < steps; i++) {
            const value = minVal + (maxVal - minVal) * (i / (steps - 1));
            this.variables[variable] = value;
            
            const probs = this.getOutcomeProbabilities(strategy);
            const warRisk = probs.FULL_WAR + probs.NUCLEAR_BREAKOUT;
            const successProb = probs.DEAL + probs.LIMITED_RETALIATION;
            
            // Calculate USA utility
            let usaUtility = 0;
            Object.keys(probs).forEach(outcome => {
                const preference = this.preferences.USA[outcome];
                usaUtility += probs[outcome] * (6 - preference);
            });

            results.push({
                [variable]: value,
                warRisk: warRisk,
                successProb: successProb,
                usaUtility: usaUtility,
                probabilities: { ...probs }
            });
        }

        // Restore original value
        this.variables[variable] = originalValue;
        
        return results;
    }
}

// Export for use in other modules
window.GameTheoryModel = GameTheoryModel;