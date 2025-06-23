// Explainable AI Visualizations for MCMC Game Theory
class ExplainableAI {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.model = null;
        this.currentAnalysis = null;
        this.currentRecommendation = null;
        
        // Color schemes for explanations
        this.colors = {
            confidence: {
                high: '#2ecc71',
                medium: '#f39c12', 
                low: '#e74c3c'
            },
            importance: {
                high: '#e74c3c',
                medium: '#f39c12',
                low: '#95a5a6'
            },
            uncertainty: '#3498db',
            background: '#2c3e50',
            text: '#ecf0f1'
        };
        
        this.initializeLayout();
    }

    initializeLayout() {
        this.container.selectAll("*").remove();
        
        // Main container with dark theme
        this.mainContainer = this.container
            .append("div")
            .attr("class", "explainable-ai-container")
            .style("background-color", this.colors.background)
            .style("color", this.colors.text)
            .style("padding", "20px")
            .style("border-radius", "10px")
            .style("font-family", "'Inter', sans-serif");

        // Title section
        this.titleSection = this.mainContainer
            .append("div")
            .attr("class", "title-section")
            .style("margin-bottom", "30px");

        this.titleSection
            .append("h2")
            .style("margin", "0 0 10px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "24px")
            .style("font-weight", "600")
            .html("🧠 AI Strategic Analysis & Explanation");

        this.titleSection
            .append("p")
            .style("margin", "0")
            .style("color", "#bdc3c7")
            .style("font-size", "14px")
            .text("Bayesian MCMC model with uncertainty quantification and reasoning");

        // Create sections
        this.createRecommendationSection();
        this.createConfidenceSection();
        this.createUncertaintySection();
        this.createFactorAnalysisSection();
        this.createScenarioSection();
        this.createEvidenceSection();
    }

    createRecommendationSection() {
        this.recommendationSection = this.mainContainer
            .append("div")
            .attr("class", "recommendation-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #3498db");

        this.recommendationSection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("📋 Strategic Recommendation");

        this.recommendationContent = this.recommendationSection
            .append("div")
            .attr("class", "recommendation-content");
    }

    createConfidenceSection() {
        this.confidenceSection = this.mainContainer
            .append("div")
            .attr("class", "confidence-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #2ecc71");

        this.confidenceSection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("🎯 Confidence Analysis");

        this.confidenceContent = this.confidenceSection
            .append("div")
            .attr("class", "confidence-content");
    }

    createUncertaintySection() {
        this.uncertaintySection = this.mainContainer
            .append("div")
            .attr("class", "uncertainty-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #9b59b6");

        this.uncertaintySection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("📊 Uncertainty Quantification");

        this.uncertaintyContent = this.uncertaintySection
            .append("div")
            .attr("class", "uncertainty-content");
    }

    createFactorAnalysisSection() {
        this.factorSection = this.mainContainer
            .append("div")
            .attr("class", "factor-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #e67e22");

        this.factorSection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("🔍 Factor Importance Analysis");

        this.factorContent = this.factorSection
            .append("div")
            .attr("class", "factor-content");
    }

    createScenarioSection() {
        this.scenarioSection = this.mainContainer
            .append("div")
            .attr("class", "scenario-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #f39c12");

        this.scenarioSection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("🎭 Scenario Robustness");

        this.scenarioContent = this.scenarioSection
            .append("div")
            .attr("class", "scenario-content");
    }

    createEvidenceSection() {
        this.evidenceSection = this.mainContainer
            .append("div")
            .attr("class", "evidence-section")
            .style("margin-bottom", "25px")
            .style("padding", "20px")
            .style("background-color", "#34495e")
            .style("border-radius", "8px")
            .style("border-left", "4px solid #e74c3c");

        this.evidenceSection
            .append("h3")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .style("font-size", "18px")
            .html("🔬 Evidence Integration");

        this.evidenceContent = this.evidenceSection
            .append("div")
            .attr("class", "evidence-content");

        // Add evidence input controls
        this.createEvidenceInputs();
    }

    setModel(model) {
        this.model = model;
    }

    async updateAnalysis() {
        if (!this.model) return;

        try {
            // Get MCMC analysis and recommendation
            this.currentAnalysis = await this.model.getStrategyAnalysisWithUncertainty();
            this.currentRecommendation = await this.model.getRecommendationWithReasoning();
            
            // Update all sections
            this.updateRecommendationDisplay();
            this.updateConfidenceDisplay();
            this.updateUncertaintyDisplay();
            this.updateFactorAnalysis();
            this.updateScenarioAnalysis();
            this.updateEvidenceDisplay();
            
        } catch (error) {
            console.error('Error updating explainable AI analysis:', error);
            this.showError(error.message);
        }
    }

    updateRecommendationDisplay() {
        const rec = this.currentRecommendation;
        if (!rec) return;

        this.recommendationContent.selectAll("*").remove();

        // Main recommendation
        const mainRec = this.recommendationContent
            .append("div")
            .style("margin-bottom", "15px");

        mainRec.append("div")
            .style("font-size", "20px")
            .style("font-weight", "600")
            .style("color", "#3498db")
            .style("margin-bottom", "10px")
            .html(`✅ ${this.model.strategies[rec.strategy]?.name || rec.strategy}`);

        // Reasoning
        mainRec.append("div")
            .style("font-size", "14px")
            .style("line-height", "1.5")
            .style("color", "#bdc3c7")
            .style("background-color", "#2c3e50")
            .style("padding", "12px")
            .style("border-radius", "6px")
            .style("border-left", "3px solid #3498db")
            .text(`💡 ${rec.reasoning}`);

        // Utility and confidence metrics
        const metricsDiv = this.recommendationContent
            .append("div")
            .style("display", "grid")
            .style("grid-template-columns", "1fr 1fr 1fr")
            .style("gap", "15px")
            .style("margin-top", "15px");

        // Expected utility
        this.createMetricCard(metricsDiv, "Expected Utility", rec.utility.toFixed(3), this.colors.confidence.high, "📈");
        
        // Confidence level
        const confidenceColor = rec.confidence > 0.8 ? this.colors.confidence.high : 
                               rec.confidence > 0.6 ? this.colors.confidence.medium : this.colors.confidence.low;
        this.createMetricCard(metricsDiv, "Confidence", `${(rec.confidence * 100).toFixed(1)}%`, confidenceColor, "🎯");
        
        // War risk
        const warRisk = rec.risks?.escalation_risk || 0;
        const riskColor = warRisk < 0.2 ? this.colors.confidence.high :
                         warRisk < 0.4 ? this.colors.confidence.medium : this.colors.confidence.low;
        this.createMetricCard(metricsDiv, "War Risk", `${(warRisk * 100).toFixed(1)}%`, riskColor, "⚠️");
    }

    updateConfidenceDisplay() {
        const rec = this.currentRecommendation;
        if (!rec) return;

        this.confidenceContent.selectAll("*").remove();

        // Confidence breakdown
        const confidence = rec.confidence;
        const confidenceLevel = confidence > 0.8 ? "High" : confidence > 0.6 ? "Medium" : "Low";
        const confidenceColor = confidence > 0.8 ? this.colors.confidence.high : 
                               confidence > 0.6 ? this.colors.confidence.medium : this.colors.confidence.low;

        // Main confidence display
        const confidenceBar = this.confidenceContent
            .append("div")
            .style("margin-bottom", "20px");

        confidenceBar.append("div")
            .style("display", "flex")
            .style("justify-content", "space-between")
            .style("margin-bottom", "8px")
            .style("font-size", "14px")
            .html(`<span>Model Confidence: ${confidenceLevel}</span><span>${(confidence * 100).toFixed(1)}%</span>`);

        // Progress bar
        const progressContainer = confidenceBar.append("div")
            .style("background-color", "#2c3e50")
            .style("border-radius", "10px")
            .style("height", "20px")
            .style("overflow", "hidden");

        progressContainer.append("div")
            .style("background-color", confidenceColor)
            .style("height", "100%")
            .style("width", `${confidence * 100}%`)
            .style("transition", "width 0.5s ease");

        // Uncertainty bounds
        if (rec.utility_bounds) {
            const boundsDiv = this.confidenceContent
                .append("div")
                .style("margin-top", "15px")
                .style("padding", "12px")
                .style("background-color", "#2c3e50")
                .style("border-radius", "6px");

            boundsDiv.append("div")
                .style("font-size", "12px")
                .style("color", "#bdc3c7")
                .style("margin-bottom", "5px")
                .text("95% Confidence Interval:");

            boundsDiv.append("div")
                .style("font-size", "14px")
                .style("color", "#ecf0f1")
                .text(`[${rec.utility_bounds[0].toFixed(3)}, ${rec.utility_bounds[1].toFixed(3)}]`);
        }

        // Alternative strategies
        if (rec.alternatives && rec.alternatives.length > 0) {
            const altDiv = this.confidenceContent
                .append("div")
                .style("margin-top", "15px");

            altDiv.append("div")
                .style("font-size", "14px")
                .style("color", "#bdc3c7")
                .style("margin-bottom", "10px")
                .text("Alternative Strategies:");

            rec.alternatives.forEach(([strategy, utility]) => {
                const altStrategy = altDiv.append("div")
                    .style("display", "flex")
                    .style("justify-content", "space-between")
                    .style("padding", "8px")
                    .style("margin-bottom", "5px")
                    .style("background-color", "#2c3e50")
                    .style("border-radius", "4px")
                    .style("font-size", "13px");

                altStrategy.append("span")
                    .text(this.model.strategies[strategy]?.name || strategy);

                altStrategy.append("span")
                    .style("color", "#95a5a6")
                    .text(utility.toFixed(3));
            });
        }
    }

    updateUncertaintyDisplay() {
        const analysis = this.currentAnalysis;
        if (!analysis) return;

        this.uncertaintyContent.selectAll("*").remove();

        // Create uncertainty visualization
        const uncertaintyData = this.model.getUncertaintyVisualizationData(analysis);
        if (!uncertaintyData) return;

        // SVG for uncertainty plot
        const margin = { top: 20, right: 20, bottom: 60, left: 120 };
        const width = 500 - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        const svg = this.uncertaintyContent
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background-color", "#2c3e50");

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Scales
        const yScale = d3.scaleBand()
            .domain(uncertaintyData.map(d => d.name))
            .range([0, height])
            .padding(0.2);

        const xScale = d3.scaleLinear()
            .domain([0, d3.max(uncertaintyData, d => d.utility_mean + d.utility_error[1]) * 1.1])
            .range([0, width]);

        // Draw error bars
        uncertaintyData.forEach(d => {
            const yPos = yScale(d.name) + yScale.bandwidth() / 2;
            
            // Error bar line
            g.append("line")
                .attr("x1", xScale(d.utility_mean - d.utility_error[0]))
                .attr("x2", xScale(d.utility_mean + d.utility_error[1]))
                .attr("y1", yPos)
                .attr("y2", yPos)
                .style("stroke", "#95a5a6")
                .style("stroke-width", 2);

            // Error bar caps
            [d.utility_mean - d.utility_error[0], d.utility_mean + d.utility_error[1]].forEach(x => {
                g.append("line")
                    .attr("x1", xScale(x))
                    .attr("x2", xScale(x))
                    .attr("y1", yPos - 5)
                    .attr("y2", yPos + 5)
                    .style("stroke", "#95a5a6")
                    .style("stroke-width", 2);
            });

            // Mean point
            g.append("circle")
                .attr("cx", xScale(d.utility_mean))
                .attr("cy", yPos)
                .attr("r", 6)
                .style("fill", d.color)
                .style("stroke", "#ecf0f1")
                .style("stroke-width", 2);
        });

        // Y axis
        g.append("g")
            .call(d3.axisLeft(yScale))
            .selectAll("text")
            .style("fill", "#ecf0f1")
            .style("font-size", "12px");

        // X axis
        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(5))
            .selectAll("text")
            .style("fill", "#ecf0f1")
            .style("font-size", "12px");

        // Axis styling
        g.selectAll(".domain, .tick line")
            .style("stroke", "#95a5a6");

        // Title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("fill", "#ecf0f1")
            .style("font-size", "14px")
            .style("font-weight", "600")
            .text("Expected Utility with Uncertainty Bounds");
    }

    updateFactorAnalysis() {
        this.factorContent.selectAll("*").remove();

        // Get current game state for factor analysis
        const gameState = this.model.getGameState();
        
        // Calculate factor importance (simplified)
        const factors = [
            { name: "Nuclear Progress", value: gameState.nuclear_progress, impact: "high", description: "Directly affects breakout timeline" },
            { name: "Economic Stress", value: gameState.economic_stress, impact: "high", description: "Increases regime desperation" },
            { name: "Regime Cohesion", value: 1 - gameState.regime_cohesion, impact: "medium", description: "Lower cohesion = higher unpredictability" },
            { name: "External Support", value: gameState.external_support, impact: "medium", description: "Chinese/Russian backing affects options" },
            { name: "Proxy Support", value: 1 - gameState.proxy_support, impact: "low", description: "Proxy degradation forces nuclear focus" },
            { name: "Oil Price Impact", value: Math.max(0, (gameState.oil_price - 80) / 70), impact: "low", description: "Higher prices increase global caution" }
        ];

        factors.forEach(factor => {
            const factorDiv = this.factorContent
                .append("div")
                .style("margin-bottom", "15px")
                .style("padding", "12px")
                .style("background-color", "#2c3e50")
                .style("border-radius", "6px")
                .style("border-left", `4px solid ${this.colors.importance[factor.impact]}`);

            // Factor header
            const header = factorDiv.append("div")
                .style("display", "flex")
                .style("justify-content", "space-between")
                .style("align-items", "center")
                .style("margin-bottom", "8px");

            header.append("span")
                .style("font-weight", "600")
                .style("color", "#ecf0f1")
                .text(factor.name);

            const importanceColor = this.colors.importance[factor.impact];
            header.append("span")
                .style("background-color", importanceColor)
                .style("color", "#ffffff")
                .style("padding", "2px 8px")
                .style("border-radius", "12px")
                .style("font-size", "11px")
                .style("text-transform", "uppercase")
                .text(`${factor.impact} impact`);

            // Factor value bar
            const valueBar = factorDiv.append("div")
                .style("margin-bottom", "5px");

            const barContainer = valueBar.append("div")
                .style("background-color", "#34495e")
                .style("border-radius", "10px")
                .style("height", "8px")
                .style("overflow", "hidden");

            barContainer.append("div")
                .style("background-color", importanceColor)
                .style("height", "100%")
                .style("width", `${factor.value * 100}%`)
                .style("transition", "width 0.5s ease");

            // Description
            factorDiv.append("div")
                .style("font-size", "12px")
                .style("color", "#bdc3c7")
                .style("font-style", "italic")
                .text(factor.description);
        });
    }

    updateScenarioAnalysis() {
        this.scenarioContent.selectAll("*").remove();

        // Define key scenarios to test
        const scenarios = {
            "Regime Collapse": {
                regime_cohesion: 0.1,
                economic_stress: 0.95,
                nuclear_progress: 0.85
            },
            "Chinese Intervention": {
                external_support: 0.9,
                economic_stress: 0.4,
                oil_price: 120
            },
            "Nuclear Sprint": {
                nuclear_progress: 0.95,
                regime_cohesion: 0.2,
                proxy_support: 0.05
            }
        };

        // Test each scenario (using fallback for demo)
        Object.keys(scenarios).forEach(scenarioName => {
            const scenario = scenarios[scenarioName];
            
            const scenarioDiv = this.scenarioContent
                .append("div")
                .style("margin-bottom", "15px")
                .style("padding", "15px")
                .style("background-color", "#2c3e50")
                .style("border-radius", "6px")
                .style("border", "1px solid #34495e");

            scenarioDiv.append("div")
                .style("font-weight", "600")
                .style("color", "#ecf0f1")
                .style("margin-bottom", "10px")
                .text(`📋 ${scenarioName} Scenario`);

            // Scenario parameters
            const paramsDiv = scenarioDiv.append("div")
                .style("font-size", "12px")
                .style("color", "#bdc3c7")
                .style("margin-bottom", "10px");

            Object.keys(scenario).forEach(param => {
                paramsDiv.append("span")
                    .style("margin-right", "15px")
                    .text(`${param}: ${scenario[param]}`);
            });

            // Simulate recommendation for this scenario
            const recommendation = this.simulateScenarioRecommendation(scenarioName);
            
            scenarioDiv.append("div")
                .style("padding", "8px")
                .style("background-color", "#34495e")
                .style("border-radius", "4px")
                .style("color", "#ecf0f1")
                .text(`→ ${recommendation.strategy} (${(recommendation.confidence * 100).toFixed(0)}% confidence)`);
        });
    }

    updateEvidenceDisplay() {
        // Update evidence history display
        const history = this.model.getBeliefChangeHistory();
        
        const evidenceHistory = this.evidenceContent.select(".evidence-history");
        if (evidenceHistory.empty()) return;

        evidenceHistory.selectAll(".evidence-item").remove();

        history.slice(-5).forEach(update => { // Show last 5 updates
            const item = evidenceHistory.append("div")
                .attr("class", "evidence-item")
                .style("padding", "8px")
                .style("margin-bottom", "5px")
                .style("background-color", "#2c3e50")
                .style("border-radius", "4px")
                .style("border-left", `3px solid ${update.significant ? '#e74c3c' : '#95a5a6'}`);

            item.append("div")
                .style("font-size", "12px")
                .style("color", "#bdc3c7")
                .text(new Date(update.timestamp).toLocaleTimeString());

            item.append("div")
                .style("font-size", "13px")
                .style("color", "#ecf0f1")
                .text(`Belief change: ${update.kl_divergence.toFixed(3)} ${update.significant ? '(Significant)' : ''}`);
        });
    }

    createEvidenceInputs() {
        const inputsDiv = this.evidenceContent
            .append("div")
            .style("margin-bottom", "20px")
            .style("padding", "15px")
            .style("background-color", "#2c3e50")
            .style("border-radius", "6px");

        inputsDiv.append("h4")
            .style("margin", "0 0 15px 0")
            .style("color", "#ecf0f1")
            .text("Add New Evidence");

        // Evidence type selector
        const evidenceTypes = [
            { value: "regime_stability_signal", label: "Regime Stability Intelligence" },
            { value: "nuclear_activity_detected", label: "Nuclear Facility Activity" },
            { value: "proxy_attacks_increased", label: "Proxy Attack Frequency" },
            { value: "chinese_naval_movement", label: "Chinese Military Movement" },
            { value: "economic_indicators", label: "Economic Intelligence" }
        ];

        const selectDiv = inputsDiv.append("div")
            .style("margin-bottom", "10px");

        selectDiv.append("label")
            .style("display", "block")
            .style("margin-bottom", "5px")
            .style("color", "#bdc3c7")
            .style("font-size", "13px")
            .text("Evidence Type:");

        const select = selectDiv.append("select")
            .attr("id", "evidence-type")
            .style("width", "100%")
            .style("padding", "8px")
            .style("background-color", "#34495e")
            .style("color", "#ecf0f1")
            .style("border", "1px solid #95a5a6")
            .style("border-radius", "4px");

        evidenceTypes.forEach(type => {
            select.append("option")
                .attr("value", type.value)
                .text(type.label);
        });

        // Value input
        const valueDiv = inputsDiv.append("div")
            .style("margin-bottom", "10px");

        valueDiv.append("label")
            .style("display", "block")
            .style("margin-bottom", "5px")
            .style("color", "#bdc3c7")
            .style("font-size", "13px")
            .text("Evidence Value (0-1):");

        valueDiv.append("input")
            .attr("type", "range")
            .attr("id", "evidence-value")
            .attr("min", "0")
            .attr("max", "1")
            .attr("step", "0.01")
            .attr("value", "0.5")
            .style("width", "100%");

        // Add evidence button
        const addButton = inputsDiv.append("button")
            .style("background-color", "#3498db")
            .style("color", "#ffffff")
            .style("border", "none")
            .style("padding", "10px 20px")
            .style("border-radius", "4px")
            .style("cursor", "pointer")
            .style("font-size", "14px")
            .text("Add Evidence")
            .on("click", () => this.addEvidence());

        // Evidence history display
        this.evidenceContent.append("div")
            .attr("class", "evidence-history")
            .style("margin-top", "20px");
    }

    async addEvidence() {
        const evidenceType = document.getElementById("evidence-type").value;
        const evidenceValue = parseFloat(document.getElementById("evidence-value").value);
        
        if (!this.model) return;

        try {
            const evidence = {};
            evidence[evidenceType] = evidenceValue;
            
            const update = await this.model.updateBeliefs(evidence, 0.8);
            
            // Update the analysis with new evidence
            await this.updateAnalysis();
            
            // Show notification
            this.showNotification(
                update.significant_change ? 
                "Significant belief update detected!" : 
                "Evidence integrated successfully"
            );
            
        } catch (error) {
            console.error('Error adding evidence:', error);
            this.showError('Failed to integrate evidence');
        }
    }

    createMetricCard(container, title, value, color, icon) {
        const card = container.append("div")
            .style("background-color", "#2c3e50")
            .style("padding", "15px")
            .style("border-radius", "6px")
            .style("border-top", `3px solid ${color}`)
            .style("text-align", "center");

        card.append("div")
            .style("font-size", "24px")
            .style("margin-bottom", "5px")
            .text(icon);

        card.append("div")
            .style("font-size", "20px")
            .style("font-weight", "600")
            .style("color", color)
            .style("margin-bottom", "5px")
            .text(value);

        card.append("div")
            .style("font-size", "12px")
            .style("color", "#bdc3c7")
            .text(title);
    }

    simulateScenarioRecommendation(scenarioName) {
        // Simulate different recommendations for different scenarios
        const recommendations = {
            "Regime Collapse": { strategy: "Deterrence + Diplomacy", confidence: 0.65 },
            "Chinese Intervention": { strategy: "Deterrence + Ultimatum", confidence: 0.75 },
            "Nuclear Sprint": { strategy: "Escalation + Diplomacy", confidence: 0.80 }
        };
        
        return recommendations[scenarioName] || { strategy: "Deterrence + Diplomacy", confidence: 0.70 };
    }

    showNotification(message) {
        const notification = this.mainContainer
            .insert("div", ":first-child")
            .style("background-color", "#27ae60")
            .style("color", "#ffffff")
            .style("padding", "10px 15px")
            .style("border-radius", "4px")
            .style("margin-bottom", "15px")
            .style("font-size", "14px")
            .text(message);

        setTimeout(() => {
            notification.transition()
                .duration(500)
                .style("opacity", 0)
                .remove();
        }, 3000);
    }

    showError(message) {
        const error = this.mainContainer
            .insert("div", ":first-child")
            .style("background-color", "#e74c3c")
            .style("color", "#ffffff")
            .style("padding", "10px 15px")
            .style("border-radius", "4px")
            .style("margin-bottom", "15px")
            .style("font-size", "14px")
            .text(`Error: ${message}`);

        setTimeout(() => {
            error.transition()
                .duration(500)
                .style("opacity", 0)
                .remove();
        }, 5000);
    }
}

// Export for use
window.ExplainableAI = ExplainableAI;