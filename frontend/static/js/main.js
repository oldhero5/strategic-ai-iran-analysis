// Main JavaScript file for Game Theory Iran Model

class GameTheoryApp {
    constructor() {
        // Use the MCMC model instead of basic model
        this.model = new MCMCGameTheoryModel();
        this.visualizations = new GameTheoryVisualizations();
        this.creativeViz = new CreativeVisualizations();
        this.explainableAI = new ExplainableAI('explainable-ai');
        this.useCreativeMode = true; // Toggle for creative vs clean mode
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.updateAllVisualizations();
        this.setupExportModal();
        this.setupExplainableAI();
    }

    setupEventListeners() {
        // Variable controls
        const controls = [
            'regime-cohesion',
            'economic-stress', 
            'proxy-support',
            'oil-price',
            'external-support',
            'nuclear-progress'
        ];

        controls.forEach(controlId => {
            const slider = document.getElementById(controlId);
            const valueDisplay = document.getElementById(controlId + '-value');
            
            if (slider && valueDisplay) {
                slider.addEventListener('input', (e) => {
                    const value = parseFloat(e.target.value);
                    
                    // Update display
                    if (controlId === 'oil-price') {
                        valueDisplay.textContent = `$${value}`;
                    } else {
                        valueDisplay.textContent = value.toFixed(2);
                    }
                    
                    // Update model
                    const variableName = controlId.replace('-', '_');
                    this.model.updateVariables({ [variableName]: value });
                    
                    // Debounce updates
                    clearTimeout(this.updateTimeout);
                    this.updateTimeout = setTimeout(() => {
                        this.updateAllVisualizations();
                        this.updateExplainableAI();
                    }, 300);
                });
            }
        });

        // Remove old sensitivity analysis controls since we're replacing with explainable AI

        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.showExportModal();
            });
        }
    }

    showLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('active');
        }
    }

    hideLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('active');
        }
    }

    updateAllVisualizations() {
        this.showLoading();
        
        // Add small delay for smooth loading animation
        setTimeout(() => {
            try {
                // Update the model in visualizations
                this.visualizations.model = this.model;
                this.creativeViz.model = this.model;
                
                // Create visualizations based on mode
                if (this.useCreativeMode) {
                    // Use creative visualizations
                    this.creativeViz.createCreativeEscalationHeatmap('escalation-heatmap');
                    this.creativeViz.createCreativePayoffMatrix('payoff-matrix');
                    this.creativeViz.createCreativeStrategyComparison('strategy-comparison');
                } else {
                    // Use clean visualizations
                    this.visualizations.createEscalationHeatmap('escalation-heatmap');
                    this.visualizations.createPayoffMatrix('payoff-matrix');
                    this.visualizations.createStrategyComparison('strategy-comparison');
                }
                
                this.createOutcomeProbabilities();
                this.createEscalationLadder();
                this.createMarketIndicators();
                
                // Add fade-in animation to containers
                document.querySelectorAll('.viz-container').forEach((container, index) => {
                    container.style.animation = `fadeIn 0.5s ease-in ${index * 0.1}s both`;
                });
                
            } catch (error) {
                console.error('Error updating visualizations:', error);
            } finally {
                this.hideLoading();
            }
        }, 200);
    }

    createOutcomeProbabilities() {
        const container = d3.select('#outcome-probabilities .chart-container');
        container.selectAll("*").remove();

        const margin = { top: 60, right: 200, bottom: 80, left: 80 };
        const width = 800 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Clean data preparation
        const strategies = Object.keys(this.model.strategies);
        const outcomes = Object.keys(this.model.outcomes);
        
        const data = [];
        strategies.forEach(strategy => {
            const probs = this.model.getOutcomeProbabilities(strategy);
            const strategyInfo = this.model.strategies[strategy];
            
            outcomes.forEach(outcome => {
                data.push({
                    strategy: strategy,
                    strategyName: strategyInfo.name,
                    outcome: outcome,
                    outcomeName: this.model.outcomes[outcome],
                    probability: probs[outcome],
                    color: strategyInfo.color,
                    icon: strategyInfo.icon
                });
            });
        });

        // Clean scales
        const xScale = d3.scaleBand()
            .domain(strategies)
            .range([0, width])
            .padding(0.15);

        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([height, 0]);

        // Clean color scheme
        const colorScale = d3.scaleOrdinal()
            .domain(outcomes)
            .range(['#2ed573', '#7bed9f', '#70a1ff', '#ff6348', '#ff3838']);

        // Stack data cleanly
        const stack = d3.stack()
            .keys(outcomes)
            .value((d, key) => d.find(item => item.outcome === key)?.probability || 0);

        const series = stack(strategies.map(strategy => 
            data.filter(d => d.strategy === strategy)
        ));

        // Clean tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("background", "rgba(0, 0, 0, 0.9)")
            .style("color", "white")
            .style("padding", "12px")
            .style("border-radius", "6px")
            .style("font-size", "14px")
            .style("border", "1px solid #007acc");

        // Draw clean stacked bars
        const groups = g.selectAll(".strategy-group")
            .data(series)
            .enter().append("g")
            .attr("class", "strategy-group")
            .style("fill", (d, i) => colorScale(outcomes[i]));

        groups.selectAll("rect")
            .data(d => d)
            .enter().append("rect")
            .attr("x", (d, i) => xScale(strategies[i]))
            .attr("y", d => yScale(d[1]))
            .attr("height", d => yScale(d[0]) - yScale(d[1]))
            .attr("width", xScale.bandwidth())
            .attr("rx", 4)
            .style("stroke", "#ffffff")
            .style("stroke-width", 1)
            .on("mouseover", function(event, d) {
                const strategyIndex = d3.select(this.parentNode).datum().indexOf(d);
                const outcomeIndex = series.indexOf(d3.select(this.parentNode).datum());
                const outcome = outcomes[outcomeIndex];
                const strategy = strategies[strategyIndex];
                const probability = d[1] - d[0];

                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("opacity", 0.8)
                    .style("stroke", "#007acc")
                    .style("stroke-width", 3);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 4px;">${gameTheoryApp.model.strategies[strategy]?.name}</div>
                    <div style="margin: 2px 0;">${gameTheoryApp.model.outcomes[outcome]}</div>
                    <div style="margin: 2px 0;"><strong>Probability:</strong> ${(probability * 100).toFixed(1)}%</div>
                `)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("opacity", 1)
                    .style("stroke", "#ffffff")
                    .style("stroke-width", 1);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 0);
            });

        // Clean axes
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .style("text-anchor", "end")
            .attr("dx", "-.8em")
            .attr("dy", ".15em")
            .attr("transform", "rotate(-45)")
            .style("fill", "#cccccc")
            .style("font-size", "12px")
            .text((d, i) => this.model.strategies[d]?.name || d);

        g.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(yScale).tickFormat(d3.format(".0%")));

        // Style axes cleanly
        g.selectAll(".x-axis text, .y-axis text")
            .style("fill", "#cccccc")
            .style("font-size", "12px");

        g.selectAll(".x-axis path, .x-axis line, .y-axis path, .y-axis line")
            .style("stroke", "#666666");

        // Clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "18px")
            .style("font-weight", "700")
            .text("Outcome Probabilities by Strategy");

        // Clean legend
        const legend = svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${width + margin.left + 20}, ${margin.top})`);

        const legendItems = legend.selectAll(".legend-item")
            .data(outcomes)
            .enter().append("g")
            .attr("class", "legend-item")
            .attr("transform", (d, i) => `translate(0, ${i * 30})`);

        legendItems.append("rect")
            .attr("width", 16)
            .attr("height", 16)
            .attr("rx", 3)
            .style("fill", d => colorScale(d));

        legendItems.append("text")
            .attr("x", 22)
            .attr("y", 8)
            .attr("dy", "0.35em")
            .style("fill", "#cccccc")
            .style("font-size", "12px")
            .style("font-weight", "500")
            .text(d => d.replace('_', ' '));
    }

    createEscalationLadder() {
        const container = d3.select('#escalation-ladder .chart-container');
        container.selectAll("*").remove();

        const margin = { top: 40, right: 60, bottom: 40, left: 120 };
        const width = 400 - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const defconData = this.model.getDefconLevel();
        const currentLevel = defconData.level;

        const levels = [
            { level: 5, name: 'DEFCON 5', desc: 'Normal', color: '#2ed573' },
            { level: 4, name: 'DEFCON 4', desc: 'Increased Watch', color: '#7bed9f' },
            { level: 3, name: 'DEFCON 3', desc: 'Round House', color: '#ffa502' },
            { level: 2, name: 'DEFCON 2', desc: 'Fast Pace', color: '#ff6348' },
            { level: 1, name: 'DEFCON 1', desc: 'Exercise Term', color: '#ff3838' }
        ];

        const yScale = d3.scaleBand()
            .domain(levels.map(d => d.level))
            .range([0, height])
            .padding(0.1);

        // Draw clean DEFCON levels
        const levelGroups = g.selectAll('.defcon-level')
            .data(levels)
            .enter().append('g')
            .attr('class', 'defcon-level')
            .attr('transform', d => `translate(0, ${yScale(d.level)})`);

        levelGroups.append('rect')
            .attr('width', width)
            .attr('height', yScale.bandwidth())
            .attr('rx', 6)
            .style('fill', d => d.color)
            .style('opacity', d => Math.abs(d.level - currentLevel) < 0.5 ? 1.0 : 0.4)
            .style('stroke', d => Math.abs(d.level - currentLevel) < 0.5 ? '#007acc' : '#ffffff')
            .style('stroke-width', d => Math.abs(d.level - currentLevel) < 0.5 ? 3 : 1);

        levelGroups.append('text')
            .attr('x', 15)
            .attr('y', yScale.bandwidth() / 2)
            .attr('dominant-baseline', 'central')
            .style('fill', '#ffffff')
            .style('font-size', '13px')
            .style('font-weight', 'bold')
            .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)')
            .text(d => d.name);

        levelGroups.append('text')
            .attr('x', width - 15)
            .attr('y', yScale.bandwidth() / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'central')
            .style('fill', '#ffffff')
            .style('font-size', '11px')
            .style('font-weight', '500')
            .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.8)')
            .text(d => d.desc);

        // Clean current level indicator
        g.append('line')
            .attr('x1', -10)
            .attr('x2', width + 10)
            .attr('y1', yScale(Math.round(currentLevel)) + yScale.bandwidth() / 2)
            .attr('y2', yScale(Math.round(currentLevel)) + yScale.bandwidth() / 2)
            .style('stroke', '#007acc')
            .style('stroke-width', 3)
            .style('stroke-dasharray', '5,5');

        // Clean current level text
        g.append('text')
            .attr('x', width + 20)
            .attr('y', yScale(Math.round(currentLevel)) + yScale.bandwidth() / 2)
            .attr('dominant-baseline', 'central')
            .style('fill', '#007acc')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text(`Current: ${currentLevel.toFixed(1)}`);

        // Clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "16px")
            .style("font-weight", "700")
            .text("Current Threat Level");
    }

    createMarketIndicators() {
        const container = d3.select('#market-indicators .chart-container');
        container.selectAll("*").remove();

        const indicators = this.model.getMarketIndicators();
        
        const margin = { top: 40, right: 20, bottom: 20, left: 20 };
        const width = 400 - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Clean market data
        const marketData = [
            { name: 'VIX', value: indicators.vix, max: 100, unit: '', color: '#ff6348', label: 'Fear Index' },
            { name: 'Oil', value: indicators.oil, max: 150, unit: '$', color: '#ffa502', label: 'Brent Crude' },
            { name: 'Gold', value: indicators.gold, max: 3000, unit: '$', color: '#2ed573', label: 'Safe Haven' },
            { name: 'Rial', value: indicators.rial / 1000, max: 600, unit: 'k', color: '#70a1ff', label: 'vs USD' }
        ];

        // Clean layout
        const cols = 2;
        const cellWidth = width / cols;
        const cellHeight = height / 2;

        marketData.forEach((data, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = col * cellWidth + cellWidth / 2;
            const y = row * cellHeight + cellHeight / 2;

            this.createCleanGauge(g, x, y, Math.min(cellWidth, cellHeight) * 0.3, data);
        });

        // Clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "16px")
            .style("font-weight", "700")
            .text("Market Impact");
    }

    createCleanGauge(container, x, y, radius, data) {
        const gaugeGroup = container.append('g')
            .attr('transform', `translate(${x}, ${y})`);

        // Clean background circle
        gaugeGroup.append('circle')
            .attr('r', radius)
            .style('fill', 'none')
            .style('stroke', '#333333')
            .style('stroke-width', 8);

        // Clean progress arc
        const progress = Math.min(data.value / data.max, 1);
        const circumference = 2 * Math.PI * radius;
        const offset = circumference * (1 - progress);

        gaugeGroup.append('circle')
            .attr('r', radius)
            .style('fill', 'none')
            .style('stroke', data.color)
            .style('stroke-width', 8)
            .style('stroke-linecap', 'round')
            .style('stroke-dasharray', circumference)
            .style('stroke-dashoffset', offset)
            .style('transform', 'rotate(-90deg)')
            .style('transform-origin', 'center');

        // Clean center value
        gaugeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('y', -5)
            .style('fill', '#ffffff')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text(`${data.unit}${Math.round(data.value)}`);

        // Clean label
        gaugeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', 10)
            .style('fill', '#cccccc')
            .style('font-size', '11px')
            .style('font-weight', '500')
            .text(data.name);

        // Clean description
        gaugeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', radius + 25)
            .style('fill', '#999999')
            .style('font-size', '10px')
            .text(data.label);
    }

    createCreativePayoffMatrix(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 80, right: 60, bottom: 80, left: 200 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "radial-gradient(circle at center, #1a1a1a 0%, #000000 100%)");

        const defs = svg.append("defs");
        
        // Create hex pattern
        const hexPattern = defs.append("pattern")
            .attr("id", "hex-pattern")
            .attr("width", 30)
            .attr("height", 30)
            .attr("patternUnits", "userSpaceOnUse");
        
        hexPattern.append("polygon")
            .attr("points", "15,3 25,10 25,20 15,27 5,20 5,10")
            .style("fill", "none")
            .style("stroke", "#333333")
            .style("stroke-width", 0.5);

        svg.append("rect")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("fill", "url(#hex-pattern)")
            .style("opacity", 0.1);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Data
        const outcomes = Object.keys(this.model.outcomes);
        const players = Object.keys(this.model.players);
        const matrixData = [];

        outcomes.forEach((outcome, i) => {
            players.forEach((player, j) => {
                const preference = this.model.preferences[player][outcome];
                matrixData.push({
                    outcome: outcome,
                    player: player,
                    preference: preference,
                    outcomeIndex: i,
                    playerIndex: j
                });
            });
        });

        // Scales
        const xScale = d3.scaleBand()
            .domain(players)
            .range([0, width])
            .padding(0.2);

        const yScale = d3.scaleBand()
            .domain(outcomes)
            .range([0, height])
            .padding(0.2);

        // Color scale with neon effect
        const colorScale = d3.scaleLinear()
            .domain([1, 5])
            .range(["#00ff88", "#ff0066"]);

        // Create cells with 3D effect
        const cells = g.selectAll(".matrix-cell")
            .data(matrixData)
            .enter().append("g")
            .attr("class", "matrix-cell")
            .attr("transform", d => `translate(${xScale(d.player)}, ${yScale(d.outcome)})`);

        // 3D effect layers
        cells.append("rect")
            .attr("x", 5)
            .attr("y", 5)
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("rx", 10)
            .style("fill", "#000000")
            .style("opacity", 0.5);

        cells.append("rect")
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("rx", 10)
            .style("fill", d => colorScale(d.preference))
            .style("stroke", d => colorScale(d.preference))
            .style("stroke-width", 2)
            .style("filter", "drop-shadow(0 0 10px currentColor)")
            .attr("class", "cell-main")
            .on("mouseover", function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr("transform", "translate(-5, -5)")
                    .style("filter", "drop-shadow(0 0 20px currentColor) brightness(1.3)");
            })
            .on("mouseout", function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr("transform", "translate(0, 0)")
                    .style("filter", "drop-shadow(0 0 10px currentColor)");
            });

        // Preference numbers with glow
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("fill", "#ffffff")
            .style("font-size", "24px")
            .style("font-weight", "bold")
            .style("text-shadow", "0 0 20px rgba(255, 255, 255, 0.8)")
            .text(d => d.preference);

        // Creative axis labels
        players.forEach(player => {
            const label = g.append("g")
                .attr("transform", `translate(${xScale(player) + xScale.bandwidth() / 2}, ${height + 40})`);
            
            label.append("text")
                .attr("text-anchor", "middle")
                .style("fill", "#00ffff")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .style("letter-spacing", "2px")
                .style("text-transform", "uppercase")
                .text(player);
        });

        outcomes.forEach(outcome => {
            const label = g.append("g")
                .attr("transform", `translate(-20, ${yScale(outcome) + yScale.bandwidth() / 2})`);
            
            label.append("text")
                .attr("text-anchor", "end")
                .style("fill", "#ff00ff")
                .style("font-size", "12px")
                .style("font-weight", "500")
                .style("letter-spacing", "1px")
                .text(outcome.replace('_', ' '));
        });

        // Title with neon effect
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 40)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "24px")
            .style("font-weight", "900")
            .style("letter-spacing", "3px")
            .style("text-transform", "uppercase")
            .style("text-shadow", "0 0 30px #00ffff")
            .text("Preference Matrix");
    }

    createCreativeStrategyComparison(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 80, right: 80, bottom: 100, left: 100 };
        const width = 600 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)");

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Add radar grid background
        const gridLevels = 5;
        const gridGroup = g.append("g").attr("class", "radar-grid");
        
        for (let i = 1; i <= gridLevels; i++) {
            gridGroup.append("rect")
                .attr("x", (width / gridLevels) * (gridLevels - i) / 2)
                .attr("y", (height / gridLevels) * (gridLevels - i) / 2)
                .attr("width", (width / gridLevels) * i)
                .attr("height", (height / gridLevels) * i)
                .style("fill", "none")
                .style("stroke", "#333333")
                .style("stroke-width", 0.5)
                .style("opacity", 0.5);
        }

        // Get strategy data
        const strategies = this.model.getStrategyRankings();

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(strategies, d => d.warRisk) * 1.1])
            .range([0, width])
            .nice();

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(strategies, d => d.utilities.USA) * 1.1])
            .range([height, 0])
            .nice();

        const radiusScale = d3.scaleSqrt()
            .domain([0, d3.max(strategies, d => d.successProb)])
            .range([20, 50]);

        // Add axes with glow effect
        const xAxis = g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format(".0%")).ticks(5));

        const yAxis = g.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(yScale).ticks(5));

        // Style axes
        g.selectAll(".axis text")
            .style("fill", "#00ffff")
            .style("font-size", "12px")
            .style("text-shadow", "0 0 5px #00ffff");

        g.selectAll(".axis path, .axis line")
            .style("stroke", "#00ffff")
            .style("opacity", 0.5);

        // Strategy bubbles with creative effects
        const bubbles = g.selectAll(".strategy-bubble")
            .data(strategies)
            .enter().append("g")
            .attr("class", "strategy-bubble")
            .attr("transform", d => `translate(${xScale(d.warRisk)}, ${yScale(d.utilities.USA)})`);

        // Outer glow circle
        bubbles.append("circle")
            .attr("r", d => radiusScale(d.successProb) + 10)
            .style("fill", "none")
            .style("stroke", (d, i) => ["#00ff88", "#ffff00", "#ff6600", "#ff0066"][i])
            .style("stroke-width", 2)
            .style("opacity", 0.3)
            .style("filter", "blur(5px)");

        // Main bubble
        bubbles.append("circle")
            .attr("r", d => radiusScale(d.successProb))
            .style("fill", (d, i) => ["#00ff88", "#ffff00", "#ff6600", "#ff0066"][i])
            .style("opacity", 0.8)
            .style("filter", "drop-shadow(0 0 20px currentColor)")
            .attr("class", "main-bubble")
            .on("mouseover", function(event, d) {
                d3.select(this.parentNode).select(".main-bubble")
                    .transition()
                    .duration(300)
                    .attr("r", radiusScale(d.successProb) + 10)
                    .style("opacity", 1);
                
                // Create ripple effect
                d3.select(this.parentNode).append("circle")
                    .attr("class", "ripple")
                    .attr("r", radiusScale(d.successProb))
                    .style("fill", "none")
                    .style("stroke", d3.select(this).style("fill"))
                    .style("stroke-width", 2)
                    .style("opacity", 1)
                    .transition()
                    .duration(800)
                    .attr("r", radiusScale(d.successProb) + 50)
                    .style("opacity", 0)
                    .remove();
            })
            .on("mouseout", function(event, d) {
                d3.select(this.parentNode).select(".main-bubble")
                    .transition()
                    .duration(300)
                    .attr("r", radiusScale(d.successProb))
                    .style("opacity", 0.8);
            });

        // Strategy icons
        bubbles.append("text")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("font-size", "28px")
            .text(d => d.icon);

        // Connecting lines between strategies
        const lineGenerator = d3.line()
            .x(d => xScale(d.warRisk))
            .y(d => yScale(d.utilities.USA))
            .curve(d3.curveCardinal.tension(0.5));

        g.append("path")
            .datum(strategies)
            .attr("d", lineGenerator)
            .style("fill", "none")
            .style("stroke", "#00ffff")
            .style("stroke-width", 1)
            .style("opacity", 0.3)
            .style("stroke-dasharray", "5 5")
            .attr("class", "data-flow-line");

        // Creative axis labels
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", height + margin.top + 80)
            .attr("text-anchor", "middle")
            .style("fill", "#ff00ff")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .style("letter-spacing", "2px")
            .style("text-transform", "uppercase")
            .style("text-shadow", "0 0 10px #ff00ff")
            .text("War Risk →");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2 - margin.top)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", "#00ff88")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .style("letter-spacing", "2px")
            .style("text-transform", "uppercase")
            .style("text-shadow", "0 0 10px #00ff88")
            .text("← USA Utility");

        // Title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 40)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "28px")
            .style("font-weight", "900")
            .style("letter-spacing", "4px")
            .style("text-transform", "uppercase")
            .style("text-shadow", "0 0 30px #ffffff")
            .text("Strategy Arena");
    }

    setupExplainableAI() {
        // Connect the explainable AI component to the MCMC model
        this.explainableAI.setModel(this.model);
        
        // Initial update
        this.updateExplainableAI();
    }

    async updateExplainableAI() {
        try {
            // Update the analysis with current model state
            await this.explainableAI.updateAnalysis();
        } catch (error) {
            console.error('Error updating explainable AI:', error);
        }
    }

    setupExportModal() {
        const modal = document.getElementById('export-modal');
        const closeBtn = modal?.querySelector('.close');
        const exportBtns = modal?.querySelectorAll('.export-btn');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }

        window.addEventListener('click', (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });

        if (exportBtns) {
            exportBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const vizType = e.target.dataset.viz;
                    this.exportVisualization(vizType);
                });
            });
        }
    }

    showExportModal() {
        const modal = document.getElementById('export-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    exportVisualization(vizType) {
        // This would typically send the SVG to a server endpoint for PNG conversion
        // For now, we'll just log the action
        console.log(`Exporting ${vizType} visualization...`);
        
        if (vizType === 'all') {
            ['escalation-heatmap', 'strategy-comparison', 'payoff-matrix', 'outcome-probabilities'].forEach(type => {
                this.downloadSVG(type);
            });
        } else {
            this.downloadSVG(vizType);
        }
    }

    downloadSVG(containerId) {
        const svgElement = document.querySelector(`#${containerId} svg`);
        if (!svgElement) return;

        const svgData = new XMLSerializer().serializeToString(svgElement);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);
        
        const downloadLink = document.createElement('a');
        downloadLink.href = svgUrl;
        downloadLink.download = `${containerId}.svg`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new GameTheoryApp();
    window.gameTheoryApp = app; // Make available globally for debugging
});