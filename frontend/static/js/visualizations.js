// Clean and Clear D3.js Visualizations for Game Theory Model

class GameTheoryVisualizations {
    constructor() {
        this.model = new GameTheoryModel();
        this.colors = {
            primary: '#007acc',
            success: '#2ed573',
            warning: '#ffa502',
            danger: '#ff3838',
            secondary: '#ff4757',
            background: '#1a1a1a',
            text: '#ffffff',
            textSecondary: '#cccccc',
            textMuted: '#999999'
        };
        
        // Clean color schemes for different data types
        this.schemes = {
            escalation: ['#2ed573', '#7bed9f', '#ffa502', '#ff6348', '#ff3838'],
            preferences: ['#2ed573', '#7bed9f', '#70a1ff', '#ff6348', '#ff3838'],
            strategies: ['#2ed573', '#ffa502', '#ff6348', '#ff3838']
        };
    }

    // Create clean escalation probability heatmap
    createEscalationHeatmap(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 80, right: 140, left: 140, bottom: 100 };
        const width = 700 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Clean data preparation
        const militaryOptions = ['Halt & Deter', 'Expand Strikes'];
        const diplomaticOptions = ['De-escalatory Off-Ramp', 'Coercive Ultimatum'];
        
        const heatmapData = [];
        militaryOptions.forEach((military, i) => {
            diplomaticOptions.forEach((diplomatic, j) => {
                const strategyKey = this.getStrategyKey(military, diplomatic);
                const escalationData = this.model.getEscalationProbabilities()
                    .find(d => d.strategy === strategyKey);
                
                heatmapData.push({
                    military: military,
                    diplomatic: diplomatic,
                    militaryIndex: i,
                    diplomaticIndex: j,
                    escalationProb: escalationData.escalationProbability,
                    warRisk: escalationData.warRisk,
                    nuclearRisk: escalationData.nuclearRisk,
                    successProb: escalationData.successProb,
                    strategy: escalationData.strategyName,
                    icon: escalationData.icon
                });
            });
        });

        // Clean scales
        const xScale = d3.scaleBand()
            .domain(diplomaticOptions)
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleBand()
            .domain(militaryOptions)
            .range([0, height])
            .padding(0.1);

        // Simple, clear color scale
        const maxEscalation = d3.max(heatmapData, d => d.escalationProb);
        const colorScale = d3.scaleLinear()
            .domain([0, maxEscalation])
            .range(['#2ed573', '#ff3838']);

        // Create clean tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("background", "rgba(0, 0, 0, 0.9)")
            .style("color", "white")
            .style("padding", "12px")
            .style("border-radius", "6px")
            .style("font-size", "14px")
            .style("border", "1px solid #007acc")
            .style("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.3)");

        // Draw clean heatmap cells
        const cells = g.selectAll(".heatmap-cell")
            .data(heatmapData)
            .enter().append("g")
            .attr("class", "heatmap-cell")
            .attr("transform", d => `translate(${xScale(d.diplomatic)}, ${yScale(d.military)})`);

        // Clean cell backgrounds
        cells.append("rect")
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("rx", 8)
            .style("fill", d => colorScale(d.escalationProb))
            .style("stroke", "#ffffff")
            .style("stroke-width", 2)
            .style("opacity", 0.9)
            .on("mouseover", function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("opacity", 1)
                    .style("stroke-width", 3)
                    .style("stroke", "#007acc");

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">${d.icon} ${d.strategy}</div>
                    <div style="margin: 4px 0;"><strong>Escalation Risk:</strong> ${(d.escalationProb * 100).toFixed(1)}%</div>
                    <div style="margin: 4px 0;"><strong>War Risk:</strong> ${(d.warRisk * 100).toFixed(1)}%</div>
                    <div style="margin: 4px 0;"><strong>Nuclear Risk:</strong> ${(d.nuclearRisk * 100).toFixed(1)}%</div>
                    <div style="margin: 4px 0;"><strong>Success Probability:</strong> ${(d.successProb * 100).toFixed(1)}%</div>
                `)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("opacity", 0.9)
                    .style("stroke-width", 2)
                    .style("stroke", "#ffffff");

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 0);
            });

        // Clean percentage text with better contrast
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2 - 15)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("fill", "#ffffff")
            .style("font-size", "20px")
            .style("font-weight", "bold")
            .style("text-shadow", "2px 2px 4px rgba(0,0,0,0.8)")
            .text(d => `${(d.escalationProb * 100).toFixed(0)}%`);

        // Clean strategy icons
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2 + 15)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("font-size", "28px")
            .text(d => d.icon);

        // Clean axis labels with better spacing
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${height + 30})`)
            .selectAll(".x-label")
            .data(diplomaticOptions)
            .enter().append("text")
            .attr("class", "x-label")
            .attr("x", d => xScale(d) + xScale.bandwidth() / 2)
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .style("fill", "#cccccc")
            .style("font-size", "14px")
            .style("font-weight", "500")
            .text(d => d);

        g.append("g")
            .attr("class", "y-axis")
            .attr("transform", `translate(-30, 0)`)
            .selectAll(".y-label")
            .data(militaryOptions)
            .enter().append("text")
            .attr("class", "y-label")
            .attr("x", 0)
            .attr("y", d => yScale(d) + yScale.bandwidth() / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "central")
            .style("fill", "#cccccc")
            .style("font-size", "14px")
            .style("font-weight", "500")
            .text(d => d);

        // Clean axis titles
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", height + margin.top + 80)
            .attr("text-anchor", "middle")
            .style("fill", "#007acc")
            .style("font-size", "16px")
            .style("font-weight", "600")
            .text("🕊️ Diplomatic Posture");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2 - margin.top)
            .attr("y", 40)
            .attr("text-anchor", "middle")
            .style("fill", "#007acc")
            .style("font-size", "16px")
            .style("font-weight", "600")
            .text("⚔️ Military Posture");

        // Clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "20px")
            .style("font-weight", "700")
            .text("Escalation Probability by Strategy");

        // Clean subtitle
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 55)
            .attr("text-anchor", "middle")
            .style("fill", "#999999")
            .style("font-size", "14px")
            .text("Percentage shows risk of war or nuclear escalation");

        // Add clean color legend
        this.addCleanColorLegend(svg, colorScale, width + margin.left + 20, margin.top, 20, height);
    }

    // Create clean payoff matrix
    createPayoffMatrix(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 60, right: 40, bottom: 60, left: 180 };
        const width = 500 - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Clean data preparation
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
                    utility: 6 - preference,
                    outcomeIndex: i,
                    playerIndex: j,
                    outcomeName: this.model.outcomes[outcome]
                });
            });
        });

        // Clean scales
        const xScale = d3.scaleBand()
            .domain(players)
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleBand()
            .domain(outcomes)
            .range([0, height])
            .padding(0.1);

        // Simple, clear color scale
        const colorScale = d3.scaleOrdinal()
            .domain([1, 2, 3, 4, 5])
            .range(this.schemes.preferences);

        // Create clean tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("background", "rgba(0, 0, 0, 0.9)")
            .style("color", "white")
            .style("padding", "12px")
            .style("border-radius", "6px")
            .style("font-size", "14px")
            .style("border", "1px solid #007acc");

        // Draw clean matrix cells
        const cells = g.selectAll(".matrix-cell")
            .data(matrixData)
            .enter().append("g")
            .attr("class", "matrix-cell")
            .attr("transform", d => `translate(${xScale(d.player)}, ${yScale(d.outcome)})`);

        cells.append("rect")
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("rx", 6)
            .style("fill", d => colorScale(d.preference))
            .style("stroke", "#ffffff")
            .style("stroke-width", 1)
            .on("mouseover", function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("stroke", "#007acc")
                    .style("stroke-width", 3);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 4px;">${d.player} Preference</div>
                    <div style="margin: 2px 0;">${d.outcomeName}</div>
                    <div style="margin: 2px 0;"><strong>Ranking:</strong> ${d.preference}/5</div>
                    <div style="font-size: 12px; color: #999;">${d.preference === 1 ? 'Most Preferred' : d.preference === 5 ? 'Least Preferred' : 'Moderate Preference'}</div>
                `)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .style("stroke", "#ffffff")
                    .style("stroke-width", 1);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 0);
            });

        // Clean preference numbers
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("fill", "#ffffff")
            .style("font-size", "18px")
            .style("font-weight", "bold")
            .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
            .text(d => d.preference);

        // Clean axes with better labels
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${height + 15})`)
            .selectAll(".x-label")
            .data(players)
            .enter().append("text")
            .attr("x", d => xScale(d) + xScale.bandwidth() / 2)
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .style("fill", "#007acc")
            .style("font-size", "16px")
            .style("font-weight", "600")
            .text(d => d);

        g.append("g")
            .attr("class", "y-axis")
            .attr("transform", "translate(-15, 0)")
            .selectAll(".y-label")
            .data(outcomes)
            .enter().append("text")
            .attr("x", 0)
            .attr("y", d => yScale(d) + yScale.bandwidth() / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "central")
            .style("fill", "#cccccc")
            .style("font-size", "13px")
            .style("font-weight", "500")
            .text(d => d.replace('_', ' '));

        // Clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "18px")
            .style("font-weight", "700")
            .text("Player Preference Matrix");

        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 50)
            .attr("text-anchor", "middle")
            .style("fill", "#999999")
            .style("font-size", "12px")
            .text("1 = Most Preferred • 5 = Least Preferred");
    }

    // Clean strategy comparison scatter plot
    createStrategyComparison(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 60, right: 60, bottom: 80, left: 80 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Get clean strategy data
        const strategies = this.model.getStrategyRankings();

        // Clean scales
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
            .range([8, 25]);

        // Add clean grid
        g.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale)
                .tickSize(-height)
                .tickFormat("")
                .ticks(5)
            )
            .selectAll("line")
            .style("stroke", "#333333")
            .style("stroke-width", 1)
            .style("stroke-dasharray", "2,2");

        g.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale)
                .tickSize(-width)
                .tickFormat("")
                .ticks(5)
            )
            .selectAll("line")
            .style("stroke", "#333333")
            .style("stroke-width", 1)
            .style("stroke-dasharray", "2,2");

        // Add clean axes
        const xAxis = g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format(".0%")).ticks(5));

        const yAxis = g.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(yScale).ticks(5));

        // Style axes
        g.selectAll(".axis text")
            .style("fill", "#cccccc")
            .style("font-size", "12px");

        g.selectAll(".axis path, .axis line")
            .style("stroke", "#666666");

        // Create clean tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("background", "rgba(0, 0, 0, 0.9)")
            .style("color", "white")
            .style("padding", "12px")
            .style("border-radius", "6px")
            .style("font-size", "14px")
            .style("border", "1px solid #007acc");

        // Draw clean strategy points
        const points = g.selectAll(".strategy-point")
            .data(strategies)
            .enter().append("circle")
            .attr("class", "strategy-point")
            .attr("cx", d => xScale(d.warRisk))
            .attr("cy", d => yScale(d.utilities.USA))
            .attr("r", d => radiusScale(d.successProb))
            .style("fill", (d, i) => this.schemes.strategies[i])
            .style("opacity", 0.8)
            .style("stroke", "#ffffff")
            .style("stroke-width", 2)
            .on("mouseover", function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr("r", radiusScale(d.successProb) + 4)
                    .style("opacity", 1)
                    .style("stroke", "#007acc")
                    .style("stroke-width", 3);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 8px;">${d.icon} ${d.strategyName}</div>
                    <div style="margin: 4px 0;"><strong>USA Utility:</strong> ${d.utilities.USA.toFixed(2)}</div>
                    <div style="margin: 4px 0;"><strong>War Risk:</strong> ${(d.warRisk * 100).toFixed(1)}%</div>
                    <div style="margin: 4px 0;"><strong>Success Probability:</strong> ${(d.successProb * 100).toFixed(1)}%</div>
                `)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr("r", radiusScale(d.successProb))
                    .style("opacity", 0.8)
                    .style("stroke", "#ffffff")
                    .style("stroke-width", 2);

                tooltip.transition()
                    .duration(200)
                    .style("opacity", 0);
            });

        // Add clean strategy labels
        g.selectAll(".strategy-label")
            .data(strategies)
            .enter().append("text")
            .attr("class", "strategy-label")
            .attr("x", d => xScale(d.warRisk))
            .attr("y", d => yScale(d.utilities.USA) - radiusScale(d.successProb) - 12)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "18px")
            .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
            .text(d => d.icon);

        // Add clean axis labels
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", height + margin.top + 60)
            .attr("text-anchor", "middle")
            .style("fill", "#cccccc")
            .style("font-size", "14px")
            .style("font-weight", "500")
            .text("War Risk →");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2 - margin.top)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .style("fill", "#cccccc")
            .style("font-size", "14px")
            .style("font-weight", "500")
            .text("← USA Utility");

        // Add clean title
        svg.append("text")
            .attr("x", width / 2 + margin.left)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "18px")
            .style("font-weight", "700")
            .text("Strategy Risk vs Reward");
    }

    // Helper function to add clean color legend
    addCleanColorLegend(svg, colorScale, x, y, width, height) {
        const legendHeight = height;
        const legendWidth = width;

        const legend = svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${x}, ${y})`);

        // Create clean gradient
        const gradient = svg.append("defs")
            .append("linearGradient")
            .attr("id", "clean-legend-gradient")
            .attr("x1", "0%")
            .attr("x2", "0%")
            .attr("y1", "0%")
            .attr("y2", "100%");

        const steps = 10;
        for (let i = 0; i <= steps; i++) {
            const value = i / steps;
            gradient.append("stop")
                .attr("offset", `${value * 100}%`)
                .attr("stop-color", colorScale(value * colorScale.domain()[1]));
        }

        // Add clean rectangle with gradient
        legend.append("rect")
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#clean-legend-gradient)")
            .style("stroke", "#ffffff")
            .style("stroke-width", 1);

        // Add clean scale
        const legendScale = d3.scaleLinear()
            .domain(colorScale.domain())
            .range([0, legendHeight]);

        const legendAxis = d3.axisRight(legendScale)
            .tickFormat(d3.format(".0%"))
            .ticks(5);

        legend.append("g")
            .attr("class", "legend-axis")
            .attr("transform", `translate(${legendWidth}, 0)`)
            .call(legendAxis)
            .selectAll("text")
            .style("fill", "#cccccc")
            .style("font-size", "12px");

        legend.selectAll(".legend-axis path, .legend-axis line")
            .style("stroke", "#666666");

        // Add clean legend title
        legend.append("text")
            .attr("x", legendWidth / 2)
            .attr("y", -15)
            .attr("text-anchor", "middle")
            .style("fill", "#cccccc")
            .style("font-size", "12px")
            .style("font-weight", "500")
            .text("Risk Level");
    }

    // Helper function to get strategy key from military and diplomatic options
    getStrategyKey(military, diplomatic) {
        if (military === 'Halt & Deter' && diplomatic === 'De-escalatory Off-Ramp') {
            return 'deterrence_diplomacy';
        } else if (military === 'Halt & Deter' && diplomatic === 'Coercive Ultimatum') {
            return 'deterrence_ultimatum';
        } else if (military === 'Expand Strikes' && diplomatic === 'De-escalatory Off-Ramp') {
            return 'escalation_diplomacy';
        } else if (military === 'Expand Strikes' && diplomatic === 'Coercive Ultimatum') {
            return 'escalation_ultimatum';
        }
        return null;
    }

    // Update all visualizations
    updateVisualizations() {
        this.createEscalationHeatmap('escalation-heatmap');
        this.createPayoffMatrix('payoff-matrix');
        this.createStrategyComparison('strategy-comparison');
    }
}

// Export for use in main.js
window.GameTheoryVisualizations = GameTheoryVisualizations;