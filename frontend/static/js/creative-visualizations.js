// Creative and Visually Stunning D3.js Visualizations

class CreativeVisualizations {
    constructor() {
        this.model = new GameTheoryModel();
        this.particleSystem = new ParticleSystem();
        
        // Creative color palettes
        this.palettes = {
            neon: {
                primary: '#00ffff',
                secondary: '#ff00ff',
                accent: '#ffff00',
                danger: '#ff0066',
                success: '#00ff88',
                glow: '#ffffff'
            },
            cyberpunk: {
                primary: '#00d9ff',
                secondary: '#ff2a6d',
                accent: '#05ffa1',
                danger: '#ff1744',
                warning: '#ffab00',
                glow: '#e0e0e0'
            },
            military: {
                primary: '#4fc3f7',
                secondary: '#ff7043',
                accent: '#81c784',
                danger: '#e53935',
                warning: '#ffd54f',
                dark: '#263238'
            }
        };
        
        this.currentPalette = this.palettes.cyberpunk;
        this.initializeCreativeElements();
    }

    initializeCreativeElements() {
        // Add creative CSS for animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse-glow {
                0%, 100% { 
                    filter: drop-shadow(0 0 10px currentColor) drop-shadow(0 0 20px currentColor);
                    transform: scale(1);
                }
                50% { 
                    filter: drop-shadow(0 0 20px currentColor) drop-shadow(0 0 40px currentColor);
                    transform: scale(1.05);
                }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes rotate-slow {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            @keyframes data-flow {
                0% { stroke-dashoffset: 0; }
                100% { stroke-dashoffset: -100; }
            }
            
            .glow-effect {
                filter: drop-shadow(0 0 10px currentColor);
            }
            
            .pulse-animation {
                animation: pulse-glow 2s infinite;
            }
            
            .float-animation {
                animation: float 3s ease-in-out infinite;
            }
            
            .rotate-animation {
                animation: rotate-slow 20s linear infinite;
            }
            
            .data-flow-line {
                stroke-dasharray: 5 5;
                animation: data-flow 2s linear infinite;
            }
        `;
        document.head.appendChild(style);
    }

    // Create stunning escalation heatmap with creative effects
    createCreativeEscalationHeatmap(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 100, right: 150, left: 150, bottom: 120 };
        const width = 800 - margin.left - margin.right;
        const height = 600 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "radial-gradient(circle at center, #1a1a1a 0%, #000000 100%)");

        // Create patterns and gradients
        const defs = svg.append("defs");
        
        // Animated gradient background
        const bgGradient = defs.append("radialGradient")
            .attr("id", "animated-bg-gradient");
        
        bgGradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "#1a1a1a")
            .append("animate")
            .attr("attributeName", "stop-color")
            .attr("values", "#1a1a1a;#2a2a2a;#1a1a1a")
            .attr("dur", "4s")
            .attr("repeatCount", "indefinite");
        
        bgGradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "#000000");

        // Grid pattern
        const pattern = defs.append("pattern")
            .attr("id", "grid-pattern")
            .attr("width", 20)
            .attr("height", 20)
            .attr("patternUnits", "userSpaceOnUse");
        
        pattern.append("line")
            .attr("x1", 0).attr("y1", 0)
            .attr("x2", 0).attr("y2", 20)
            .style("stroke", "#333333")
            .style("stroke-width", 0.5);
        
        pattern.append("line")
            .attr("x1", 0).attr("y1", 0)
            .attr("x2", 20).attr("y2", 0)
            .style("stroke", "#333333")
            .style("stroke-width", 0.5);

        // Glow filters
        const glowFilter = defs.append("filter")
            .attr("id", "neon-glow")
            .attr("x", "-50%").attr("y", "-50%")
            .attr("width", "200%").attr("height", "200%");
        
        glowFilter.append("feGaussianBlur")
            .attr("stdDeviation", "4")
            .attr("result", "coloredBlur");
        
        const feMerge = glowFilter.append("feMerge");
        feMerge.append("feMergeNode").attr("in", "coloredBlur");
        feMerge.append("feMergeNode").attr("in", "SourceGraphic");

        // Background with grid
        svg.append("rect")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("fill", "url(#grid-pattern)")
            .style("opacity", 0.1);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Add floating particles in background
        this.addFloatingParticles(g, width, height);

        // Data preparation
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

        // Scales
        const xScale = d3.scaleBand()
            .domain(diplomaticOptions)
            .range([0, width])
            .padding(0.15);

        const yScale = d3.scaleBand()
            .domain(militaryOptions)
            .range([0, height])
            .padding(0.15);

        // Creative color scale
        const colorScale = d3.scaleLinear()
            .domain([0, 0.3, 0.7, 1])
            .range([this.currentPalette.success, this.currentPalette.accent, this.currentPalette.warning, this.currentPalette.danger]);

        // Create cells with creative effects
        const cells = g.selectAll(".heatmap-cell")
            .data(heatmapData)
            .enter().append("g")
            .attr("class", "heatmap-cell")
            .attr("transform", d => `translate(${xScale(d.diplomatic)}, ${yScale(d.military)})`);

        // Cell background with gradient
        cells.each(function(d) {
            const cell = d3.select(this);
            const cellId = `cell-gradient-${d.militaryIndex}-${d.diplomaticIndex}`;
            
            // Create unique gradient for each cell
            const cellGradient = defs.append("linearGradient")
                .attr("id", cellId)
                .attr("x1", "0%").attr("y1", "0%")
                .attr("x2", "100%").attr("y2", "100%");
            
            cellGradient.append("stop")
                .attr("offset", "0%")
                .attr("stop-color", colorScale(d.escalationProb))
                .attr("stop-opacity", 0.8);
            
            cellGradient.append("stop")
                .attr("offset", "100%")
                .attr("stop-color", colorScale(d.escalationProb))
                .attr("stop-opacity", 1);
            
            // Main cell rectangle
            cell.append("rect")
                .attr("width", xScale.bandwidth())
                .attr("height", yScale.bandwidth())
                .attr("rx", 15)
                .style("fill", `url(#${cellId})`)
                .style("stroke", "none")
                .style("filter", "url(#neon-glow)")
                .attr("class", "cell-rect");
            
            // Add inner glow
            cell.append("rect")
                .attr("width", xScale.bandwidth() - 10)
                .attr("height", yScale.bandwidth() - 10)
                .attr("x", 5)
                .attr("y", 5)
                .attr("rx", 12)
                .style("fill", "none")
                .style("stroke", colorScale(d.escalationProb))
                .style("stroke-width", 2)
                .style("opacity", 0.5)
                .attr("class", "inner-glow");
        });

        // Add creative hover effects
        cells.on("mouseover", function(event, d) {
            const cell = d3.select(this);
            
            // Animate the cell
            cell.select(".cell-rect")
                .transition()
                .duration(300)
                .attr("transform", "scale(1.05)")
                .style("filter", "url(#neon-glow) brightness(1.3)");
            
            cell.select(".inner-glow")
                .transition()
                .duration(300)
                .style("stroke-width", 4)
                .style("opacity", 1);
            
            // Create explosion effect
            const explosion = cell.append("g")
                .attr("class", "explosion-effect");
            
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2;
                const distance = 50;
                
                explosion.append("circle")
                    .attr("cx", xScale.bandwidth() / 2)
                    .attr("cy", yScale.bandwidth() / 2)
                    .attr("r", 3)
                    .style("fill", colorScale(d.escalationProb))
                    .style("opacity", 1)
                    .transition()
                    .duration(600)
                    .attr("cx", xScale.bandwidth() / 2 + Math.cos(angle) * distance)
                    .attr("cy", yScale.bandwidth() / 2 + Math.sin(angle) * distance)
                    .attr("r", 0)
                    .style("opacity", 0)
                    .remove();
            }
            
            // Show tooltip
            showCreativeTooltip(event, d);
        })
        .on("mouseout", function() {
            const cell = d3.select(this);
            
            cell.select(".cell-rect")
                .transition()
                .duration(300)
                .attr("transform", "scale(1)")
                .style("filter", "url(#neon-glow)");
            
            cell.select(".inner-glow")
                .transition()
                .duration(300)
                .style("stroke-width", 2)
                .style("opacity", 0.5);
            
            cell.select(".explosion-effect").remove();
            hideTooltip();
        });

        // Add percentage with creative styling
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2 - 20)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("fill", "#ffffff")
            .style("font-size", "28px")
            .style("font-weight", "bold")
            .style("text-shadow", `0 0 20px ${this.currentPalette.glow}`)
            .style("letter-spacing", "2px")
            .text(d => `${(d.escalationProb * 100).toFixed(0)}%`)
            .attr("class", "percentage-text pulse-animation");

        // Add strategy icons with animation
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2 + 20)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("font-size", "36px")
            .text(d => d.icon)
            .attr("class", "float-animation");

        // Add risk indicator bars
        cells.append("g")
            .attr("class", "risk-indicators")
            .attr("transform", `translate(10, ${yScale.bandwidth() - 30})`)
            .each(function(d) {
                const indicator = d3.select(this);
                
                // War risk bar
                indicator.append("rect")
                    .attr("width", (xScale.bandwidth() - 20) * d.warRisk)
                    .attr("height", 4)
                    .attr("rx", 2)
                    .style("fill", "#ff1744")
                    .style("opacity", 0.8);
                
                // Nuclear risk bar
                indicator.append("rect")
                    .attr("y", 6)
                    .attr("width", (xScale.bandwidth() - 20) * d.nuclearRisk)
                    .attr("height", 4)
                    .attr("rx", 2)
                    .style("fill", "#ff00ff")
                    .style("opacity", 0.8);
            });

        // Creative axis labels
        this.addCreativeAxisLabels(g, xScale, yScale, width, height, diplomaticOptions, militaryOptions);

        // Add creative title
        this.addCreativeTitle(svg, width, margin);

        // Add animated legend
        this.addAnimatedLegend(svg, colorScale, width + margin.left + 30, margin.top, 25, height);

        // Add corner decorations
        this.addCornerDecorations(svg, width + margin.left + margin.right, height + margin.top + margin.bottom);
    }

    addFloatingParticles(container, width, height) {
        const particleContainer = container.append("g")
            .attr("class", "particle-container")
            .style("opacity", 0.3);

        const particles = d3.range(20).map(i => ({
            x: Math.random() * width,
            y: Math.random() * height,
            r: Math.random() * 3 + 1,
            dx: (Math.random() - 0.5) * 0.5,
            dy: (Math.random() - 0.5) * 0.5
        }));

        const particleElements = particleContainer.selectAll(".particle")
            .data(particles)
            .enter().append("circle")
            .attr("class", "particle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.r)
            .style("fill", this.currentPalette.primary)
            .style("filter", "blur(1px)");

        // Animate particles
        const animateParticles = () => {
            particleElements
                .attr("cx", d => {
                    d.x += d.dx;
                    if (d.x < 0 || d.x > width) d.dx *= -1;
                    return d.x;
                })
                .attr("cy", d => {
                    d.y += d.dy;
                    if (d.y < 0 || d.y > height) d.dy *= -1;
                    return d.y;
                });
        };

        d3.interval(animateParticles, 50);
    }

    addCreativeAxisLabels(g, xScale, yScale, width, height, xLabels, yLabels) {
        // X-axis labels with icons
        const xAxisGroup = g.append("g")
            .attr("class", "x-axis-creative")
            .attr("transform", `translate(0, ${height + 40})`);

        xLabels.forEach((label, i) => {
            const labelGroup = xAxisGroup.append("g")
                .attr("transform", `translate(${xScale(label) + xScale.bandwidth() / 2}, 0)`);

            // Background circle
            labelGroup.append("circle")
                .attr("r", 25)
                .style("fill", this.currentPalette.primary)
                .style("opacity", 0.2)
                .style("filter", "blur(10px)");

            // Icon
            labelGroup.append("text")
                .attr("y", -5)
                .attr("text-anchor", "middle")
                .style("font-size", "24px")
                .text(i === 0 ? "🕊️" : "⚔️");

            // Label text
            labelGroup.append("text")
                .attr("y", 20)
                .attr("text-anchor", "middle")
                .style("fill", this.currentPalette.primary)
                .style("font-size", "14px")
                .style("font-weight", "600")
                .style("letter-spacing", "1px")
                .text(label);
        });

        // Y-axis labels with icons
        const yAxisGroup = g.append("g")
            .attr("class", "y-axis-creative")
            .attr("transform", `translate(-40, 0)`);

        yLabels.forEach((label, i) => {
            const labelGroup = yAxisGroup.append("g")
                .attr("transform", `translate(0, ${yScale(label) + yScale.bandwidth() / 2})`);

            // Background circle
            labelGroup.append("circle")
                .attr("r", 25)
                .style("fill", this.currentPalette.secondary)
                .style("opacity", 0.2)
                .style("filter", "blur(10px)");

            // Icon
            labelGroup.append("text")
                .attr("x", -25)
                .attr("text-anchor", "middle")
                .style("font-size", "24px")
                .text(i === 0 ? "🛡️" : "💥");

            // Label text
            labelGroup.append("text")
                .attr("text-anchor", "end")
                .style("fill", this.currentPalette.secondary)
                .style("font-size", "14px")
                .style("font-weight", "600")
                .style("letter-spacing", "1px")
                .text(label);
        });
    }

    addCreativeTitle(svg, width, margin) {
        const titleGroup = svg.append("g")
            .attr("class", "title-group")
            .attr("transform", `translate(${width / 2 + margin.left}, 50)`);

        // Background glow
        titleGroup.append("ellipse")
            .attr("rx", 300)
            .attr("ry", 40)
            .style("fill", this.currentPalette.primary)
            .style("opacity", 0.1)
            .style("filter", "blur(20px)");

        // Main title
        titleGroup.append("text")
            .attr("text-anchor", "middle")
            .style("fill", "#ffffff")
            .style("font-size", "32px")
            .style("font-weight", "900")
            .style("letter-spacing", "3px")
            .style("text-transform", "uppercase")
            .style("text-shadow", `0 0 30px ${this.currentPalette.primary}`)
            .text("ESCALATION MATRIX")
            .attr("class", "glow-effect");

        // Subtitle
        titleGroup.append("text")
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("fill", this.currentPalette.accent)
            .style("font-size", "16px")
            .style("font-weight", "300")
            .style("letter-spacing", "2px")
            .text("Strategic Risk Assessment")
            .style("opacity", 0)
            .transition()
            .duration(1000)
            .style("opacity", 1);

        // Decorative lines
        [-1, 1].forEach(direction => {
            titleGroup.append("line")
                .attr("x1", -200 * direction)
                .attr("x2", -50 * direction)
                .attr("y1", 0)
                .attr("y2", 0)
                .style("stroke", this.currentPalette.primary)
                .style("stroke-width", 2)
                .style("opacity", 0)
                .transition()
                .duration(1000)
                .style("opacity", 0.5);
        });
    }

    addAnimatedLegend(svg, colorScale, x, y, width, height) {
        const legendGroup = svg.append("g")
            .attr("class", "animated-legend")
            .attr("transform", `translate(${x}, ${y})`);

        // Create animated gradient
        const gradientId = "animated-legend-gradient";
        const gradient = svg.select("defs").append("linearGradient")
            .attr("id", gradientId)
            .attr("x1", "0%").attr("x2", "0%")
            .attr("y1", "0%").attr("y2", "100%");

        // Animated color stops
        const stops = 20;
        for (let i = 0; i <= stops; i++) {
            const offset = i / stops;
            gradient.append("stop")
                .attr("offset", `${offset * 100}%`)
                .attr("stop-color", colorScale(offset))
                .append("animate")
                .attr("attributeName", "stop-opacity")
                .attr("values", `${0.8 + 0.2 * Math.sin(i)};1;${0.8 + 0.2 * Math.sin(i)}`)
                .attr("dur", `${2 + i * 0.1}s`)
                .attr("repeatCount", "indefinite");
        }

        // Legend rectangle
        legendGroup.append("rect")
            .attr("width", width)
            .attr("height", height)
            .attr("rx", 10)
            .style("fill", `url(#${gradientId})`)
            .style("stroke", this.currentPalette.glow)
            .style("stroke-width", 1)
            .style("filter", "url(#neon-glow)");

        // Scale labels
        const legendScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, height]);

        const legendAxis = d3.axisRight(legendScale)
            .tickFormat(d3.format(".0%"))
            .ticks(5);

        const axisGroup = legendGroup.append("g")
            .attr("transform", `translate(${width + 5}, 0)`)
            .call(legendAxis);

        axisGroup.selectAll("text")
            .style("fill", this.currentPalette.accent)
            .style("font-size", "12px")
            .style("font-weight", "500");

        axisGroup.selectAll("line, path")
            .style("stroke", this.currentPalette.accent);

        // Title
        legendGroup.append("text")
            .attr("x", width / 2)
            .attr("y", -20)
            .attr("text-anchor", "middle")
            .style("fill", this.currentPalette.accent)
            .style("font-size", "14px")
            .style("font-weight", "600")
            .style("letter-spacing", "1px")
            .text("RISK LEVEL");
    }

    addCornerDecorations(svg, totalWidth, totalHeight) {
        const corners = [
            { x: 20, y: 20, rotate: 0 },
            { x: totalWidth - 20, y: 20, rotate: 90 },
            { x: totalWidth - 20, y: totalHeight - 20, rotate: 180 },
            { x: 20, y: totalHeight - 20, rotate: 270 }
        ];

        corners.forEach(corner => {
            const decoration = svg.append("g")
                .attr("transform", `translate(${corner.x}, ${corner.y}) rotate(${corner.rotate})`);

            // Corner bracket
            decoration.append("path")
                .attr("d", "M0,-15 L0,0 L15,0")
                .style("stroke", this.currentPalette.primary)
                .style("stroke-width", 2)
                .style("fill", "none")
                .style("opacity", 0.5);

            // Corner dot
            decoration.append("circle")
                .attr("r", 3)
                .style("fill", this.currentPalette.primary)
                .attr("class", "pulse-animation");
        });
    }

    // Helper function to get strategy key
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

    // Creative 3D Payoff Matrix with hexagonal pattern
    createCreativePayoffMatrix(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 80, right: 60, bottom: 80, left: 200 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "radial-gradient(circle at center, #0a0a0a 0%, #000000 100%)");

        // Add creative defs for patterns and filters
        const defs = svg.append("defs");
        
        // Hexagonal pattern
        const hexPattern = defs.append("pattern")
            .attr("id", "hex-pattern")
            .attr("width", 30)
            .attr("height", 26)
            .attr("patternUnits", "userSpaceOnUse");
        
        hexPattern.append("polygon")
            .attr("points", "15,2 27,9 27,19 15,26 3,19 3,9")
            .style("fill", "none")
            .style("stroke", this.currentPalette.primary)
            .style("stroke-width", 0.5)
            .style("opacity", 0.1);

        // 3D effect filter
        const shadow3D = defs.append("filter")
            .attr("id", "shadow-3d")
            .attr("x", "-50%").attr("y", "-50%")
            .attr("width", "200%").attr("height", "200%");
        
        shadow3D.append("feDropShadow")
            .attr("dx", 3).attr("dy", 3)
            .attr("stdDeviation", 3)
            .attr("flood-color", this.currentPalette.primary)
            .attr("flood-opacity", 0.3);

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Add hexagonal background
        g.append("rect")
            .attr("width", width)
            .attr("height", height)
            .style("fill", "url(#hex-pattern)");

        // Data preparation
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

        // Scales
        const xScale = d3.scaleBand()
            .domain(players)
            .range([0, width])
            .padding(0.15);

        const yScale = d3.scaleBand()
            .domain(outcomes)
            .range([0, height])
            .padding(0.15);

        // Creative 3D color scale
        const colorScale = d3.scaleLinear()
            .domain([1, 5])
            .range([this.currentPalette.success, this.currentPalette.danger]);

        // Create cells with 3D effects
        const cells = g.selectAll(".matrix-cell")
            .data(matrixData)
            .enter().append("g")
            .attr("class", "matrix-cell")
            .attr("transform", d => `translate(${xScale(d.player)}, ${yScale(d.outcome)})`);

        // 3D cell background
        cells.append("rect")
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("rx", 12)
            .style("fill", d => colorScale(d.preference))
            .style("filter", "url(#shadow-3d)")
            .style("stroke", this.currentPalette.glow)
            .style("stroke-width", 2)
            .attr("class", "cell-3d");

        // Inner glow rectangle
        cells.append("rect")
            .attr("width", xScale.bandwidth() - 8)
            .attr("height", yScale.bandwidth() - 8)
            .attr("x", 4)
            .attr("y", 4)
            .attr("rx", 8)
            .style("fill", "none")
            .style("stroke", d => colorScale(d.preference))
            .style("stroke-width", 1)
            .style("opacity", 0.5);

        // 3D highlight
        cells.append("rect")
            .attr("width", xScale.bandwidth() - 16)
            .attr("height", 8)
            .attr("x", 8)
            .attr("y", 8)
            .attr("rx", 4)
            .style("fill", "url(#neon-glow)")
            .style("opacity", 0.3);

        // Animated preference numbers
        cells.append("text")
            .attr("x", xScale.bandwidth() / 2)
            .attr("y", yScale.bandwidth() / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("fill", "#ffffff")
            .style("font-size", "24px")
            .style("font-weight", "900")
            .style("text-shadow", `0 0 20px ${this.currentPalette.glow}`)
            .text(d => d.preference)
            .attr("class", "pulse-animation");

        // Ripple effect on hover
        cells.on("mouseover", function(event, d) {
            const cell = d3.select(this);
            
            // Create ripple effect
            const ripple = cell.append("circle")
                .attr("cx", xScale.bandwidth() / 2)
                .attr("cy", yScale.bandwidth() / 2)
                .attr("r", 0)
                .style("fill", "none")
                .style("stroke", colorScale(d.preference))
                .style("stroke-width", 3)
                .style("opacity", 1);
            
            ripple.transition()
                .duration(800)
                .attr("r", Math.max(xScale.bandwidth(), yScale.bandwidth()) / 2)
                .style("opacity", 0)
                .remove();

            // Show tooltip
            showCreativeTooltip(event, {
                ...d,
                icon: d.preference === 1 ? "🥇" : d.preference === 2 ? "🥈" : d.preference === 3 ? "🥉" : d.preference === 4 ? "🔸" : "🔹",
                strategy: `${d.player} Preference`
            });
        })
        .on("mouseout", function() {
            hideTooltip();
        });

        // Creative axis labels
        this.addCreative3DAxisLabels(g, xScale, yScale, width, height, players, outcomes);
    }

    // Creative Strategy Comparison with floating bubbles
    createCreativeStrategyComparison(containerId) {
        const container = d3.select(`#${containerId} .chart-container`);
        container.selectAll("*").remove();

        const margin = { top: 80, right: 80, bottom: 100, left: 100 };
        const width = 600 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "radial-gradient(circle at center, #0a0a0a 0%, #000000 100%)");

        const defs = svg.append("defs");
        
        // Animated constellation background
        const constellation = defs.append("pattern")
            .attr("id", "constellation")
            .attr("width", 100)
            .attr("height", 100)
            .attr("patternUnits", "userSpaceOnUse");

        // Add stars to constellation
        for (let i = 0; i < 20; i++) {
            constellation.append("circle")
                .attr("cx", Math.random() * 100)
                .attr("cy", Math.random() * 100)
                .attr("r", Math.random() * 1.5 + 0.5)
                .style("fill", this.currentPalette.primary)
                .style("opacity", Math.random() * 0.8 + 0.2)
                .append("animate")
                .attr("attributeName", "opacity")
                .attr("values", "0.2;1;0.2")
                .attr("dur", `${2 + Math.random() * 3}s`)
                .attr("repeatCount", "indefinite");
        }

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Background constellation
        g.append("rect")
            .attr("width", width)
            .attr("height", height)
            .style("fill", "url(#constellation)")
            .style("opacity", 0.3);

        // Data
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
            .range([15, 40]);

        // Add floating bubbles in background
        this.addFloatingBubbles(g, width, height);

        // Creative grid lines with animation
        const gridGroup = g.append("g").attr("class", "animated-grid");
        
        // Vertical grid lines
        xScale.ticks(5).forEach((tick, i) => {
            gridGroup.append("line")
                .attr("x1", xScale(tick))
                .attr("x2", xScale(tick))
                .attr("y1", 0)
                .attr("y2", height)
                .style("stroke", this.currentPalette.primary)
                .style("stroke-width", 1)
                .style("opacity", 0.2)
                .style("stroke-dasharray", "4 4")
                .attr("class", "data-flow-line");
        });

        // Horizontal grid lines
        yScale.ticks(5).forEach((tick, i) => {
            gridGroup.append("line")
                .attr("x1", 0)
                .attr("x2", width)
                .attr("y1", yScale(tick))
                .attr("y2", yScale(tick))
                .style("stroke", this.currentPalette.secondary)
                .style("stroke-width", 1)
                .style("opacity", 0.2)
                .style("stroke-dasharray", "4 4")
                .attr("class", "data-flow-line");
        });

        // Strategy bubbles with creative effects
        const bubbles = g.selectAll(".strategy-bubble")
            .data(strategies)
            .enter().append("g")
            .attr("class", "strategy-bubble")
            .attr("transform", d => `translate(${xScale(d.warRisk)}, ${yScale(d.utilities.USA)})`);

        // Outer glow ring
        bubbles.append("circle")
            .attr("r", d => radiusScale(d.successProb) + 10)
            .style("fill", "none")
            .style("stroke", (d, i) => [this.currentPalette.primary, this.currentPalette.secondary, this.currentPalette.accent, this.currentPalette.warning][i])
            .style("stroke-width", 2)
            .style("opacity", 0.3)
            .attr("class", "pulse-animation");

        // Main bubble
        bubbles.append("circle")
            .attr("r", d => radiusScale(d.successProb))
            .style("fill", (d, i) => [this.currentPalette.primary, this.currentPalette.secondary, this.currentPalette.accent, this.currentPalette.warning][i])
            .style("opacity", 0.8)
            .style("filter", "url(#neon-glow)")
            .attr("class", "main-bubble");

        // Inner core
        bubbles.append("circle")
            .attr("r", d => radiusScale(d.successProb) / 3)
            .style("fill", "#ffffff")
            .style("opacity", 0.6);

        // Strategy icons with animation
        bubbles.append("text")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("font-size", "28px")
            .style("text-shadow", `0 0 20px ${this.currentPalette.glow}`)
            .text(d => d.icon)
            .attr("class", "float-animation");

        // Hover effects
        bubbles.on("mouseover", function(event, d) {
            const bubble = d3.select(this);
            
            // Expansion effect
            bubble.select(".main-bubble")
                .transition()
                .duration(300)
                .attr("r", radiusScale(d.successProb) * 1.2)
                .style("opacity", 1);

            // Create energy rings
            for (let i = 0; i < 3; i++) {
                bubble.append("circle")
                    .attr("r", radiusScale(d.successProb))
                    .style("fill", "none")
                    .style("stroke", "#ffffff")
                    .style("stroke-width", 2)
                    .style("opacity", 0.8)
                    .transition()
                    .delay(i * 100)
                    .duration(800)
                    .attr("r", radiusScale(d.successProb) * 2.5)
                    .style("opacity", 0)
                    .remove();
            }

            showCreativeTooltip(event, d);
        })
        .on("mouseout", function(event, d) {
            const bubble = d3.select(this);
            
            bubble.select(".main-bubble")
                .transition()
                .duration(300)
                .attr("r", radiusScale(d.successProb))
                .style("opacity", 0.8);

            hideTooltip();
        });

        // Creative axes with glowing labels
        this.addCreativeAxes(g, xScale, yScale, width, height);
    }

    addFloatingBubbles(container, width, height) {
        const bubbleContainer = container.append("g")
            .attr("class", "floating-bubbles")
            .style("opacity", 0.1);

        const bubbles = d3.range(15).map(i => ({
            x: Math.random() * width,
            y: Math.random() * height,
            r: Math.random() * 8 + 3,
            dx: (Math.random() - 0.5) * 0.3,
            dy: (Math.random() - 0.5) * 0.3,
            color: [this.currentPalette.primary, this.currentPalette.secondary, this.currentPalette.accent][i % 3]
        }));

        const bubbleElements = bubbleContainer.selectAll(".floating-bubble")
            .data(bubbles)
            .enter().append("circle")
            .attr("class", "floating-bubble")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.r)
            .style("fill", d => d.color)
            .style("opacity", 0.3);

        // Animate bubbles
        const animateBubbles = () => {
            bubbleElements
                .attr("cx", d => {
                    d.x += d.dx;
                    if (d.x < 0 || d.x > width) d.dx *= -1;
                    return d.x;
                })
                .attr("cy", d => {
                    d.y += d.dy;
                    if (d.y < 0 || d.y > height) d.dy *= -1;
                    return d.y;
                });
        };

        d3.interval(animateBubbles, 50);
    }

    addCreative3DAxisLabels(g, xScale, yScale, width, height, xLabels, yLabels) {
        // 3D X-axis labels
        const xAxisGroup = g.append("g")
            .attr("class", "x-axis-3d")
            .attr("transform", `translate(0, ${height + 40})`);

        xLabels.forEach((label, i) => {
            const labelGroup = xAxisGroup.append("g")
                .attr("transform", `translate(${xScale(label) + xScale.bandwidth() / 2}, 0)`);

            // 3D base
            labelGroup.append("ellipse")
                .attr("rx", 35)
                .attr("ry", 15)
                .attr("cy", 5)
                .style("fill", this.currentPalette.primary)
                .style("opacity", 0.3);

            // Main circle
            labelGroup.append("circle")
                .attr("r", 30)
                .style("fill", `linear-gradient(135deg, ${this.currentPalette.primary}, ${this.currentPalette.secondary})`)
                .style("stroke", this.currentPalette.glow)
                .style("stroke-width", 2)
                .style("filter", "url(#neon-glow)");

            // Label text
            labelGroup.append("text")
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "central")
                .style("fill", "#ffffff")
                .style("font-size", "14px")
                .style("font-weight", "700")
                .style("text-shadow", `0 0 10px ${this.currentPalette.glow}`)
                .text(label);
        });

        // 3D Y-axis labels
        const yAxisGroup = g.append("g")
            .attr("class", "y-axis-3d")
            .attr("transform", "translate(-40, 0)");

        yLabels.forEach((label, i) => {
            const labelGroup = yAxisGroup.append("g")
                .attr("transform", `translate(0, ${yScale(label) + yScale.bandwidth() / 2})`);

            // 3D base
            labelGroup.append("ellipse")
                .attr("rx", 15)
                .attr("ry", 35)
                .attr("cx", -5)
                .style("fill", this.currentPalette.secondary)
                .style("opacity", 0.3);

            // Main circle
            labelGroup.append("circle")
                .attr("r", 30)
                .style("fill", `linear-gradient(135deg, ${this.currentPalette.secondary}, ${this.currentPalette.accent})`)
                .style("stroke", this.currentPalette.glow)
                .style("stroke-width", 2)
                .style("filter", "url(#neon-glow)");

            // Label text
            labelGroup.append("text")
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "central")
                .style("fill", "#ffffff")
                .style("font-size", "12px")
                .style("font-weight", "700")
                .style("text-shadow", `0 0 10px ${this.currentPalette.glow}`)
                .text(label.replace('_', ' '));
        });
    }

    addCreativeAxes(g, xScale, yScale, width, height) {
        // X-axis with glow
        const xAxis = g.append("g")
            .attr("class", "x-axis-creative")
            .attr("transform", `translate(0, ${height})`);

        xAxis.append("line")
            .attr("x1", 0)
            .attr("x2", width)
            .style("stroke", this.currentPalette.primary)
            .style("stroke-width", 2)
            .style("filter", "url(#neon-glow)");

        // Y-axis with glow
        const yAxis = g.append("g")
            .attr("class", "y-axis-creative");

        yAxis.append("line")
            .attr("y1", 0)
            .attr("y2", height)
            .style("stroke", this.currentPalette.secondary)
            .style("stroke-width", 2)
            .style("filter", "url(#neon-glow)");

        // Animated tick marks
        xScale.ticks(5).forEach(tick => {
            xAxis.append("circle")
                .attr("cx", xScale(tick))
                .attr("r", 3)
                .style("fill", this.currentPalette.primary)
                .attr("class", "pulse-animation");
        });

        yScale.ticks(5).forEach(tick => {
            yAxis.append("circle")
                .attr("cy", yScale(tick))
                .attr("r", 3)
                .style("fill", this.currentPalette.secondary)
                .attr("class", "pulse-animation");
        });
    }
}

// Particle System for background effects
class ParticleSystem {
    constructor() {
        this.particles = [];
    }

    createExplosion(container, x, y, color, count = 20) {
        const particles = container.append("g")
            .attr("class", "particle-explosion");

        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 2;
            const velocity = 2 + Math.random() * 3;
            const size = 2 + Math.random() * 4;

            particles.append("circle")
                .attr("cx", x)
                .attr("cy", y)
                .attr("r", size)
                .style("fill", color)
                .style("opacity", 1)
                .transition()
                .duration(1000)
                .ease(d3.easeExpOut)
                .attr("cx", x + Math.cos(angle) * velocity * 50)
                .attr("cy", y + Math.sin(angle) * velocity * 50)
                .attr("r", 0)
                .style("opacity", 0)
                .remove();
        }
    }
}

// Creative tooltip functions
function showCreativeTooltip(event, data) {
    // Remove any existing tooltip
    d3.select(".creative-tooltip").remove();

    const tooltip = d3.select("body").append("div")
        .attr("class", "creative-tooltip")
        .style("position", "absolute")
        .style("background", "linear-gradient(135deg, rgba(0, 0, 0, 0.9) 0%, rgba(20, 20, 20, 0.9) 100%)")
        .style("border", "2px solid")
        .style("border-image", `linear-gradient(45deg, #00ffff, #ff00ff) 1`)
        .style("padding", "20px")
        .style("border-radius", "10px")
        .style("color", "white")
        .style("font-family", "monospace")
        .style("box-shadow", "0 0 30px rgba(0, 255, 255, 0.5)")
        .style("opacity", 0)
        .style("transform", "scale(0.8)")
        .style("pointer-events", "none");

    // Add content with creative styling
    tooltip.html(`
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 48px;">${data.icon}</span>
        </div>
        <h3 style="margin: 0 0 10px 0; font-size: 18px; color: #00ffff; text-transform: uppercase; letter-spacing: 2px;">
            ${data.strategy}
        </h3>
        <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px; font-size: 14px;">
            <span style="color: #ff00ff;">▸ ESCALATION:</span>
            <span style="color: #ffffff; font-weight: bold;">${(data.escalationProb * 100).toFixed(1)}%</span>
            
            <span style="color: #ff6666;">▸ WAR RISK:</span>
            <span style="color: #ffffff; font-weight: bold;">${(data.warRisk * 100).toFixed(1)}%</span>
            
            <span style="color: #ff00ff;">▸ NUCLEAR:</span>
            <span style="color: #ffffff; font-weight: bold;">${(data.nuclearRisk * 100).toFixed(1)}%</span>
            
            <span style="color: #00ff88;">▸ SUCCESS:</span>
            <span style="color: #ffffff; font-weight: bold;">${(data.successProb * 100).toFixed(1)}%</span>
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;">
            <div style="height: 4px; background: linear-gradient(90deg, #00ff88 0%, #00ff88 ${data.successProb * 100}%, #ff0066 ${data.successProb * 100}%, #ff0066 100%); border-radius: 2px;"></div>
        </div>
    `);

    // Position and animate
    tooltip
        .style("left", (event.pageX + 20) + "px")
        .style("top", (event.pageY - 20) + "px")
        .transition()
        .duration(300)
        .style("opacity", 1)
        .style("transform", "scale(1)");
}

function hideTooltip() {
    d3.select(".creative-tooltip")
        .transition()
        .duration(200)
        .style("opacity", 0)
        .style("transform", "scale(0.8)")
        .remove();
}

// Export for use
window.CreativeVisualizations = CreativeVisualizations;