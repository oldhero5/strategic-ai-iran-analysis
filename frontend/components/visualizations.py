"""
Visualization components for the game theory model.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from backend.models.players import GameTheoryModel, Outcome


def create_payoff_matrix(model: GameTheoryModel):
    """Create a heatmap visualization of player preferences."""
    
    # Create payoff matrix data
    outcomes = list(Outcome)
    players = ['USA', 'Iran', 'Israel']
    
    # Get preference data (convert to positive scale for better visualization)
    payoff_data = []
    for outcome in outcomes:
        row = []
        for player_obj in [model.usa, model.iran, model.israel]:
            # Convert to positive scale (6 - rank so higher is better)
            utility = 6 - player_obj.get_utility(outcome)
            row.append(utility)
        payoff_data.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=payoff_data,
        x=players,
        y=[outcome.value for outcome in outcomes],
        colorscale='RdYlGn',
        reversescale=False,
        text=payoff_data,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Player Preference Matrix (Higher = More Preferred)",
        xaxis_title="Players",
        yaxis_title="Outcomes",
        height=400
    )
    
    return fig


def create_strategy_comparison(strategy_df: pd.DataFrame):
    """Create a comparison chart of US strategies."""
    
    # Create scatter plot of USA utility vs War risk
    fig = px.scatter(
        strategy_df,
        x='war_risk',
        y='usa_utility',
        size='success_probability',
        color='uncertainty',
        hover_name='strategy',
        hover_data={
            'military_posture': True,
            'diplomatic_posture': True,
            'war_risk': ':.1%',
            'success_probability': ':.1%'
        },
        labels={
            'war_risk': 'War Risk (%)',
            'usa_utility': 'USA Expected Utility',
            'uncertainty': 'Uncertainty',
            'success_probability': 'Success Probability'
        },
        title="US Strategy Comparison: Utility vs Risk",
        color_continuous_scale='Viridis_r'
    )
    
    # Add annotations for strategy names
    for idx, row in strategy_df.iterrows():
        fig.add_annotation(
            x=row['war_risk'],
            y=row['usa_utility'],
            text=row['strategy'].replace('_', '<br>').title(),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(height=500)
    
    return fig


def create_outcome_probabilities(all_probs: Dict[str, Dict[Outcome, float]]):
    """Create stacked bar chart of outcome probabilities by strategy."""
    
    # Prepare data for stacked bar chart
    strategies = list(all_probs.keys())
    outcomes = list(Outcome)
    
    # Create traces for each outcome
    fig = go.Figure()
    
    colors = ['#2E8B57', '#FFD700', '#FF6347', '#DC143C', '#8B0000']  # Green to Red gradient
    
    for i, outcome in enumerate(outcomes):
        probabilities = [all_probs[strategy][outcome] for strategy in strategies]
        
        fig.add_trace(go.Bar(
            name=outcome.value,
            x=[s.replace('_', ' ').title() for s in strategies],
            y=probabilities,
            marker_color=colors[i],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='inside'
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Outcome Probabilities by US Strategy',
        xaxis_title='US Strategy',
        yaxis_title='Probability',
        yaxis=dict(tickformat='.0%'),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_escalation_ladder(escalation_metrics: Dict):
    """Create escalation ladder visualization."""
    
    # DEFCON levels and descriptions
    defcon_levels = [5, 4, 3, 2, 1]
    descriptions = [
        "Normal Readiness",
        "Increased Watch", 
        "Round House",
        "Fast Pace",
        "Exercise Term"
    ]
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    current_level = escalation_metrics['defcon_level']
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    for i, (level, desc, color) in enumerate(zip(defcon_levels, descriptions, colors)):
        # Determine if this is the current level
        opacity = 1.0 if abs(level - current_level) < 0.5 else 0.3
        
        fig.add_trace(go.Bar(
            x=[1],
            y=[f"DEFCON {level}"],
            orientation='h',
            marker=dict(color=color, opacity=opacity),
            text=desc,
            textposition='inside',
            name=f"DEFCON {level}",
            showlegend=False
        ))
    
    # Add current level indicator
    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=5.5 - current_level, y1=5.5 - current_level,
        line=dict(color="black", width=3, dash="dash"),
    )
    
    fig.add_annotation(
        x=0.5,
        y=5.5 - current_level,
        text=f"Current: {current_level:.1f}",
        showarrow=True,
        arrowhead=2,
        bgcolor="white",
        bordercolor="black"
    )
    
    fig.update_layout(
        title="Escalation Ladder (DEFCON Scale)",
        xaxis=dict(visible=False),
        height=300,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig


def create_sensitivity_chart(sensitivity_df: pd.DataFrame, variable_name: str):
    """Create sensitivity analysis chart."""
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # USA utility line
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df[variable_name],
            y=sensitivity_df['usa_utility'],
            mode='lines+markers',
            name='USA Utility',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )
    
    # War risk line
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df[variable_name],
            y=sensitivity_df['war_risk'],
            mode='lines+markers',
            name='War Risk',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True,
    )
    
    # Success probability line
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df[variable_name],
            y=sensitivity_df['success_prob'],
            mode='lines+markers',
            name='Success Probability',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_xaxes(title_text=variable_name.replace('_', ' ').title())
    fig.update_yaxes(title_text="Expected Utility", secondary_y=False)
    fig.update_yaxes(title_text="Probability", secondary_y=True, tickformat='.0%')
    
    fig.update_layout(
        title=f'Sensitivity Analysis: {variable_name.replace("_", " ").title()}',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_risk_gauge(value: float, title: str, max_value: float = 1.0):
    """Create a gauge chart for risk metrics."""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': max_value * 0.5},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value * 0.3], 'color': "lightgray"},
                {'range': [max_value * 0.3, max_value * 0.7], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig


def create_game_tree_visualization(model: GameTheoryModel, strategy: str):
    """Create a simplified game tree visualization."""
    
    # Get probabilities for the strategy
    probs = model.get_outcome_probabilities(strategy)
    
    # Create tree structure data
    # This is a simplified representation of the sequential game
    
    # Root node (US Strategy)
    military, diplomatic = model.strategies[strategy]
    root_text = f"US Strategy:<br>{military.value}<br>{diplomatic.value}"
    
    # Outcome nodes
    outcome_texts = []
    outcome_probs = []
    
    for outcome in probs:
        outcome_texts.append(f"{outcome.value}<br>({probs[outcome]:.1%})")
        outcome_probs.append(probs[outcome])
    
    # Create tree diagram using plotly (simplified)
    fig = go.Figure()
    
    # Add root node
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[0.8],
        mode='markers+text',
        text=[root_text],
        textposition='middle center',
        marker=dict(size=100, color='lightblue'),
        name='US Decision'
    ))
    
    # Add outcome nodes
    x_positions = np.linspace(0.1, 0.9, len(outcome_texts))
    
    for i, (text, prob) in enumerate(zip(outcome_texts, outcome_probs)):
        # Node size based on probability
        size = 50 + prob * 200
        
        # Color based on desirability for USA
        if 'Deal' in text or 'Limited' in text:
            color = 'lightgreen'
        elif 'War' in text or 'Nuclear' in text:
            color = 'lightcoral'
        else:
            color = 'lightyellow'
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[0.2],
            mode='markers+text',
            text=[text],
            textposition='middle center',
            marker=dict(size=size, color=color),
            name=f'Outcome {i+1}',
            showlegend=False
        ))
        
        # Add connecting line
        fig.add_shape(
            type="line",
            x0=0.5, y0=0.8,
            x1=x_positions[i], y1=0.2,
            line=dict(color="gray", width=2),
        )
    
    fig.update_layout(
        title=f"Game Tree: {strategy.replace('_', ' ').title()}",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig


def create_export_graphics_for_x():
    """Create graphics optimized for X (Twitter) posts."""
    
    # This function will create publication-ready graphics
    # matching the style from x.md
    
    # Payoff matrix for X
    def create_x_payoff_matrix():
        data = {
            'Outcome': ['Deal', 'Limited', 'Frozen', 'War', 'Nuclear'],
            'USA': [1, 2, 4, 5, 3],
            'IRAN': [3, 2, 4, 5, 1], 
            'ISRAEL': [2, 4, 3, 5, 1]
        }
        
        df = pd.DataFrame(data)
        
        fig = px.imshow(
            df.set_index('Outcome').T.values,
            x=df['Outcome'],
            y=['USA', 'IRAN', 'ISRAEL'],
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            text_auto=True
        )
        
        fig.update_layout(
            title="🎮 PLAYER PREFERENCES (1=best, 5=worst)",
            font=dict(size=14, family="Arial Black"),
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    # Strategy comparison for X
    def create_x_strategy_chart():
        strategies = ['Deterrence + Diplomacy', 'Deterrence + Ultimatum', 
                     'Escalation + Diplomacy', 'Escalation + Ultimatum']
        risks = [0.1, 0.2, 0.4, 0.6]
        rewards = [0.8, 0.6, 0.4, 0.3]
        
        fig = go.Figure()
        
        colors = ['green', 'yellow', 'orange', 'red']
        symbols = ['✅', '⚠️', '🚫', '☠️']
        
        for i, (strategy, risk, reward, color, symbol) in enumerate(zip(strategies, risks, rewards, colors, symbols)):
            fig.add_trace(go.Scatter(
                x=[risk],
                y=[reward],
                mode='markers+text',
                text=[f'{symbol} {strategy}'],
                textposition='top center',
                marker=dict(size=30, color=color),
                name=strategy,
                showlegend=False
            ))
        
        fig.update_layout(
            title="🎯 US STRATEGIC OPTIONS: RISK vs REWARD",
            xaxis_title="War Risk",
            yaxis_title="Success Probability",
            font=dict(size=12, family="Arial"),
            width=600,
            height=500
        )
        
        return fig
    
    return {
        'payoff_matrix': create_x_payoff_matrix(),
        'strategy_chart': create_x_strategy_chart()
    }