"""
Interactive Game Theory Model for Iran-Israel-US Conflict Analysis
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.models.players import GameTheoryModel, GameVariables, Outcome
from backend.utils.calculations import (
    calculate_strategy_rankings, 
    sensitivity_analysis, 
    calculate_escalation_ladder_position,
    calculate_market_impact_metrics,
    run_monte_carlo_simulation
)
from frontend.components.visualizations import (
    create_payoff_matrix,
    create_strategy_comparison,
    create_escalation_ladder,
    create_outcome_probabilities,
    create_sensitivity_chart
)


def main():
    st.set_page_config(
        page_title="Game Theory: Iran-Israel-US Conflict Model",
        page_icon="♟️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("♟️ Game Theory Model: Iran-Israel-US Strategic Analysis")
    st.markdown("*Interactive model based on June 2025 conflict research*")
    
    # Sidebar for variable controls
    st.sidebar.header("🎛️ Game Variables")
    
    # Create variable controls
    variables = create_variable_controls()
    
    # Initialize model
    model = GameTheoryModel(variables)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Analysis", 
        "🎯 Outcome Probabilities", 
        "📈 Sensitivity Analysis",
        "🚨 Risk Dashboard",
        "🎲 Monte Carlo Simulation"
    ])
    
    with tab1:
        strategy_analysis_tab(model)
    
    with tab2:
        outcome_probabilities_tab(model)
    
    with tab3:
        sensitivity_analysis_tab(model)
    
    with tab4:
        risk_dashboard_tab(model)
    
    with tab5:
        monte_carlo_tab(model)
    
    # Footer
    st.markdown("---")
    st.markdown("*Based on game-theoretic analysis of Iran-Israel-US strategic interactions*")


def create_variable_controls():
    """Create sidebar controls for game variables."""
    st.sidebar.markdown("### Current Crisis Variables")
    
    # Regime cohesion
    regime_cohesion = st.sidebar.slider(
        "🏛️ Iranian Regime Cohesion",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Internal unity of Iranian leadership (0=fractured, 1=unified)"
    )
    
    # Economic stress
    economic_stress = st.sidebar.slider(
        "💰 Economic Stress Level",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Iranian economic distress from sanctions (0=stable, 1=crisis)"
    )
    
    # Proxy support
    proxy_support = st.sidebar.slider(
        "🔗 Proxy Network Support",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Effectiveness of Iran's regional proxies (0=collapsed, 1=strong)"
    )
    
    # Oil price
    oil_price = st.sidebar.slider(
        "🛢️ Oil Price (USD/barrel)",
        min_value=50.0,
        max_value=150.0,
        value=97.0,
        step=1.0,
        help="Brent crude oil price affecting global markets"
    )
    
    # External support
    external_support = st.sidebar.slider(
        "🌍 External Support (China/Russia)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Level of material support from China/Russia (0=none, 1=full)"
    )
    
    # Nuclear progress
    nuclear_progress = st.sidebar.slider(
        "☢️ Nuclear Program Progress",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Iranian nuclear weapons capability (0=none, 1=weapon ready)"
    )
    
    return GameVariables(
        regime_cohesion=regime_cohesion,
        economic_stress=economic_stress,
        proxy_support=proxy_support,
        oil_price=oil_price,
        external_support=external_support,
        nuclear_progress=nuclear_progress
    )


def strategy_analysis_tab(model):
    """Strategy analysis and comparison tab."""
    st.header("🎯 US Strategic Options Analysis")
    
    # Calculate strategy rankings
    strategy_df = calculate_strategy_rankings(model)
    
    # Display strategy comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Strategy Comparison")
        
        # Create interactive strategy comparison chart
        fig = create_strategy_comparison(strategy_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Strategy Rankings")
        
        # Display ranked strategies
        for idx, row in strategy_df.iterrows():
            rank = idx + 1
            strategy_name = row['strategy'].replace('_', ' ').title()
            
            # Color code by rank
            color = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🔸"
            
            st.markdown(f"""
            **{color} {rank}. {strategy_name}**
            - USA Utility: {row['usa_utility']:.2f}
            - War Risk: {row['war_risk']:.1%}
            - Success Prob: {row['success_probability']:.1%}
            """)
    
    # Detailed strategy breakdown
    st.subheader("Detailed Strategy Analysis")
    
    selected_strategy = st.selectbox(
        "Select strategy for detailed analysis:",
        options=list(model.strategies.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Show detailed analysis for selected strategy
    probs = model.get_outcome_probabilities(selected_strategy)
    utilities = model.get_expected_utilities(selected_strategy)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("USA Expected Utility", f"{utilities['USA']:.2f}")
        st.metric("Iran Expected Utility", f"{utilities['Iran']:.2f}")
        st.metric("Israel Expected Utility", f"{utilities['Israel']:.2f}")
    
    with col2:
        st.metric("Success Probability", f"{probs[Outcome.DEAL] + probs[Outcome.LIMITED_RETALIATION]:.1%}")
        st.metric("War Risk", f"{probs[Outcome.FULL_WAR] + probs[Outcome.NUCLEAR_BREAKOUT]:.1%}")
        st.metric("Negotiation Probability", f"{probs[Outcome.DEAL]:.1%}")
    
    with col3:
        # Show strategy components
        military, diplomatic = model.strategies[selected_strategy]
        st.markdown(f"**Military Posture:**  \n{military.value}")
        st.markdown(f"**Diplomatic Posture:**  \n{diplomatic.value}")


def outcome_probabilities_tab(model):
    """Outcome probabilities visualization tab."""
    st.header("🎯 Outcome Probabilities by Strategy")
    
    # Calculate probabilities for all strategies
    all_probs = {}
    for strategy in model.strategies:
        all_probs[strategy] = model.get_outcome_probabilities(strategy)
    
    # Create outcome probability visualization
    fig = create_outcome_probabilities(all_probs)
    st.plotly_chart(fig, use_container_width=True)
    
    # Payoff matrix visualization
    st.subheader("Player Preference Matrix")
    
    # Create payoff matrix
    payoff_fig = create_payoff_matrix(model)
    st.plotly_chart(payoff_fig, use_container_width=True)
    
    # Detailed outcome descriptions
    st.subheader("Outcome Descriptions")
    
    outcome_descriptions = {
        Outcome.DEAL: "🤝 **Iranian Capitulation & Verifiable Deal**: Iran agrees to permanent, verifiable end to nuclear weapons program in exchange for sanctions relief.",
        Outcome.LIMITED_RETALIATION: "⚡ **Limited Iranian Retaliation & De-escalation**: Iran conducts face-saving, limited strike followed by negotiations.",
        Outcome.FROZEN_CONFLICT: "🧊 **Protracted Low-Intensity Conflict**: Return to 'shadow war' with higher baseline tension.",
        Outcome.FULL_WAR: "💥 **Full-Scale Regional War**: Iran retaliates massively, potential Strait of Hormuz closure.",
        Outcome.NUCLEAR_BREAKOUT: "☢️ **Iranian Nuclear Breakout**: Iran expels inspectors and dashes for nuclear weapon."
    }
    
    for outcome, description in outcome_descriptions.items():
        st.markdown(description)


def sensitivity_analysis_tab(model):
    """Sensitivity analysis tab."""
    st.header("📈 Sensitivity Analysis")
    
    # Variable selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_variable = st.selectbox(
            "Select variable to analyze:",
            options=['regime_cohesion', 'economic_stress', 'proxy_support', 'oil_price', 'external_support', 'nuclear_progress'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        selected_strategy = st.selectbox(
            "Select strategy:",
            options=list(model.strategies.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            key="sensitivity_strategy"
        )
    
    # Determine variable range
    if selected_variable == 'oil_price':
        min_val, max_val = 50.0, 150.0
    else:
        min_val, max_val = 0.0, 1.0
    
    # Run sensitivity analysis
    with st.spinner("Running sensitivity analysis..."):
        sensitivity_df = sensitivity_analysis(model, selected_strategy, selected_variable, min_val, max_val)
    
    # Create sensitivity chart
    fig = create_sensitivity_chart(sensitivity_df, selected_variable)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show key insights
    st.subheader("Key Sensitivity Insights")
    
    # Calculate correlations
    if selected_variable != 'oil_price':
        usa_corr = sensitivity_df[selected_variable].corr(sensitivity_df['usa_utility'])
        war_corr = sensitivity_df[selected_variable].corr(sensitivity_df['war_risk'])
        
        st.markdown(f"""
        - **USA Utility Correlation**: {usa_corr:.3f} {'(positive)' if usa_corr > 0 else '(negative)'}
        - **War Risk Correlation**: {war_corr:.3f} {'(positive)' if war_corr > 0 else '(negative)'}
        - **Variable Impact**: {'High' if abs(usa_corr) > 0.5 else 'Medium' if abs(usa_corr) > 0.3 else 'Low'}
        """)


def risk_dashboard_tab(model):
    """Risk dashboard tab."""
    st.header("🚨 Risk Dashboard")
    
    # Calculate risk metrics
    escalation_metrics = calculate_escalation_ladder_position(model.variables)
    market_metrics = calculate_market_impact_metrics(model.variables)
    
    # Escalation ladder
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎚️ Escalation Ladder")
        
        # Create escalation ladder visualization
        fig = create_escalation_ladder(escalation_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Current DEFCON level
        defcon_level = escalation_metrics['defcon_level']
        defcon_color = "🟢" if defcon_level > 4 else "🟡" if defcon_level > 3 else "🟠" if defcon_level > 2 else "🔴"
        st.metric("Current DEFCON Estimate", f"{defcon_level:.1f} {defcon_color}")
    
    with col2:
        st.subheader("📊 Market Impact Indicators")
        
        # Market metrics
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("VIX (Fear Index)", f"{market_metrics['vix_estimate']:.0f}")
            st.metric("Gold Price", f"${market_metrics['gold_price_estimate']:.0f}")
        
        with col2_2:
            st.metric("Oil Price", f"${market_metrics['oil_price']:.0f}/bbl")
            st.metric("Iranian Rial", f"{market_metrics['rial_rate_estimate']:.0f}/USD")
    
    # Risk factor breakdown
    st.subheader("🔍 Risk Factor Analysis")
    
    # Create risk factor chart
    factors_df = pd.DataFrame({
        'Factor': list(escalation_metrics['escalation_factors'].keys()),
        'Impact': list(escalation_metrics['escalation_factors'].values())
    })
    
    fig = px.bar(
        factors_df, 
        x='Impact', 
        y='Factor',
        orientation='h',
        title="Escalation Risk Factors",
        color='Impact',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def monte_carlo_tab(model):
    """Monte Carlo simulation tab."""
    st.header("🎲 Monte Carlo Simulation")
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
    
    with col2:
        uncertainty = st.slider("Variable Uncertainty", 0.05, 0.3, 0.1, 0.05)
    
    with col3:
        selected_strategy = st.selectbox(
            "Strategy to Simulate",
            options=list(model.strategies.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            key="monte_carlo_strategy"
        )
    
    # Run simulation button
    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_simulations} simulations..."):
            results = run_monte_carlo_simulation(model, selected_strategy, n_simulations, uncertainty)
        
        # Display results
        st.subheader("Simulation Results")
        
        # Outcome distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Outcome Probabilities:**")
            outcome_probs = results['summary']['outcome_probabilities']
            for outcome, prob in outcome_probs.items():
                st.markdown(f"- {outcome.value}: {prob:.1%}")
        
        with col2:
            st.markdown("**Expected Utilities:**")
            mean_utils = results['summary']['mean_utilities']
            std_utils = results['summary']['std_utilities']
            for player in ['USA', 'Iran', 'Israel']:
                st.markdown(f"- {player}: {mean_utils[player]:.2f} ± {std_utils[player]:.2f}")
        
        # Outcome distribution chart
        outcome_counts = pd.Series(results['outcomes']).value_counts()
        fig = px.pie(
            values=outcome_counts.values,
            names=[outcome.value for outcome in outcome_counts.index],
            title=f"Outcome Distribution ({n_simulations} simulations)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Utility distribution
        utility_data = []
        for player in ['USA', 'Iran', 'Israel']:
            for util in results['utilities'][player]:
                utility_data.append({'Player': player, 'Utility': util})
        
        utility_df = pd.DataFrame(utility_data)
        fig = px.box(
            utility_df,
            x='Player',
            y='Utility',
            title="Utility Distribution by Player"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()