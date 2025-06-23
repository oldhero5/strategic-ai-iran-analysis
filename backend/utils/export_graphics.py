"""
Export functions for creating X (Twitter) post graphics.
"""

import os
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from backend.models.players import GameTheoryModel, GameVariables, Outcome


def setup_export_directory():
    """Ensure graphics export directory exists."""
    graphics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graphics')
    os.makedirs(graphics_dir, exist_ok=True)
    return graphics_dir


def create_x_payoff_matrix(save_path: str = None) -> str:
    """Create payoff matrix graphic optimized for X post."""
    
    # Data from research
    data = {
        'USA': [1, 2, 4, 5, 3],
        'IRAN': [3, 2, 4, 5, 1],
        'ISRAEL': [2, 4, 3, 5, 1]
    }
    
    outcomes = ['Deal', 'Limited', 'Frozen', 'War', 'Nuclear']
    
    # Create matplotlib figure for better text control
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    df = pd.DataFrame(data, index=outcomes)
    sns.heatmap(df.T, annot=True, cmap='RdYlGn_r', cbar=False, 
                square=True, linewidths=2, linecolor='white',
                fmt='d', annot_kws={'size': 16, 'weight': 'bold'})
    
    # Styling
    plt.title('🎮 PLAYER PREFERENCES (1=best, 5=worst)', 
              fontsize=18, fontweight='bold', pad=20, color='white')
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(3)
    
    # Style axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(colors='white', labelsize=12)
    
    # Add description text
    plt.figtext(0.02, 0.02, 'Notice: Iran ranks "Nuclear Breakout" as #1!', 
                fontsize=10, style='italic', color='yellow')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        graphics_dir = setup_export_directory()
        save_path = os.path.join(graphics_dir, 'payoff_matrix_x.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='white')
    plt.close()
    
    return save_path


def create_x_escalation_ladder(variables: GameVariables, save_path: str = None) -> str:
    """Create escalation ladder graphic for X post."""
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # DEFCON levels
    levels = ['DEFCON 5\n░░░░░ Normal', 
              'DEFCON 4\n████░ Current*', 
              'DEFCON 3\n████░ Border strikes',
              'DEFCON 2\n████░ Major war',
              'DEFCON 1\n████░ Nuclear use']
    
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    y_positions = range(len(levels))
    
    # Current level calculation (simplified)
    current_level = 4 - (variables.economic_stress + (1-variables.regime_cohesion)) * 1.5
    current_level = max(1, min(5, current_level))
    
    # Create bars
    for i, (level, color) in enumerate(zip(levels, colors)):
        # Highlight current level
        alpha = 1.0 if abs(5-i - current_level) < 0.5 else 0.4
        ax.barh(i, 1, color=color, alpha=alpha, height=0.8)
        
        # Add text
        ax.text(0.5, i, level, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Add current level indicator
    current_y = 5 - current_level
    ax.axhline(y=current_y, color='white', linestyle='--', linewidth=3)
    ax.text(1.1, current_y, f'Current: {current_level:.1f}', 
            va='center', fontsize=12, fontweight='bold', color='white')
    
    # Styling
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title('🚨 ESCALATION LADDER\n*After US strikes', 
                 fontsize=16, fontweight='bold', pad=20, color='white')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save_path is None:
        graphics_dir = setup_export_directory()
        save_path = os.path.join(graphics_dir, 'escalation_ladder_x.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    return save_path


def create_x_strategy_matrix(save_path: str = None) -> str:
    """Create strategy option matrix for X post."""
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Strategy data
    strategies = [
        ['✅ Halt', 'Offer', 'Low'],
        ['⚠️ Halt', 'Ultimatum', 'Med'],
        ['🚫 Expand', 'Offer', 'High'],
        ['☠️ Expand', 'Ultimatum', 'MAX']
    ]
    
    headers = ['Military', 'Diplomatic', 'Risk']
    
    # Create table
    table_data = []
    colors = []
    for i, row in enumerate(strategies):
        table_data.append(row)
        # Color code by risk
        if row[2] == 'Low':
            colors.append(['lightgreen'] * 3)
        elif row[2] == 'Med':
            colors.append(['yellow'] * 3)
        elif row[2] == 'High':
            colors.append(['orange'] * 3)
        else:
            colors.append(['red'] * 3)
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     cellColours=colors)
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    
    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('darkblue')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(strategies) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_text_props(weight='bold', color='black')
    
    ax.set_title('🎯 THE FOUR STRATEGIC PATHS\nOnly Option 1 avoids triggering desperation moves', 
                 fontsize=16, fontweight='bold', pad=20, color='white')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        graphics_dir = setup_export_directory()
        save_path = os.path.join(graphics_dir, 'strategy_matrix_x.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    return save_path


def create_x_market_indicators(variables: GameVariables, save_path: str = None) -> str:
    """Create market impact indicators for X post."""
    
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Calculate market metrics
    oil_price = variables.oil_price
    vix_estimate = 20 + variables.economic_stress * 25 + (1-variables.regime_cohesion) * 20
    gold_estimate = 2000 + (variables.economic_stress + (1-variables.regime_cohesion)) * 400
    rial_rate = 42000 * (1 + variables.economic_stress * 12)
    
    # Oil price gauge
    ax1.pie([oil_price, 150-oil_price], labels=['', ''], 
            colors=['red' if oil_price > 100 else 'orange', 'gray'],
            startangle=90, counterclock=False)
    ax1.text(0, 0, f'${oil_price:.0f}', ha='center', va='center', 
             fontsize=20, fontweight='bold', color='white')
    ax1.set_title('🛢️ Oil (Brent)', fontsize=14, fontweight='bold', color='white')
    
    # VIX gauge
    ax2.pie([vix_estimate, 100-vix_estimate], labels=['', ''],
            colors=['red' if vix_estimate > 40 else 'orange', 'gray'],
            startangle=90, counterclock=False)
    ax2.text(0, 0, f'{vix_estimate:.0f}', ha='center', va='center',
             fontsize=20, fontweight='bold', color='white')
    ax2.set_title('📊 VIX (Fear)', fontsize=14, fontweight='bold', color='white')
    
    # Gold price
    ax3.pie([gold_estimate-2000, 3000-(gold_estimate-2000)], labels=['', ''],
            colors=['gold', 'gray'], startangle=90, counterclock=False)
    ax3.text(0, 0, f'${gold_estimate:.0f}', ha='center', va='center',
             fontsize=18, fontweight='bold', color='white')
    ax3.set_title('🥇 Gold', fontsize=14, fontweight='bold', color='white')
    
    # Iranian Rial
    rial_display = f'{rial_rate/1000:.0f}k'
    ax4.pie([rial_rate/500000, 1-rial_rate/500000], labels=['', ''],
            colors=['darkred', 'gray'], startangle=90, counterclock=False)
    ax4.text(0, 0, rial_display, ha='center', va='center',
             fontsize=16, fontweight='bold', color='white')
    ax4.set_title('💰 Rial/USD', fontsize=14, fontweight='bold', color='white')
    
    plt.suptitle('📊 REAL-TIME MARKET INDICATORS\nMarkets pricing in serious escalation risk', 
                 fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    if save_path is None:
        graphics_dir = setup_export_directory()
        save_path = os.path.join(graphics_dir, 'market_indicators_x.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    return save_path


def create_x_two_level_game(save_path: str = None) -> str:
    """Create two-level game visualization for X post."""
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create network diagram
    # Positions for nodes
    positions = {
        'USA': (0.5, 0.8),
        'IRAN': (0.2, 0.3),
        'ISRAEL': (0.8, 0.3),
        'INTL': (0.5, 0.6),
        'ALLIANCE': (0.5, 0.1)
    }
    
    # Draw nodes
    for node, (x, y) in positions.items():
        if node == 'USA':
            color, size = 'lightblue', 2000
        elif node in ['IRAN', 'ISRAEL']:
            color, size = 'lightcoral', 1500
        else:
            color, size = 'lightgray', 1000
        
        ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=2)
        ax.text(x, y, node, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')
    
    # Draw connections
    connections = [
        ('USA', 'IRAN'),
        ('USA', 'ISRAEL'),
        ('IRAN', 'ISRAEL'),
        ('USA', 'INTL'),
        ('USA', 'ALLIANCE')
    ]
    
    for node1, node2 in connections:
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        ax.plot([x1, x2], [y1, y2], 'white', linewidth=2, alpha=0.7)
    
    # Add game level labels
    ax.text(0.1, 0.6, '🌍 INTERNATIONAL\nGAME', fontsize=12, fontweight='bold', 
            color='yellow', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    ax.text(0.1, 0.1, '🤝 ALLIANCE\nGAME', fontsize=12, fontweight='bold',
            color='yellow', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('🎮 THE TWO-LEVEL GAME\nMust solve BOTH simultaneously!', 
                 fontsize=16, fontweight='bold', pad=20, color='white')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        graphics_dir = setup_export_directory()
        save_path = os.path.join(graphics_dir, 'two_level_game_x.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    return save_path


def export_all_x_graphics(model: GameTheoryModel) -> Dict[str, str]:
    """Export all graphics optimized for X posts."""
    
    graphics_dir = setup_export_directory()
    
    exported_files = {
        'payoff_matrix': create_x_payoff_matrix(),
        'escalation_ladder': create_x_escalation_ladder(model.variables),
        'strategy_matrix': create_x_strategy_matrix(),
        'market_indicators': create_x_market_indicators(model.variables),
        'two_level_game': create_x_two_level_game()
    }
    
    # Create summary file
    summary_path = os.path.join(graphics_dir, 'export_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Game Theory Iran Model - X Post Graphics Export\n")
        f.write("=" * 50 + "\n\n")
        for name, path in exported_files.items():
            f.write(f"{name}: {os.path.basename(path)}\n")
        f.write(f"\nExported to: {graphics_dir}\n")
        f.write(f"Total files: {len(exported_files)}\n")
    
    exported_files['summary'] = summary_path
    
    return exported_files