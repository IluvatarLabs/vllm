#!/usr/bin/env python3
"""
Figure 1: Elastic Speculation Overview
Clean block diagram showing adaptive draft length and confidence-based early exit
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set up clean, modern style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme - clean, professional
COLOR_DRAFT = '#E8F4F8'  # Light blue for draft model
COLOR_TARGET = '#FFF4E6'  # Light orange for target model
COLOR_CONTROL = '#F0F0F0'  # Light gray for control
COLOR_KV = '#E8F5E9'  # Light green for KV cache
COLOR_ACCENT = '#2E86AB'  # Blue accent
COLOR_ACCENT2 = '#A23B72'  # Purple accent
COLOR_TEXT = '#2C3E50'

# Title
title = ax.text(6, 9.5, 'Elastic Speculation Overview',
                ha='center', va='top', fontsize=18, fontweight='bold',
                color=COLOR_TEXT)

# Subtitle
subtitle = ax.text(6, 9.0, 'Adaptive Draft Length + Confidence-Based Early Exit (KV Gating)',
                   ha='center', va='top', fontsize=11, color='#7F8C8D', style='italic')

# ============================================================================
# SECTION 1: ADAPTIVE DRAFT LENGTH (LEFT SIDE)
# ============================================================================

# Draft Model block
draft_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.8,
                           boxstyle="round,pad=0.1",
                           facecolor=COLOR_DRAFT,
                           edgecolor=COLOR_ACCENT, linewidth=2)
ax.add_patch(draft_box)
ax.text(1.75, 6.7, 'Draft Model', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_TEXT)
ax.text(1.75, 6.3, '(EAGLE)', ha='center', va='center',
        fontsize=9, color='#7F8C8D')
ax.text(1.75, 5.9, 'Generate K tokens', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)

# Adaptive Control block
control_box = FancyBboxPatch((0.5, 3.5), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_CONTROL,
                             edgecolor=COLOR_ACCENT2, linewidth=2)
ax.add_patch(control_box)
ax.text(1.75, 4.6, 'Adaptive Control', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_TEXT)
ax.text(1.75, 4.2, 'Adjust K based on', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(1.75, 3.9, 'acceptance rate', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)

# Arrow: Control → Draft (feedback)
arrow1 = FancyArrowPatch((1.75, 5.0), (1.75, 5.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=COLOR_ACCENT2,
                        connectionstyle="arc3,rad=0")
ax.add_patch(arrow1)
ax.text(2.1, 5.25, 'K ∈ {5,10,15}', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLOR_ACCENT2, linewidth=1.5))

# Example values
ax.text(1.75, 3.2, 'Examples:', ha='center', va='center',
        fontsize=8, fontweight='bold', color=COLOR_TEXT)
ax.text(1.75, 2.9, 'High accept → K=15', ha='center', va='center',
        fontsize=8, color='#27AE60')
ax.text(1.75, 2.6, 'Low accept → K=5', ha='center', va='center',
        fontsize=8, color='#E74C3C')

# ============================================================================
# SECTION 2: TARGET MODEL VERIFICATION (CENTER)
# ============================================================================

# Arrow: Draft → Target
arrow2 = FancyArrowPatch((3.0, 6.4), (4.5, 6.4),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color=COLOR_ACCENT)
ax.add_patch(arrow2)
ax.text(3.75, 6.8, 'K draft tokens', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLOR_ACCENT, linewidth=1.5))

# Target Model block
target_box = FancyBboxPatch((4.5, 5.5), 3.0, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=COLOR_TARGET,
                            edgecolor=COLOR_ACCENT, linewidth=2)
ax.add_patch(target_box)
ax.text(6.0, 6.7, 'Target Model', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_TEXT)
ax.text(6.0, 6.3, 'Verify in parallel', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(6.0, 5.9, 'Accept prefix', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)

# Arrow: Target → Control (feedback)
arrow3 = FancyArrowPatch((5.5, 5.5), (2.5, 5.0),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=1.5, color=COLOR_ACCENT2, linestyle='--',
                        connectionstyle="arc3,rad=0.3")
ax.add_patch(arrow3)
ax.text(4.0, 4.8, 'acceptance rate', fontsize=8, style='italic', color=COLOR_ACCENT2)

# ============================================================================
# SECTION 3: CONFIDENCE-BASED EARLY EXIT (RIGHT SIDE)
# ============================================================================

# Early Exit Control block
exit_box = FancyBboxPatch((8.0, 5.5), 3.5, 1.8,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_CONTROL,
                          edgecolor=COLOR_ACCENT2, linewidth=2)
ax.add_patch(exit_box)
ax.text(9.75, 6.8, 'Early Exit Control', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_TEXT)
ax.text(9.75, 6.4, 'Monitor draft confidence', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(9.75, 6.0, 'If conf < threshold:', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(9.75, 5.7, 'Stop generating & gate KV', ha='center', va='center',
        fontsize=9, color='#E74C3C', fontweight='bold')

# Arrow: Draft → Early Exit
arrow4 = FancyArrowPatch((3.0, 6.0), (8.0, 6.0),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=1.5, color=COLOR_ACCENT2, linestyle='--',
                        connectionstyle="arc3,rad=0")
ax.add_patch(arrow4)
ax.text(5.5, 6.35, 'confidence scores', fontsize=8, style='italic', color=COLOR_ACCENT2)

# KV Cache block
kv_box = FancyBboxPatch((8.0, 3.2), 3.5, 1.8,
                        boxstyle="round,pad=0.1",
                        facecolor=COLOR_KV,
                        edgecolor='#27AE60', linewidth=2)
ax.add_patch(kv_box)
ax.text(9.75, 4.5, 'KV Cache Writes', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_TEXT)
ax.text(9.75, 4.1, 'Gate low-confidence', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(9.75, 3.7, 'draft tokens', ha='center', va='center',
        fontsize=9, color=COLOR_TEXT)
ax.text(9.75, 3.35, '~50% DRAM reduction', ha='center', va='center',
        fontsize=9, color='#27AE60', fontweight='bold')

# Arrow: Early Exit → KV Cache
arrow5 = FancyArrowPatch((9.75, 5.5), (9.75, 5.0),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#27AE60')
ax.add_patch(arrow5)
ax.text(10.3, 5.25, 'gate signal', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#27AE60', linewidth=1.5))

# ============================================================================
# BENEFITS SECTION (BOTTOM)
# ============================================================================

# Benefit box 1
benefit1_box = FancyBboxPatch((0.5, 0.5), 5.0, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='#E8F4F8',
                              edgecolor=COLOR_ACCENT, linewidth=1.5)
ax.add_patch(benefit1_box)
ax.text(3.0, 1.4, '📊 Adaptive Draft Length', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_TEXT)
ax.text(3.0, 1.0, '~30–50% lower latency vs. fixed-length', ha='center', va='center',
        fontsize=9, color=COLOR_ACCENT, fontweight='bold')
ax.text(3.0, 0.7, 'Dynamically adjusts speculation depth', ha='center', va='center',
        fontsize=8, color='#7F8C8D')

# Benefit box 2
benefit2_box = FancyBboxPatch((6.5, 0.5), 5.0, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9',
                              edgecolor='#27AE60', linewidth=1.5)
ax.add_patch(benefit2_box)
ax.text(9.0, 1.4, '💾 Confidence-Based Early Exit', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_TEXT)
ax.text(9.0, 1.0, '~50% less KV-cache DRAM traffic', ha='center', va='center',
        fontsize=9, color='#27AE60', fontweight='bold')
ax.text(9.0, 0.7, 'Minimal latency impact (~1–3%)', ha='center', va='center',
        fontsize=8, color='#7F8C8D')

plt.tight_layout()
plt.savefig('figure1_elastic_spec_overview.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figure1_elastic_spec_overview.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved figure1_elastic_spec_overview.png")
print("✓ Saved figure1_elastic_spec_overview.pdf")
plt.close()
