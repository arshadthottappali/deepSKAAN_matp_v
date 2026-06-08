"""
Generate additional poster figures:
  - Kinetic scheme diagram (A→B→C→GS with decay times)
  - DeepSKAN pipeline overview
  - Comparison table figure
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

out_dir = os.path.dirname(os.path.abspath(__file__))


def plot_kinetic_scheme():
    """Publication-quality kinetic scheme with decay times."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1.5, 3)
    ax.axis('off')

    # Energy levels (vertical position = relative energy)
    states = {
        'A': (1, 2.5, 'S₁ (hot)'),
        'B': (3.5, 2.0, 'S₁ (relaxed)'),
        'C': (6, 1.2, 'T₁ / CT'),
        'GS': (8.5, 0, 'S₀ (GS)'),
    }

    # Draw states as boxes
    for key, (x, y, label) in states.items():
        color = {'A': '#E74C3C', 'B': '#27AE60', 'C': '#2980B9', 'GS': '#95A5A6'}[key]
        rect = mpatches.FancyBboxPatch((x-0.6, y-0.2), 1.2, 0.4,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw arrows with decay times
    arrows = [
        ('A', 'B', 'τ₁\nD: 9.4 ps\nH: 13.8 ps'),
        ('B', 'C', 'τ₂\nD: 91.8 ps\nH: 173.9 ps'),
        ('C', 'GS', 'τ₃\nD: >6 ns\nH: >6 ns'),
    ]

    for start_key, end_key, label in arrows:
        x1, y1, _ = states[start_key]
        x2, y2, _ = states[end_key]
        ax.annotate('', xy=(x2 - 0.6, y2 + 0.1), xytext=(x1 + 0.6, y1 - 0.1),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='#2C3E50',
                                   connectionstyle='arc3,rad=-0.2'))
        # Label on arrow
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 0.5
        ax.text(mid_x, mid_y, label, ha='center', va='top', fontsize=9,
                color='#2C3E50', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.8))

    # Excitation arrow
    ax.annotate('', xy=(1, 2.3), xytext=(1, 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple',
                               linestyle='--'))
    ax.text(0.3, 1.2, 'hν\n(pump)', ha='center', fontsize=10, color='purple')

    ax.set_title('Kinetic Scheme: Sequential Relaxation', fontsize=14, pad=20)
    path = os.path.join(out_dir, 'kinetic_scheme.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close()


def plot_pipeline_overview():
    """DeepSKAN pipeline diagram."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (1, 1.5, 'TA Data\n(2D map)', '#3498DB'),
        (3.5, 1.5, 'Preprocess\n(256×64)', '#9B59B6'),
        (6, 1.5, 'CNN\n(ResNet)', '#E67E22'),
        (8.5, 1.5, 'Model\nClass', '#27AE60'),
        (11, 1.5, 'GA/TA\nFit', '#E74C3C'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 1.0,
                                        boxstyle="round,pad=0.15",
                                        facecolor=color, alpha=0.2,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.9
        x2 = boxes[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))

    # Labels below
    labels = ['Input', 'Normalize &\ninterpolate', '21 class\nclassification',
              'Kinetic\ntopology', 'Decay times\n+ EADS']
    for i, (x, y, _, _) in enumerate(boxes):
        ax.text(x, 0.4, labels[i], ha='center', va='center', fontsize=8, color='gray')

    # Failure annotation
    ax.annotate('✗ Failed here\n(sim-to-real gap)', xy=(7.2, 2.0),
                fontsize=9, color='red', ha='center',
                bbox=dict(boxstyle='round', facecolor='#FADBD8', edgecolor='red'))

    ax.set_title('DeepSKAN Pipeline', fontsize=13, pad=10)
    path = os.path.join(out_dir, 'pipeline_overview.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close()


def plot_comparison_table():
    """Figure showing CNN vs Direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [
        ['', 'DeepSKAN (CNN → GA)', 'Direct Global Analysis'],
        ['Model selection', 'Automated (but wrong\nfor CBK-D)', 'Manual (correct:\nA→B→C→GS)'],
        ['τ₁', '0.1 ps ✗', '9.4 / 13.8 ps ✓'],
        ['τ₂', '0.0 ps ✗', '91.8 / 173.9 ps ✓'],
        ['τ₃', '—', '>6 ns ✓'],
        ['EADS', 'Not meaningful', 'Physically interpretable'],
        ['Time required', '~30 seconds', '~5 minutes'],
        ['Expertise needed', 'None', 'Some (model known)'],
    ]

    colors = [['#ECF0F1'] * 3]
    for i in range(1, len(table_data)):
        colors.append(['#FDFEFE', '#FADBD8', '#D5F5E3'])

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[0, j].set_text_props(fontweight='bold', fontsize=10)
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Comparison: CNN-Assisted vs Direct Analysis', fontsize=13, pad=20)
    path = os.path.join(out_dir, 'comparison_table.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close()


if __name__ == '__main__':
    plot_kinetic_scheme()
    plot_pipeline_overview()
    plot_comparison_table()
    print("\nAll poster figures generated!")
