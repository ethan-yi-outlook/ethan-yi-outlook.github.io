"""生成网络架构对比图"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Diffusion Policy U-Net
ax = axes[0]
layers = ['Input', 'Down1', 'Down2', 'Down3', 'Up3', 'Up2', 'Up1', 'Output']
y_pos = [0, 1, 2, 3, 2.5, 1.5, 0.5, 0]
x_pos = [1, 0.7, 0.4, 0.1, 0.4, 0.7, 1, 1]

for i, (layer, y, x) in enumerate(zip(layers, y_pos, x_pos)):
    color = 'lightblue' if 'Down' in layer else 'lightcoral' if 'Up' in layer else 'lightgreen'
    rect = FancyBboxPatch((x, y), 0.5, 0.3, boxstyle="round,pad=0.02",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+0.25, y+0.15, layer, ha='center', va='center', fontsize=9)

ax.set_xlim(-0.2, 1.8)
ax.set_ylim(-0.5, 3.5)
ax.axis('off')
ax.set_title('Diffusion Policy\n(U-Net Architecture)', fontsize=12, weight='bold')

# ACT Transformer
ax = axes[1]
components = ['Encoder\nLayers', 'Cross\nAttention', 'Decoder\nLayers']
colors = ['lightblue', 'lightyellow', 'lightcoral']
y_positions = [2, 1.5, 1]

for comp, color, y in zip(components, colors, y_positions):
    rect = FancyBboxPatch((0.3, y), 1, 0.4, boxstyle="round,pad=0.02",
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.8, y+0.2, comp, ha='center', va='center', fontsize=10, weight='bold')

ax.set_xlim(0, 1.6)
ax.set_ylim(0.5, 2.8)
ax.axis('off')
ax.set_title('ACT\n(Transformer Architecture)', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('diffusion_network.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('act_network.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: diffusion_network.png and act_network.png")
