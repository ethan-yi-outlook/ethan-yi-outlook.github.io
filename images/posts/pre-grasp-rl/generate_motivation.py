"""生成Pre-Grasp引导RL的示意图"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))

# 物体
object_circle = Circle((2, 3), 0.4, color='orange', alpha=0.7, linewidth=2, edgecolor='black')
ax.add_patch(object_circle)
ax.text(2, 3, 'Object', ha='center', va='center', fontsize=10, weight='bold')

# Pre-Grasp姿态
pregrasp_box = FancyBboxPatch((4, 2.5), 1.2, 1, boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(pregrasp_box)
ax.text(4.6, 3, 'Pre-Grasp\nPose', ha='center', va='center', fontsize=10, weight='bold')

# RL策略
rl_box = FancyBboxPatch((7, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(rl_box)
ax.text(7.75, 3, 'RL Policy', ha='center', va='center', fontsize=11, weight='bold')

# 最终抓取
grasp_box = FancyBboxPatch((10, 2.5), 1.2, 1, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(grasp_box)
ax.text(10.6, 3, 'Grasp\nSuccess', ha='center', va='center', fontsize=10, weight='bold')

# 箭头
arrows = [
    ((2.4, 3), (4, 3), 'Guidance'),
    ((5.2, 3), (7, 3), 'Learning'),
    ((8.5, 3), (10, 3), 'Execute')
]

for start, end, label in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    mid_x = (start[0] + end[0]) / 2
    ax.text(mid_x, 3.8, label, ha='center', fontsize=9, style='italic')

ax.set_xlim(1, 12)
ax.set_ylim(2, 4.5)
ax.axis('off')
ax.set_title('Pre-Grasp Guided Reinforcement Learning', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('motivation.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: motivation.png")
