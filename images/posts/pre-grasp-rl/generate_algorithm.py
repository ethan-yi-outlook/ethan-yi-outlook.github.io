"""生成算法流程图"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 10))

steps = [
    ('Initialize Policy', 0.5, 9, 'lightblue'),
    ('Generate Pre-Grasp', 0.5, 7.5, 'lightyellow'),
    ('Collect Trajectories', 0.5, 6, 'lightgreen'),
    ('Compute Rewards', 0.5, 4.5, 'lightcoral'),
    ('Update Policy', 0.5, 3, 'plum'),
    ('Decay λ', 0.5, 1.5, 'lightgray')
]

for text, x, y, color in steps:
    box = FancyBboxPatch((x, y), 2, 0.8, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+1, y+0.4, text, ha='center', va='center', fontsize=10, weight='bold')

# 箭头
for i in range(len(steps)-1):
    start_y = steps[i][2]
    end_y = steps[i+1][2] + 0.8
    arrow = FancyArrowPatch((1.5, start_y), (1.5, end_y), arrowstyle='->',
                           mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)

# 循环箭头
loop_arrow = FancyArrowPatch((2.5, 1.5), (3.5, 9), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='red',
                            connectionstyle="arc3,rad=.5")
ax.add_patch(loop_arrow)
ax.text(3.8, 5, 'Loop', ha='center', fontsize=10, style='italic', color='red')

ax.set_xlim(0, 4.5)
ax.set_ylim(0.5, 10)
ax.axis('off')
ax.set_title('Training Algorithm Flow', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('algorithm.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: algorithm.png")
