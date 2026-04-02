"""生成框架图"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))

# 环境
env_box = FancyBboxPatch((0.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(env_box)
ax.text(1.5, 5.75, 'Environment\n(Object + Hand)', ha='center', va='center', fontsize=10, weight='bold')

# Pre-Grasp生成器
pg_box = FancyBboxPatch((4, 5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(pg_box)
ax.text(5, 5.75, 'Pre-Grasp\nGenerator', ha='center', va='center', fontsize=10, weight='bold')

# 策略网络
policy_box = FancyBboxPatch((8, 5), 2, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(policy_box)
ax.text(9, 5.75, 'Policy\nNetwork', ha='center', va='center', fontsize=10, weight='bold')

# 奖励计算
reward_box = FancyBboxPatch((4, 2), 2, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(reward_box)
ax.text(5, 2.75, 'Reward\nComputation', ha='center', va='center', fontsize=10, weight='bold')

# 箭头
arrows = [
    ((2.5, 5.75), (4, 5.75)),
    ((6, 5.75), (8, 5.75)),
    ((9, 5), (9, 3.5)),
    ((7, 2.75), (2.5, 2.75)),
    ((1.5, 5), (1.5, 3.5))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.text(3.2, 6.2, 'state', ha='center', fontsize=9, style='italic')
ax.text(7, 6.2, 'action', ha='center', fontsize=9, style='italic')
ax.text(9.5, 4.2, 'next state', ha='center', fontsize=9, style='italic')
ax.text(4.5, 2.2, 'reward', ha='center', fontsize=9, style='italic')

ax.set_xlim(0, 11)
ax.set_ylim(1, 7.5)
ax.axis('off')
ax.set_title('Pre-Grasp Guided RL Framework', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('framework.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: framework.png")
