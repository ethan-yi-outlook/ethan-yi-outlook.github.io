"""生成ACT框架图"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))

# 观测
obs_box = FancyBboxPatch((0.5, 3), 1.5, 1, boxstyle="round,pad=0.1",
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(obs_box)
ax.text(1.25, 3.5, 'Observation', ha='center', va='center', fontsize=11, weight='bold')

# 编码器
encoder_box = FancyBboxPatch((3, 3), 1.5, 1, boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(encoder_box)
ax.text(3.75, 3.5, 'Encoder', ha='center', va='center', fontsize=11, weight='bold')

# 潜在变量 (训练)
z_train_box = FancyBboxPatch((5.5, 5), 1.2, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(z_train_box)
ax.text(6.1, 5.4, 'z ~ q(z|a,o)', ha='center', va='center', fontsize=9)

# 潜在变量 (推理)
z_infer_box = FancyBboxPatch((5.5, 1.5), 1.2, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(z_infer_box)
ax.text(6.1, 1.9, 'z ~ N(0,I)', ha='center', va='center', fontsize=9)

# 解码器
decoder_box = FancyBboxPatch((8, 3), 1.8, 1, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(decoder_box)
ax.text(8.9, 3.5, 'Decoder\n(Transformer)', ha='center', va='center', fontsize=11, weight='bold')

# 动作输出
action_box = FancyBboxPatch((10.5, 3), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(action_box)
ax.text(11.25, 3.5, 'Action\nChunk', ha='center', va='center', fontsize=11, weight='bold')

# 箭头
arrows = [
    ((2, 3.5), (3, 3.5)),
    ((4.5, 3.5), (5.5, 3.5)),
    ((6.7, 5.4), (8, 4)),
    ((6.7, 1.9), (8, 3)),
    ((9.8, 3.5), (10.5, 3.5))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# 标注
ax.text(6.1, 6.2, 'Training', ha='center', fontsize=10, weight='bold', color='red')
ax.text(6.1, 0.8, 'Inference', ha='center', fontsize=10, weight='bold', color='blue')

ax.set_xlim(0, 12.5)
ax.set_ylim(0.5, 6.5)
ax.axis('off')
ax.set_title('ACT (CVAE) Framework', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('act_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: act_framework.png")
