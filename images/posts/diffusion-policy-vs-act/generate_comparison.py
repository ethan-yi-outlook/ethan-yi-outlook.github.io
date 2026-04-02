"""生成对比表格图"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))

# 表格数据
data = [
    ['Aspect', 'Diffusion Policy', 'ACT'],
    ['Model Type', 'Denoising Diffusion', 'CVAE'],
    ['Architecture', 'U-Net/Transformer', 'Transformer Enc-Dec'],
    ['Inference', 'Multi-step (10-100)', 'Single forward pass'],
    ['Latency', '50-200ms', '20-50ms'],
    ['Multimodal', 'Implicit (sampling)', 'Explicit (latent z)'],
    ['Training', 'Stable', 'KL collapse risk'],
    ['Best For', 'Accuracy & multimodal', 'Speed & real-time']
]

colors = [['lightgray']*3] + [['white', 'lightblue', 'lightcoral']]*7

table = ax.table(cellText=data, cellColours=colors, cellLoc='center',
                loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(len(data)):
    for j in range(3):
        cell = table[(i, j)]
        if i == 0:
            cell.set_text_props(weight='bold', fontsize=11)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

ax.axis('off')
ax.set_title('Diffusion Policy vs ACT Comparison', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: comparison_table.png")
