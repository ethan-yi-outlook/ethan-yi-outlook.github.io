"""生成消融实验图"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
success_rates = [58, 65, 72, 78, 76, 68, 62, 55]

ax.plot(lambdas, success_rates, 'o-', linewidth=2.5, markersize=8, color='steelblue')
ax.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='Optimal λ=0.4')

ax.set_xlabel('Guidance Weight (λ)', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Impact of Guidance Weight', fontsize=14, weight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation.png', dpi=300, bbox_inches='tight')
print("Generated: ablation.png")
