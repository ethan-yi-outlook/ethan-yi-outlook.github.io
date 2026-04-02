"""
运行所有图片生成脚本
使用方法: python generate_all_images.py
"""
import subprocess
import sys

scripts = [
    'generate_noise_schedule.py',
    'generate_diffusion_process.py',
    'generate_latent_space.py',
    'generate_attention_map.py'
]

print("开始生成图片...\n")

for script in scripts:
    print(f"运行 {script}...")
    try:
        subprocess.run([sys.executable, script], check=True)
    except Exception as e:
        print(f"错误: {e}")

print("\n完成！请检查生成的图片。")
print("\n注意：以下图片需要手动创建或从论文获取：")
print("- diffusion_policy_framework.png (框架图)")
print("- act_framework.png (框架图)")
print("- diffusion_network.png (网络架构)")
print("- act_network.png (网络架构)")
print("- comparison_table.png (对比图)")
