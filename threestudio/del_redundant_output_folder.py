import os
import shutil

system_name = 'best_dreamer'  # 修改这里就好
output_folder = './outputs'

system_pth = os.path.join(output_folder, system_name)

# 确保 system_pth 存在
if not os.path.exists(system_pth):
    print(f"路径 {system_pth} 不存在")
    exit()

for exp_name in sorted(os.listdir(system_pth)):
    exp_path = os.path.join(system_pth, exp_name)

    if not os.path.isdir(exp_path):
        continue

    ckpt_folder = os.path.join(exp_path, 'ckpts')

    # 检查 ckpts 文件夹是否存在
    if os.path.exists(ckpt_folder):
        ckpts = os.listdir(ckpt_folder)
        # 检查 ckpts 文件夹中是否有 last.ckpt
        if 'last.ckpt' not in ckpts:
            # 删除实验文件夹
            print(f"删除文件夹: {exp_path} (原因: 缺少 last.ckpt)")
            shutil.rmtree(exp_path, ignore_errors=True)  # 这里打断点
        else:
            print(f"==========保留文件夹: {exp_path}=========")  # 这里打断点

    else:
        # 删除实验文件夹（没有 ckpt 文件夹）
        print(f"删除文件夹: {exp_path} (原因: 没有 ckpt 文件夹)")
        shutil.rmtree(exp_path, ignore_errors=True)  # 这里打断点
