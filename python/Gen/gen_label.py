"""
    File    :gen_label.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Ideal dataset comes with tags and can be saved as txt;
             This file is used to label the data actually collected by the camera
"""
import re

def process_labels(input_file, output_file, cycles):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for cycle in range(cycles):
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    label = parts[1]
                    # # 检查标签中是否包含任何英文字母或数字
                    # if re.search(r'[a-zA-Z0-9]', label):
                    #     continue
                    image_number = int(parts[0].split('.')[0]) + cycle * 1500
                    new_label = f'{image_number:05d}.png,{label}\n'
                    f.write(new_label)

file_name = '../data/chinese/labels without26.txt'
target_name = '../data/chinese/labels without26 Q1.txt'
process_labels(file_name, target_name, 1)
