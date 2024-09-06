"""
    File    :generate_data.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Generate an ideal dataset, but this work uses a dataset reprocessed by cameras
"""
import os
from PIL import Image, ImageDraw, ImageFont
import random

# 创建保存图片的目录
font_paths = [
    "font/SimSun.ttf",
    "font/KaiTi.ttf",
    "font/SimHei.ttf"
]
pic_length = 50
pic_width = 250
size_min = 40
size_max = 40

if size_max == size_min:
    output_dir = f"..//data//chinese//size{size_max}font3shape{pic_width}{pic_length}"
else:
    output_dir = f"..//data//chinese//size{size_min}{size_max}font3shape{pic_width}{pic_length}"

os.makedirs(output_dir, exist_ok=True)

file_path = 'filtered_output.txt'  # 替换成你的txt文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


# 生成中文字符（按顺序）
def generate_sequential_chinese_text(index):
    return lines[index % len(lines)].strip()

def generate_bold_text(draw, text, position, font):
    # 在多个位置绘制文本，模拟加粗效果
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for offset in offsets:
        pos = (position[0] + offset[0], position[1] + offset[1])
        draw.text(pos, text, font=font, fill=(0, 0, 0))


# 生成图片
def generate_image_with_text(text, font_path, output_path, is_bold):
    # 随机灰色背景
    gray_value = random.randint(220, 255)
    image = Image.new('RGB', (pic_width, pic_length), color=(gray_value, gray_value, gray_value))
    draw = ImageDraw.Draw(image)

    font_size = size_min  # 使用固定的字体大小
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # 获取文本边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 计算文本位置
    text_x = (pic_width - text_width) / 2
    text_y = (pic_length - text_height) / 2

    # 绘制文本
    if is_bold != 2:
        position = (text_x, text_y)
        generate_bold_text(draw, text, position, font)
    else:
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    # 保存图片
    image.save(output_path)


label_file = os.path.join(output_dir, 'labels.txt')
# 生成图片
with open(label_file, 'w', encoding='utf-8') as f:
    image_count = 0  # 图片计数器

    for font_index, font_path in enumerate(font_paths):
        print(font_index)
        for i in range(500):  # 每种字体生成500张图片
            text = generate_sequential_chinese_text(image_count)
            file_name = ("%05d" % (500 * font_index + i)) + ".png"  # 生成文件名
            output_path = os.path.join(output_dir, file_name)

            generate_image_with_text(text, font_path, output_path,font_index)

            # 写入标签文件
            f.write(f"{file_name},{text}\n")

            image_count += 1

print("图片生成完毕！")
