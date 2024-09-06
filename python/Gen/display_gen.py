"""
    File    :display_gen.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Display the ideal dataset and save the actual data received through the serial port
"""
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import serial
import winsound


# 串口配置
SERIAL_PORT = 'COM3'  # 修改为你的串口名称
BAUD_RATE = 115200

# 初始化串口连接
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

times = 6
# 图片文件夹路径
image_folder = '../data/chinese/size40font3shape25050'  # 修改为你的图片文件夹路径
save_path = f'../data/chinese/{times}'
# 获取文件夹中所有.png图片的文件名列表
png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 排序文件名列表，确保按照你想要的顺序播放
png_files.sort()

# 索引变量，用于跟踪当前显示的图片
current_index = 0
bias = 1500 * times

def receive_serial_data():
    # 接收串口数据
    data = ser.read(12500)  # 250*50 = 12500 bytes
    if len(data) == 12500:
        # 将数据转换为numpy数组
        print(len(data))
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((50, 250)) - 48
        # 将数据转换为图像
        img = Image.fromarray(img_array * 255)  # 将0/1转换为黑白图像
        print(img_array)
        # 保存图像
        file_name = ("%05d" % (current_index + bias - 1)) + ".png"  # 生成文件名
        img.save(os.path.join(save_path, file_name))
        # img.save(f'{current_index + bias}.png')

def show_next_image():
    global current_index
    if current_index < len(png_files):
        # 打开当前图片
        img_path = os.path.join(image_folder, png_files[current_index])
        img = Image.open(img_path).resize((400, 80))
        photo = ImageTk.PhotoImage(img)

        # 更新标签中的图片
        label.config(image=photo)
        label.image = photo  # 保持对图片的引用，防止被垃圾回收

        # 设置延迟1秒后通过串口发送数据
        root.after(200, send_serial_command)

        # 等待一秒后接收数据并生成图像
        root.after(1000, receive_serial_data)

        # 更新索引以播放下一张图片
        current_index += 1

        # 更新进度条
        progress_var.set((current_index / len(png_files)) * 100)

        # 等待两秒后切换到下一张图片
        root.after(1200, show_next_image)  # 总共3秒的延迟
    else:
        start_button.config(state=tk.NORMAL)  # 恢复开始按钮
        winsound.Beep(1000, 1000)  # 播放提示音，频率1000Hz，持续500毫秒

def send_serial_command():
    # 通过串口发送数据
    ser.write(b'\x31')

def start_slideshow():
    global current_index
    current_index = 0  # 重置索引
    # 启动图片播放
    show_next_image()
    start_button.config(state=tk.DISABLED)  # 禁用开始按钮

def on_closing():
    ser.close()
    root.destroy()

# 创建主窗口
root = tk.Tk()
root.title("Image Viewer")

# 将窗口最大化
root.state('zoomed')

# 创建一个标签来显示图片
label = tk.Label(root)
label.pack(expand=True)

# 创建一个按钮来启动图片播放
start_button = tk.Button(root, text="开始播放", command=start_slideshow)
start_button.pack(pady=20)

# 创建一个进度条
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=20, fill=tk.X)

# 当窗口关闭时，关闭串口连接
root.protocol("WM_DELETE_WINDOW", on_closing)

# 运行主循环
root.mainloop()
