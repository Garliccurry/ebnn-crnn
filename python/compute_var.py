"""
    File    :compute_var.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Show the difference between float8 quantization and full precision floating-point
"""
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib import font_manager

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data = np.linspace(-1, 1, 10000)
# 将提取的字符串转换为浮点数
print(f"数据长度: {len(data)}")
# print(data)

# 对数据进行截取
data = np.clip(data, -1, 1)
qyf = np.vectorize(utils.flt_to_qtf)
flt = np.vectorize(utils.qtf_to_flt)

mid_data = qyf(data)
poc_data = flt(mid_data)

# 设置字体大小
title_fontsize = 20
label_fontsize = 15


# 绘制密度图
plt.figure(figsize=(10, 5))
density = plt.hist(data, bins=1000, density=True, color='green', alpha=0.5)
plt.plot(density[1][1:], density[0], 'r-')
plt.title('原本数据分布密度图', fontsize=title_fontsize)
plt.xlabel('值', fontsize=label_fontsize)
plt.ylabel('密度', fontsize=label_fontsize)
plt.xlim(-1, 1)  # 设置x轴范围
plt.grid(True)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)

plt.figure(figsize=(10, 5))
density = plt.hist(poc_data, bins=1000, density=True, color='green', alpha=0.5)
plt.plot(density[1][1:], density[0], 'r-')
plt.title('量化数据分布密度图', fontsize=title_fontsize)
plt.xlabel('值', fontsize=label_fontsize)
plt.ylabel('密度', fontsize=label_fontsize)
plt.xlim(-1, 1)  # 设置x轴范围
plt.grid(True)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)


plt.show()
