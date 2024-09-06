import torch
from bnn_layer import BLinear, BConv2d, QLinear, QConv2d, HLinear,  BLSTMCell
from qtf_ops import flt_to_qtf, qtf_to_flt, flt_to_qtf_tensor, qtf_to_flt_tensor
import numpy as np


B_Clayers = ["conv"]
B_layers = ["linear1","rnn"]
F_layers = ["bias","bn","linear2"]
Q_layers = []


def binarize_weights(model):
    # 遍历模型的所有模块
    for module in model.modules():
        if isinstance(module, (BConv2d, QConv2d)):
            # 获取卷积层的权重
            with torch.no_grad():
                weights = module.conv.weight.data
                weights[weights >= 0] = 1
                weights[weights < 0] = -1
                module.conv.weight.data = weights
        if isinstance(module, (BLinear, QLinear)) :
             with torch.no_grad():
                weights = module.linear.weight.data
                weights[weights >= 0] = 1
                weights[weights < 0] = -1
                module.linear.weight.data = weights
        if isinstance(module, BLSTMCell):
            with torch.no_grad():
                W_ih = module.W_ih.data
                W_hh = module.W_hh.data
                W_ih[W_ih >= 0] = 1
                W_ih[W_ih < 0] = -1
                W_hh[W_hh >= 0] = 1
                W_hh[W_hh < 0] = -1
                module.W_ih.data = W_ih
                module.W_hh.data = W_hh

def quantify_weights(model):
    for module in model.modules():
        if isinstance(module, HLinear):
            with torch.no_grad():
                weights = module.linear.weight.data
                compressed_weights = flt_to_qtf_tensor(weights)
                decompressed_weights = qtf_to_flt_tensor(compressed_weights)
                module.linear.weight.data = decompressed_weights


def float_to_binary_int(value):
    """Convert float to binary int representation."""
    if value > 0:
        return 1
    else:
        return 0


def binary_string_to_decimal(binary_str):
    """Convert binary string to decimal integer."""
    return int(binary_str, 2)


def save_weights_to_h_file(model, output_path):
    with open(output_path, 'w') as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(name.startswith(layer) for layer in B_Clayers):
                    print("B_Clayers", name)
                    layer_name = name.replace('.', '_')
                    param_data = param.cpu().detach().numpy()
                    O_C, I_C, H, W = param_data.shape
                    k_size = I_C * H * W
                    print(k_size)
                    if k_size % 8 == 0:
                        pad_bit = 0
                    else:
                        pad_bit = 8 - k_size % 8
                    param_data = param_data.flatten()
                    new_length = len(param_data) + O_C * I_C * pad_bit
                    new_param_data = np.zeros(new_length, dtype=float)
                    i = 0
                    j = 0
                    while i < len(param_data):
                        # 每隔 k_sizex 位插入 pad_bit 个值为 1 的元素
                        new_param_data[j:j + k_size] = param_data[i:i + k_size]
                        j += k_size
                        i += k_size
                        if i < len(param_data) + 1:  # 在数组范围内插入 1
                            new_param_data[j:j + pad_bit] = 1
                            j += pad_bit
                    binary_data = [float_to_binary_int(x) for x in new_param_data]
                    binary_str = ''.join(map(str, binary_data))
                    decimal_values = [binary_string_to_decimal(binary_str[i:i + 8]) for i in
                                      range(0, len(binary_str), 8)]
                    num = len(decimal_values)
                    decimal_values_str = ', '.join(map(str, decimal_values))
                    f.write(f'uint8_t {layer_name}[{num}] = {{{decimal_values_str}}};\n')
                if any(name.startswith(layer) for layer in B_layers):
                    print("B_layers", name)
                    layer_name = name.replace('.', '_')
                    param_data = param.cpu().detach().numpy().flatten()
                    binary_data = [float_to_binary_int(x) for x in param_data]
                    binary_str = ''.join(map(str, binary_data))
                    decimal_values = [binary_string_to_decimal(binary_str[i:i + 8]) for i in
                                      range(0, len(binary_str), 8)]
                    num = len(decimal_values)
                    decimal_values_str = ', '.join(map(str, decimal_values))
                    f.write(f'uint8_t {layer_name}[{num}] = {{{decimal_values_str}}};\n')
                elif any(name.startswith(layer) for layer in F_layers):
                    print("F_layers", name)
                    layer_name = name.replace('.', '_')
                    param_data = param.cpu().detach().numpy().flatten()
                    num = len(param_data)
                    values_str = ', '.join(map(str, param_data))
                    f.write(f'float {layer_name}[{num}] = {{{values_str}}};\n')
                if any(name.startswith(layer) for layer in Q_layers):
                    print("Q_layers", name)
                    layer_name = name.replace('.', '_')
                    param_data = param.cpu().detach().numpy().flatten()
                    compressed_weights = [flt_to_qtf(x) for x in param_data]
                    binary_str = ''.join(compressed_weights)
                    actual_values = [binary_string_to_decimal(binary_str[i:i + 8]) for i in
                                      range(0, len(binary_str), 8)]
                    num = len(actual_values)
                    actual_values_str = ', '.join(map(str, actual_values))
                    f.write(f'uint8_t {layer_name}[{num}] = {{{actual_values_str}}};\n')
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # 如果当前模块是Batch Normalization层
                print(name)
                mean = module.running_mean.cpu().detach().numpy().flatten()
                std = torch.sqrt(module.running_var + module.eps).cpu().detach().numpy().flatten()
                num = len(mean)
                values_str = ', '.join(map(str, mean))
                f.write(f'float {name}_mean[{num}] = {{{values_str}}};\n')
                values_str = ', '.join(map(str, std))
                f.write(f'float {name}_std[{num}] = {{{values_str}}};\n')
            if isinstance(module, torch.nn.BatchNorm1d):
                # 如果当前模块是Batch Normalization层
                print(name)
                mean = module.running_mean.cpu().detach().numpy().flatten()
                std = torch.sqrt(module.running_var + module.eps).cpu().detach().numpy().flatten()
                num = len(mean)
                values_str = ', '.join(map(str, mean))
                f.write(f'float {name}_mean[{num}] = {{{values_str}}};\n')
                values_str = ', '.join(map(str, std))
                f.write(f'float {name}_std[{num}] = {{{values_str}}};\n')



def save_data_to_h_file(np_picture, output_path):
    with open(output_path, 'w') as f:
        pic = np_picture.flatten()
        pic = pic/255.0
        num = len(pic)
        values_str = ', '.join(map(str, pic))
        f.write(f'float pic_data[{num}] = {{{values_str}}};\n')





