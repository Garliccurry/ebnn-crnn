"""
    Float8 quantization operation
    File    :qtf_ops.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Customize the float8 quantization operation of [-1 ~ 1] for fully connected layers after RNNs

    Bit0    used for symbol bits    ->  "S"
    Bit1-4  used for index bits     ->  "E"
    Bit5-7  used for tail bits      ->  "M"

    Y = -Sign(S)*M**(-E)
"""
import torch

binary_length = 32
E_length = 4
M_length = 3
def flt_to_qtf(num):
    # 自定义8位浮点数量化格式，假设：1位符号位、4位指数位、3(+4)位尾数位 一共8(/12)
    S_bit = '0' if num >= 0 else '1'
    num = abs(num)
    quantify_num = ''
    while num > 0 and len(quantify_num) <= binary_length:
        num *= 2
        if num >= 1:
            quantify_num += '1'
            num -= 1
        else:
            quantify_num += '0'
    quantify_num = quantify_num + "0" * (binary_length - len(quantify_num))  # 补齐32位

    E_num = 0
    while quantify_num[0] != "1" and E_num < 2 ** E_length:
        E_num += 1
        quantify_num = quantify_num[1:]
    E_bit = bin(E_num)[2:]
    E_bit = "0" * (E_length - len(E_bit)) + E_bit
    M_bit = quantify_num[1:4]
    M_bit = M_bit + "0" * (M_length - len(M_bit))
    FLT8 = S_bit + E_bit + M_bit
    return FLT8

def qtf_to_flt(num):
    S_bit = num[0]
    E_bit = num[1:5]
    M_bit = num[5:]
    E_num = int(E_bit,2)
    binary_bit = "0" * E_num + "1" + M_bit
    quantify_num = sum(int(bit) * 2**(-i-1) for i, bit in enumerate(binary_bit))
    return quantify_num if S_bit == "0" else -quantify_num

def flt_to_qtf_tensor(tensor):
    binary_tensor = torch.zeros_like(tensor, dtype=torch.int32)
    for i in range(tensor.numel()):
        binary_tensor.view(-1)[i] = int(flt_to_qtf(tensor.view(-1)[i].item()), 2)
    return binary_tensor

def qtf_to_flt_tensor(tensor):
    float_tensor = torch.zeros_like(tensor, dtype=torch.float32)
    for i in range(tensor.numel()):
        float_tensor.view(-1)[i] = qtf_to_flt(bin(tensor.view(-1)[i].item())[2:].zfill(8))
    return float_tensor

def test_quantify():
    a = -0.00219731234
    b = flt_to_qtf(a)
    # c = "01001101"
    d = qtf_to_flt(b)
    print(f"原始数据：{a} 量化bit：{b}")
    print(f"量化bit：{b} 复原数据{d}")

if __name__ == "__main__":
    test_quantify()