# core/util.py
import os
import random
import numpy as np

def seed_everything(seed: int):
    """
    设置全局随机种子，确保实验结果完全可复现。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def softmax(x, axis=-1):
    """
    带有防溢出保护的标准 Softmax 概率归一化函数。
    """
    x = x - np.max(x, axis=axis, keepdims=True)   # 减去最大值防止指数爆炸溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)