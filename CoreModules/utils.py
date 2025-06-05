# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
import pickle
import os
from config import DIRS

def download_nltk_resources():
    """下载NLTK必要资源"""
    resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"下载NLTK资源: {resource}")
            nltk.download(resource)

def set_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def load_data(filepath):
    """加载数据集"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None

def save_preprocessed_data(data, filename):
    """保存预处理后的数据"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"预处理数据已保存到: {filename}")

def load_preprocessed_data(filename):
    """加载预处理后的数据"""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"错误：找不到预处理数据文件 {filename}")
        return None
    
def save_model(model, filename):
    """保存模型"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"模型已保存到: {filename}")

def load_model(filename):
    """加载模型"""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {filename}")
        return None

def get_wordnet_pos(treebank_tag):
    """获取词性的WordNet格式"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # 默认作为名词

if __name__ == "__main__":
    # 测试工具函数
    download_nltk_resources()
    print("NLTK资源下载完成")
    
    # 创建必要的目录
    for dir_name, dir_path in DIRS.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"目录已创建/确认: {dir_path}")