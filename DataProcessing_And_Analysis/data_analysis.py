# data_analysis.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from tqdm import tqdm
import math
import os
import argparse
from CoreModules.utils import download_nltk_resources, load_data, set_matplotlib_chinese
from CoreModules.config import DATA_PATH, DIRS

def analyze_data(train_file=None, valid_file=None, test_file=None, save_results=True):
    """对IMDB数据集进行全面分析"""
    print("开始数据分析...")
    
    # 确保下载了NLTK必要资源
    download_nltk_resources()
    
    # 设置matplotlib支持中文
    set_matplotlib_chinese()
    
    # 加载数据
    train_file = train_file or DATA_PATH['train']
    valid_file = valid_file or DATA_PATH['valid']
    test_file = test_file or DATA_PATH['test']
    
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    test_data = load_data(test_file)
    
    if train_data is None or valid_data is None or test_data is None:
        print("数据加载失败，分析终止")
        return None
    
    # 创建图表目录
    figures_dir = DIRS['figures']
    os.makedirs(figures_dir, exist_ok=True)
    
    # 打印数据集大小
    print(f"训练集大小: {train_data.shape[0]} 条评论")
    print(f"开发集大小: {valid_data.shape[0]} 条评论")
    print(f"测试集大小: {test_data.shape[0]} 条评论")
    
    # 分析情感分布
    pos_count = sum(train_data['label'] == 1)
    neg_count = sum(train_data['label'] == 0)
    print(f"训练集中正向情感评论数量: {pos_count}")
    print(f"训练集中负向情感评论数量: {neg_count}")
    
    # 可视化情感分布
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train_data, x='label')
    plt.title("训练集中正负向情感分布")
    plt.xlabel("情感标签 (0=负向, 1=正向)")
    plt.ylabel("评论数量")
    plt.savefig(os.path.join(figures_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 分析评论长度
    train_data['length'] = train_data['text'].apply(len)
    pos_lengths = train_data[train_data['label'] == 1]['length']
    neg_lengths = train_data[train_data['label'] == 0]['length']
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_lengths, alpha=0.5, label='正向', bins=50)
    plt.hist(neg_lengths, alpha=0.5, label='负向', bins=50)
    plt.title("不同情感评论的长度分布")
    plt.xlabel("评论长度 (字符数)")
    plt.ylabel("评论数量")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'review_length_distribution.png'))
    plt.close()
    
    # 标记化并统计词频
    print("正在分析词频信息...")
    
    # 初始化停用词
    stop_words = set(stopwords.words('english'))
    
    # 处理每条评论并统计词频
    pos_words = []
    neg_words = []
    
    for index, row in tqdm(train_data.iterrows(), total=len(train_data), desc="处理评论"):
        text = row['text'].lower()
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # 去除标点和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 标记化
        tokens = word_tokenize(text)
        # 移除停用词
        filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
        
        if row['label'] == 1:
            pos_words.extend(filtered_tokens)
        else:
            neg_words.extend(filtered_tokens)
    
    # 统计词频
    pos_word_freq = Counter(pos_words)
    neg_word_freq = Counter(neg_words)
    
    # 计算PMI
    print("计算PMI值...")
    
    # 合并所有词
    all_words = pos_words + neg_words
    total_words = len(all_words)
    
    # 计算每个词的PMI
    word_pmi = {}
    
    # 计算正向情感的PMI
    for word in tqdm(set(all_words), desc="计算PMI"):
        pos_count = pos_word_freq.get(word, 0)
        neg_count = neg_word_freq.get(word, 0)
        total_count = pos_count + neg_count
        
        # 如果词频太低则跳过
        if total_count < 20:
            continue
            
        # P(word)
        p_word = total_count / total_words
        
        # P(pos)
        p_pos = len(pos_words) / total_words
        
        # P(word|pos) 
        p_word_given_pos = pos_count / len(pos_words) if pos_count > 0 else 1e-10
        
        # PMI(word, pos) = log(P(word|pos)/P(word))
        if p_word_given_pos > 0:
            pmi = math.log2(p_word_given_pos / p_word)
            word_pmi[word] = pmi
    
    # 按PMI值排序
    sorted_pmi = sorted(word_pmi.items(), key=lambda x: x[1], reverse=True)
    
    # 获取正向情感PMI最高的10个词
    pos_pmi_top10 = sorted_pmi[:10]
    
    # 获取负向情感PMI最低的10个词(负PMI值表示与负向情感相关)
    neg_pmi_top10 = sorted_pmi[-10:]
    
    # 打印结果
    print("\n正向情感频率前10的词:")
    for word, count in pos_word_freq.most_common(10):
        print(f"{word}: {count}")
    
    print("\n负向情感频率前10的词:")
    for word, count in neg_word_freq.most_common(10):
        print(f"{word}: {count}")
    
    print("\n正向情感PMI前10的词:")
    for word, pmi in pos_pmi_top10:
        print(f"{word}: {pmi:.4f}")
    
    print("\n负向情感PMI前10的词:")
    for word, pmi in reversed(neg_pmi_top10):
        print(f"{word}: {pmi:.4f}")
    
    # 进行词性分析
    print("\n进行词性分析...")
    
    # 从所有评论中抽取一部分进行词性标注
    sample_size = min(1000, len(train_data))
    sample_data = train_data.sample(sample_size)
    
    all_pos_tags = []
    
    for text in tqdm(sample_data['text'], desc="词性标注"):
        tokens = word_tokenize(text.lower())
        # 词性标注
        pos_tags = nltk.pos_tag(tokens)
        all_pos_tags.extend([tag for _, tag in pos_tags])
    
    # 统计词性分布
    pos_tag_freq = Counter(all_pos_tags)
    
    # 打印结果
    print("\n词性分布:")
    for tag, count in pos_tag_freq.most_common(10):
        print(f"{tag}: {count}")
    
    # 可视化词性分布
    plt.figure(figsize=(12, 6))
    top_pos_tags = dict(pos_tag_freq.most_common(15))
    sns.barplot(x=list(top_pos_tags.keys()), y=list(top_pos_tags.values()))
    plt.title("Top 15 词性分布")
    plt.xlabel("词性标签")
    plt.ylabel("数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pos_tag_distribution.png'))
    plt.close()
    
    # 可视化高频词
    plt.figure(figsize=(12, 6))
    top_pos_words = dict(pos_word_freq.most_common(20))
    sns.barplot(x=list(top_pos_words.keys()), y=list(top_pos_words.values()))
    plt.title("正向情感高频词 Top 20")
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pos_top_words.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    top_neg_words = dict(neg_word_freq.most_common(20))
    sns.barplot(x=list(top_neg_words.keys()), y=list(top_neg_words.values()))
    plt.title("负向情感高频词 Top 20")
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'neg_top_words.png'))
    plt.close()
    
    # 将分析结果保存到文件
    if save_results:
        results = {
            'train_data': train_data,
            'valid_data': valid_data,
            'test_data': test_data,
            'pos_word_freq': pos_word_freq,
            'neg_word_freq': neg_word_freq,
            'pos_pmi_top10': pos_pmi_top10,
            'neg_pmi_top10': neg_pmi_top10
        }
        
        # 保存分析结果
        results_file = os.path.join(DIRS['results'], 'analysis_results.pkl')
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"分析结果已保存到: {results_file}")
        except Exception as e:
            print(f"保存分析结果失败: {str(e)}")
    
    return {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'pos_word_freq': pos_word_freq,
        'neg_word_freq': neg_word_freq,
        'pos_pmi_top10': pos_pmi_top10,
        'neg_pmi_top10': neg_pmi_top10
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMDB数据集分析')
    parser.add_argument('--train', type=str, default=DATA_PATH['train'], help='训练集文件路径')
    parser.add_argument('--valid', type=str, default=DATA_PATH['valid'], help='验证集文件路径')
    parser.add_argument('--test', type=str, default=DATA_PATH['test'], help='测试集文件路径')
    parser.add_argument('--no-save', action='store_false', dest='save_results', help='不保存分析结果')
    
    args = parser.parse_args()
    
    # 运行数据分析
    analyze_data(args.train, args.valid, args.test, args.save_results)