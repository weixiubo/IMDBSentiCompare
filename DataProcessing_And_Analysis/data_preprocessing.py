# data_preprocessing.py
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pickle
import argparse
from CoreModules.utils import download_nltk_resources, load_data, get_wordnet_pos, save_preprocessed_data
from CoreModules.config import DATA_PATH, DIRS, PREPROCESSING

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """文本预处理函数"""
    # 转换为小写
    text = text.lower()

    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 移除数字
    text = re.sub(r'\d+', '', text)

    # 标记化
    tokens = word_tokenize(text)

    # 移除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # 词形还原
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        # 获取词性信息以提高词形还原的准确性
        tagged_tokens = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]

    return ' '.join(tokens)

def prepare_datasets(train_file=None, valid_file=None, test_file=None, 
                     remove_stopwords=None, lemmatize=None, max_features=None,
                     output_file=None):
    """准备数据集，包括预处理和特征提取"""
    print("开始数据预处理...")

    # 使用配置文件中的默认值或传入的参数
    remove_stopwords = PREPROCESSING['remove_stopwords'] if remove_stopwords is None else remove_stopwords
    lemmatize = PREPROCESSING['lemmatize'] if lemmatize is None else lemmatize
    max_features = PREPROCESSING['max_features'] if max_features is None else max_features
    
    # 确保下载了NLTK必要资源
    download_nltk_resources()

    # 加载数据
    train_file = train_file or DATA_PATH['train']
    valid_file = valid_file or DATA_PATH['valid']
    test_file = test_file or DATA_PATH['test']
    
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    test_data = load_data(test_file)
    
    if train_data is None or valid_data is None or test_data is None:
        print("数据加载失败，预处理终止")
        return None

    # 处理训练集
    print("预处理训练集...")
    train_data['processed_text'] = train_data['text'].apply(
        lambda x: preprocess_text(x, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )

    # 处理验证集
    print("预处理验证集...")
    valid_data['processed_text'] = valid_data['text'].apply(
        lambda x: preprocess_text(x, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )

    # 处理测试集
    print("预处理测试集...")
    test_data['processed_text'] = test_data['text'].apply(
        lambda x: preprocess_text(x, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )

    # 创建词袋模型特征
    print("创建词袋模型特征...")
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(train_data['processed_text'])
    X_valid_bow = vectorizer.transform(valid_data['processed_text'])
    X_test_bow = vectorizer.transform(test_data['processed_text'])

    # 创建TF-IDF特征
    print("创建TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['processed_text'])
    X_valid_tfidf = tfidf_vectorizer.transform(valid_data['processed_text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_data['processed_text'])

    # 获取标签
    y_train = train_data['label'].values
    y_valid = valid_data['label'].values
    y_test = test_data['label'].values

    # 保存预处理后的数据
    preprocessed_data = {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'X_train_bow': X_train_bow,
        'X_valid_bow': X_valid_bow,
        'X_test_bow': X_test_bow,
        'X_train_tfidf': X_train_tfidf,
        'X_valid_tfidf': X_valid_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer
    }
    
    # 保存预处理数据
    if output_file:
        save_preprocessed_data(preprocessed_data, output_file)
    else:
        output_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
        save_preprocessed_data(preprocessed_data, output_file)

    return preprocessed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMDB数据集预处理')
    parser.add_argument('--train', type=str, default=DATA_PATH['train'], help='训练集文件路径')
    parser.add_argument('--valid', type=str, default=DATA_PATH['valid'], help='验证集文件路径')
    parser.add_argument('--test', type=str, default=DATA_PATH['test'], help='测试集文件路径')
    parser.add_argument('--no-stopwords', action='store_false', dest='remove_stopwords', help='不移除停用词')
    parser.add_argument('--no-lemmatize', action='store_false', dest='lemmatize', help='不进行词形还原')
    parser.add_argument('--max-features', type=int, default=PREPROCESSING['max_features'], help='最大特征数量')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 运行数据预处理
    prepare_datasets(
        args.train, args.valid, args.test,
        args.remove_stopwords, args.lemmatize, args.max_features,
        args.output
    )