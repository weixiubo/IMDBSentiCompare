# ml_models.py
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
import pickle
import argparse
from tqdm import tqdm
from CoreModules.utils import save_model, load_preprocessed_data
from CoreModules.config import DIRS

def train_naive_bayes(preprocessed_data=None, preprocessed_file=None, output_file=None):
    """训练朴素贝叶斯模型"""
    print("训练朴素贝叶斯模型...")
    
    # 如果没有提供预处理数据，尝试加载
    if preprocessed_data is None:
        if preprocessed_file:
            preprocessed_data = load_preprocessed_data(preprocessed_file)
        else:
            preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
            preprocessed_data = load_preprocessed_data(preprocessed_file)
            
        if preprocessed_data is None:
            print("预处理数据加载失败，朴素贝叶斯模型训练终止")
            return None
    
    # 获取训练数据
    X_train = preprocessed_data['X_train_bow']
    y_train = preprocessed_data['y_train']
    
    # 获取验证数据
    X_valid = preprocessed_data['X_valid_bow']
    y_valid = preprocessed_data['y_valid']
    
    # 训练模型
    start_time = time.time()
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    
    # 在验证集上评估
    y_valid_pred = nb_classifier.predict(X_valid)
    
    # 计算评估指标
    accuracy = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    
    print(f"朴素贝叶斯模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"验证集上的准确率: {accuracy:.4f}")
    print(f"验证集上的精确率: {precision:.4f}")
    print(f"验证集上的召回率: {recall:.4f}")
    print(f"验证集上的F1分数: {f1:.4f}")
    
    # 保存模型
    if output_file:
        save_model(nb_classifier, output_file)
    else:
        output_file = os.path.join(DIRS['models'], 'naive_bayes.pkl')
        save_model(nb_classifier, output_file)
    
    return nb_classifier

def train_logistic_regression(preprocessed_data=None, preprocessed_file=None, feature_type='bow', C=1.0, max_iter=100, output_file=None):
    """训练逻辑回归模型
    
    参数:
    - feature_type: 'bow'或'tfidf'，选择使用词袋模型或TF-IDF特征
    - C: 正则化强度，默认为1.0
    - max_iter: 最大迭代次数，默认为100
    """
    print(f"训练逻辑回归模型 (特征类型: {feature_type}, C={C})...")
    
    # 如果没有提供预处理数据，尝试加载
    if preprocessed_data is None:
        if preprocessed_file:
            preprocessed_data = load_preprocessed_data(preprocessed_file)
        else:
            preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
            preprocessed_data = load_preprocessed_data(preprocessed_file)
            
        if preprocessed_data is None:
            print("预处理数据加载失败，逻辑回归模型训练终止")
            return None
    
    # 根据特征类型选择训练数据
    if feature_type == 'bow':
        X_train = preprocessed_data['X_train_bow']
        X_valid = preprocessed_data['X_valid_bow']
    else:  # tfidf
        X_train = preprocessed_data['X_train_tfidf']
        X_valid = preprocessed_data['X_valid_tfidf']
    
    y_train = preprocessed_data['y_train']
    y_valid = preprocessed_data['y_valid']
    
    # 训练模型
    start_time = time.time()
    lr_classifier = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    lr_classifier.fit(X_train, y_train)
    
    # 在验证集上评估
    y_valid_pred = lr_classifier.predict(X_valid)
    
    # 计算评估指标
    accuracy = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    
    print(f"逻辑回归模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"验证集上的准确率: {accuracy:.4f}")
    print(f"验证集上的精确率: {precision:.4f}")
    print(f"验证集上的召回率: {recall:.4f}")
    print(f"验证集上的F1分数: {f1:.4f}")
    
    # 保存模型
    if output_file:
        save_model(lr_classifier, output_file)
    else:
        model_name = f"logistic_regression_{feature_type}_C{C}.pkl"
        output_file = os.path.join(DIRS['models'], model_name)
        save_model(lr_classifier, output_file)
    
    return lr_classifier

def train_ml_models(preprocessed_data=None, preprocessed_file=None, output_dir=None):
    """训练所有机器学习模型"""
    print("开始训练机器学习模型...")
    
    # 如果没有提供预处理数据，尝试加载
    if preprocessed_data is None:
        if preprocessed_file:
            preprocessed_data = load_preprocessed_data(preprocessed_file)
        else:
            preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
            preprocessed_data = load_preprocessed_data(preprocessed_file)
            
        if preprocessed_data is None:
            print("预处理数据加载失败，机器学习模型训练终止")
            return None
    
    output_dir = output_dir or DIRS['models']
    os.makedirs(output_dir, exist_ok=True)
    
    # 朴素贝叶斯模型
    nb_model = train_naive_bayes(preprocessed_data, output_file=os.path.join(output_dir, 'naive_bayes.pkl'))
    
    # 逻辑回归模型 (BOW特征)
    lr_bow_model = train_logistic_regression(preprocessed_data, feature_type='bow', C=1.0, 
                                           output_file=os.path.join(output_dir, 'logistic_regression_bow.pkl'))
    
    # 逻辑回归模型 (TF-IDF特征)
    lr_tfidf_model = train_logistic_regression(preprocessed_data, feature_type='tfidf', C=1.0,
                                             output_file=os.path.join(output_dir, 'logistic_regression_tfidf.pkl'))
    
    # 返回所有模型
    return {
        'naive_bayes': nb_model,
        'logistic_regression_bow': lr_bow_model,
        'logistic_regression_tfidf': lr_tfidf_model
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练机器学习模型')
    parser.add_argument('--preprocessed', type=str, default=None, help='预处理数据文件路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出模型目录')
    parser.add_argument('--model', type=str, choices=['all', 'nb', 'lr-bow', 'lr-tfidf'], default='all', 
                        help='选择要训练的模型 (all: 全部, nb: 朴素贝叶斯, lr-bow: 词袋逻辑回归, lr-tfidf: TF-IDF逻辑回归)')
    parser.add_argument('--C', type=float, default=1.0, help='逻辑回归正则化强度')
    parser.add_argument('--max-iter', type=int, default=100, help='逻辑回归最大迭代次数')
    
    args = parser.parse_args()
    
    # 加载预处理数据
    if args.preprocessed:
        preprocessed_data = load_preprocessed_data(args.preprocessed)
    else:
        preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
        preprocessed_data = load_preprocessed_data(preprocessed_file)
        
    if preprocessed_data is None:
        print("预处理数据加载失败，训练终止")
    else:
        # 训练模型
        if args.model == 'all':
            train_ml_models(preprocessed_data, output_dir=args.output_dir)
        elif args.model == 'nb':
            train_naive_bayes(preprocessed_data, 
                             output_file=os.path.join(args.output_dir or DIRS['models'], 'naive_bayes.pkl'))
        elif args.model == 'lr-bow':
            train_logistic_regression(preprocessed_data, feature_type='bow', C=args.C, max_iter=args.max_iter,
                                     output_file=os.path.join(args.output_dir or DIRS['models'], 'logistic_regression_bow.pkl'))
        elif args.model == 'lr-tfidf':
            train_logistic_regression(preprocessed_data, feature_type='tfidf', C=args.C, max_iter=args.max_iter,
                                     output_file=os.path.join(args.output_dir or DIRS['models'], 'logistic_regression_tfidf.pkl'))