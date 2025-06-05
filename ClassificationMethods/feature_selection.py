# feature_selection.py
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import os
import argparse
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import re
from nltk.tokenize import word_tokenize
from CoreModules.utils import save_model, load_preprocessed_data
from CoreModules.config import DIRS, FEATURE_SELECTION

# 将自定义特征提取函数移到全局作用域
def extract_custom_features(texts):
    """提取自定义特征"""
    features = np.zeros((len(texts), 2))
    
    for i, text in enumerate(tqdm(texts, desc="提取自定义特征")):
        # 特征1: 评论长度
        features[i, 0] = len(text)
        
        # 特征2: 情感词比例
        pos_words = ['good', 'great', 'excellent', 'best', 'love', 'amazing', 'enjoy', 'perfect', 'recommend', 'favorite']
        neg_words = ['bad', 'worst', 'terrible', 'awful', 'hate', 'poor', 'waste', 'boring', 'disappointed', 'fails']
        
        # 将文本转为小写并标记化
        tokens = word_tokenize(text.lower())
        
        # 计算正向词和负向词的比例
        pos_count = sum(word in pos_words for word in tokens)
        neg_count = sum(word in neg_words for word in tokens)
        total_count = len(tokens)
        
        if total_count > 0:
            features[i, 1] = (pos_count - neg_count) / total_count
        
    return features

def extract_custom_features_2(texts):
    """提取更多自定义特征"""
    features = np.zeros((len(texts), 4))
    
    # 否定词列表
    negation_words = ['not', 'no', 'never', "don't", "doesn't", "didn't", "cannot", "can't", "won't", "wouldn't"]
    
    for i, text in enumerate(tqdm(texts, desc="提取自定义特征2")):
        # 特征1: 评论长度
        features[i, 0] = len(text)
        
        # 将文本转为小写并标记化
        tokens = word_tokenize(text.lower())
        
        # 特征2: 正向情感词计数
        pos_words = ['good', 'great', 'excellent', 'best', 'love', 'amazing', 'enjoy', 'perfect', 'recommend', 'favorite']
        pos_count = sum(word in pos_words for word in tokens)
        
        # 特征3: 负向情感词计数
        neg_words = ['bad', 'worst', 'terrible', 'awful', 'hate', 'poor', 'waste', 'boring', 'disappointed', 'fails']
        neg_count = sum(word in neg_words for word in tokens)
        
        # 特征4: 否定词计数
        negation_count = sum(word in negation_words for word in tokens)
        
        features[i, 1] = pos_count
        features[i, 2] = neg_count
        features[i, 3] = negation_count
        
    return features

def train_feature_selection_models(preprocessed_data=None, preprocessed_file=None, feature_sizes=None, output_dir=None):
    """使用特征选择训练模型"""
    print("开始进行特征选择实验...")
    
    # 如果没有提供预处理数据，尝试加载
    if preprocessed_data is None:
        if preprocessed_file:
            preprocessed_data = load_preprocessed_data(preprocessed_file)
        else:
            preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
            preprocessed_data = load_preprocessed_data(preprocessed_file)
            
        if preprocessed_data is None:
            print("预处理数据加载失败，特征选择实验终止")
            return None
    
    # 设置输出目录
    output_dir = output_dir or DIRS['models']
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取训练数据
    X_train_bow = preprocessed_data['X_train_bow']
    X_valid_bow = preprocessed_data['X_valid_bow']
    y_train = preprocessed_data['y_train']
    y_valid = preprocessed_data['y_valid']
    
    # 设置特征大小
    feature_sizes = feature_sizes or FEATURE_SELECTION['feature_sizes']
    
    # 特征选择 - Chi-squared
    print("\n使用卡方(Chi-squared)统计量进行特征选择")
    chi2_models = {}

    for k in feature_sizes:
        print(f"\n选择最佳的 {k} 个特征:")
        
        # 构建pipeline
        start_time = time.time()
        chi2_selector = SelectKBest(chi2, k=k)
        lr = LogisticRegression(C=1.0, max_iter=200, random_state=42, solver='lbfgs')
        
        pipeline = Pipeline([
            ('feature_selection', chi2_selector),
            ('classification', lr)
        ])
        
        # 训练模型
        pipeline.fit(X_train_bow, y_train)
        
        # 在验证集上评估
        y_valid_pred = pipeline.predict(X_valid_bow)
        accuracy = accuracy_score(y_valid, y_valid_pred)
        
        print(f"使用Chi-squared选择{k}个特征的模型训练完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"验证集上的准确率: {accuracy:.4f}")
        
        # 保存模型
        model_name = f"chi2_top{k}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        save_model(pipeline, model_path)
        chi2_models[model_name] = pipeline
    
    # 特征选择 - 互信息
    print("\n使用互信息(Mutual Information)进行特征选择")
    mi_models = {}
    
    for k in feature_sizes:
        print(f"\n选择最佳的 {k} 个特征:")
        
        # 构建pipeline
        start_time = time.time()
        mi_selector = SelectKBest(mutual_info_classif, k=k)
        lr = LogisticRegression(C=1.0, max_iter=200, random_state=42, solver='lbfgs')
        
        pipeline = Pipeline([
            ('feature_selection', mi_selector),
            ('classification', lr)
        ])
        
        # 训练模型
        pipeline.fit(X_train_bow, y_train)
        
        # 在验证集上评估
        y_valid_pred = pipeline.predict(X_valid_bow)
        accuracy = accuracy_score(y_valid, y_valid_pred)
        
        print(f"使用互信息选择{k}个特征的模型训练完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"验证集上的准确率: {accuracy:.4f}")
        
        # 保存模型
        model_name = f"mi_top{k}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        save_model(pipeline, model_path)
        mi_models[model_name] = pipeline
    
    # 定义自定义特征
    print("\n添加自定义特征...")
    
    # 获取原始文本数据
    train_texts = preprocessed_data['train_data']['text'].values
    valid_texts = preprocessed_data['valid_data']['text'].values
    
    # 提取自定义特征
    X_train_custom = extract_custom_features(train_texts)
    X_valid_custom = extract_custom_features(valid_texts)
    
    # 使用自定义特征训练模型
    print("\n使用自定义特征训练模型...")
    start_time = time.time()
    lr_custom = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    lr_custom.fit(X_train_custom, y_train)
    
    # 在验证集上评估
    y_valid_pred = lr_custom.predict(X_valid_custom)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    
    print(f"使用自定义特征的模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"验证集上的准确率: {accuracy:.4f}")
    
    # 保存模型和特征提取器
    model_path = os.path.join(output_dir, "custom_features.pkl")
    save_model((lr_custom, extract_custom_features), model_path)
    
    # 自定义特征2: 情感词计数 + 否定词检测
    print("\n添加更多自定义特征...")
    
    # 提取自定义特征2
    X_train_custom2 = extract_custom_features_2(train_texts)
    X_valid_custom2 = extract_custom_features_2(valid_texts)
    
    # 使用自定义特征2训练模型
    print("\n使用自定义特征2训练模型...")
    start_time = time.time()
    lr_custom2 = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    lr_custom2.fit(X_train_custom2, y_train)
    
    # 在验证集上评估
    y_valid_pred = lr_custom2.predict(X_valid_custom2)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    
    print(f"使用自定义特征2的模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"验证集上的准确率: {accuracy:.4f}")
    
    # 保存模型和特征提取器
    model_path = os.path.join(output_dir, "custom_features_2.pkl")
    save_model((lr_custom2, extract_custom_features_2), model_path)
    
    # 返回所有模型
    return {
        'chi2_models': chi2_models,
        'mi_models': mi_models,
        'custom_features': (lr_custom, extract_custom_features),
        'custom_features_2': (lr_custom2, extract_custom_features_2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='特征选择实验')
    parser.add_argument('--preprocessed', type=str, default=None, help='预处理数据文件路径')
    parser.add_argument('--feature-sizes', type=int, nargs='+', default=None, help='特征大小列表')
    parser.add_argument('--output-dir', type=str, default=None, help='输出模型目录')
    
    args = parser.parse_args()
    
    # 设置特征大小
    feature_sizes = args.feature_sizes or FEATURE_SELECTION['feature_sizes']
    
    # 训练特征选择模型
    train_feature_selection_models(
        preprocessed_file=args.preprocessed,
        feature_sizes=feature_sizes,
        output_dir=args.output_dir
    )