# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import argparse
from CoreModules.utils import download_nltk_resources, get_wordnet_pos, load_preprocessed_data, load_model
from CoreModules.config import DIRS
from ClassificationMethods.feature_selection import extract_custom_features, extract_custom_features_2
from ClassificationMethods.rule_based_classifier import RuleBasedClassifier

def evaluate_models(models, X_test, y_test, model_names):
    """评估模型性能
    
    参数:
    - models: 模型列表
    - X_test: 测试特征
    - y_test: 测试标签
    - model_names: 模型名称列表
    
    返回:
    - 评估结果字典
    """
    print("开始评估模型...")
    
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\n评估模型: {name}")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {prec:.4f}")
        print(f"召回率: {rec:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 存储结果
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred
        }
    
    return results

def evaluate_custom_feature_models(models, test_texts, y_test, model_names):
    """评估使用自定义特征的模型
    
    参数:
    - models: (模型, 特征提取函数)元组列表
    - test_texts: 测试文本
    - y_test: 测试标签
    - model_names: 模型名称列表
    
    返回:
    - 评估结果字典
    """
    print("\n评估自定义特征模型...")
    
    results = {}
    
    for (model, feature_extractor), name in zip(models, model_names):
        print(f"\n评估模型: {name}")
        
        # 提取特征
        X_test_custom = feature_extractor(test_texts)
        
        # 预测
        y_pred = model.predict(X_test_custom)
        
        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {prec:.4f}")
        print(f"召回率: {rec:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 存储结果
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred
        }
    
    return results

def evaluate_rule_based_model(rule_classifier, test_data, y_test):
    """评估规则系统模型
    
    参数:
    - rule_classifier: 规则系统分类器
    - test_data: 测试数据
    - y_test: 测试标签
    
    返回:
    - 评估结果字典
    """
    print("\n评估规则系统模型...")
    
    # 预测
    y_pred = rule_classifier.evaluate(test_data)
    
    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 存储结果
    results = {
        'rule_based': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred
        }
    }
    
    return results

def evaluate_llm_models(llm_results):
    """评估大语言模型结果
    
    参数:
    - llm_results: 大语言模型结果字典，格式为 {model_name: (predictions, true_labels)}
    
    返回:
    - 评估结果字典
    """
    print("\n评估大语言模型...")
    
    results = {}
    
    for name, (predictions, true_labels) in llm_results.items():
        print(f"\n评估模型: {name}")
        
        # 转换为NumPy数组
        y_pred = np.array(predictions)
        y_test = np.array(true_labels)
        
        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {prec:.4f}")
        print(f"召回率: {rec:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 存储结果
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred
        }
    
    return results

def visualize_results(all_results, output_dir=None):
    """可视化所有模型的评估结果 - 修复混淆矩阵绘制错误"""
    print("\n可视化评估结果...")
    
    # 设置输出目录
    output_dir = output_dir or DIRS['results']
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 创建DataFrame便于可视化
    df = pd.DataFrame(index=models, columns=metrics)
    
    for model in models:
        for metric in metrics:
            df.loc[model, metric] = all_results[model][metric]
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制条形图比较各个指标
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(x=df.index, y=df[metric])
        plt.title(f'{metric.capitalize()} 比较')
        plt.xlabel('模型')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # 修复混淆矩阵绘制 - 只对有匹配标签的模型绘制
    for model in models:
        try:
            y_pred = all_results[model]['y_pred']
            y_test = all_results[model]['y_test']
            
            # 确保预测和真实标签长度匹配
            if len(y_pred) != len(y_test):
                # 对于LLM模型，只使用对应的测试标签子集
                if model.startswith('llm_'):
                    print(f"注意: {model}使用了{len(y_pred)}个样本的子集进行混淆矩阵计算")
                    # 跳过混淆矩阵绘制，因为样本数量不匹配
                    continue
                else:
                    print(f"警告: {model}的预测和真实标签长度不匹配")
                    continue
            
            plt.figure(figsize=(8, 6))
            
            # 创建混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'{model} 混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model}.png'))
            plt.close()
        except Exception as e:
            print(f"无法为{model}绘制混淆矩阵: {e}")
    
    # 将结果保存为CSV
    df.to_csv(os.path.join(output_dir, 'model_metrics.csv'))
    
    print("可视化完成。结果已保存到结果目录。")

def safe_correlation(x, y):
    """安全的相关系数计算，处理异常值"""
    try:
        # 检查是否有足够的变化
        if np.var(x) == 0 or np.var(y) == 0:
            return np.nan  # 如果方差为0，返回NaN
        
        # 检查是否有有效值
        if len(x) == 0 or len(y) == 0:
            return np.nan
            
        correlation = np.corrcoef(x, y)[0, 1]
        
        # 检查结果是否有效
        if np.isnan(correlation) or np.isinf(correlation):
            return np.nan
            
        return correlation
    except:
        return np.nan

def analyze_with_sentiwordnet(test_data, all_results, output_dir=None):
    """使用SentiWordNet分析模型输出结果 - 修复异常值问题"""
    print("\n使用SentiWordNet分析模型输出结果...")
    
    # 设置输出目录
    output_dir = output_dir or DIRS['results']
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保下载了NLTK必要资源
    download_nltk_resources()
    
    # 初始化SentiWordNet分析器
    sia = SentimentIntensityAnalyzer()
    
    # 存储分析结果
    swn_analysis = {}
    
    # 获取测试文本
    test_texts = test_data['text'].values
    
    # 对每个模型进行分析
    for model_name, results in all_results.items():
        if model_name == 'rule_based':
            continue  # 跳过规则系统
            
        print(f"\n分析{model_name}的输出结果...")
        
        y_pred = results['y_pred']
        
        # 分析结果
        pos_words_count = []
        neg_words_count = []
        compound_scores = []
        
        for i, text in enumerate(tqdm(test_texts, desc=f"分析{model_name}")):
            # 使用VADER计算情感得分
            scores = sia.polarity_scores(text)
            compound_scores.append(scores['compound'])
            
            # 标记化并标注词性
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # 计算正向和负向词数量
            pos_count = 0
            neg_count = 0
            
            for word, tag in tagged:
                wn_tag = get_wordnet_pos(tag)
                
                synsets = list(swn.senti_synsets(word, wn_tag))
                if not synsets:
                    continue
                    
                # 取第一个同义词集的情感分数
                synset = synsets[0]
                
                if synset.pos_score() > synset.neg_score():
                    pos_count += 1
                elif synset.neg_score() > synset.pos_score():
                    neg_count += 1
            
            pos_words_count.append(pos_count)
            neg_words_count.append(neg_count)
        
        # 使用安全的相关系数计算
        if len(y_pred) < len(pos_words_count):
            # 只使用与预测长度匹配的词语子集
            pos_words_subset = pos_words_count[:len(y_pred)]
            neg_words_subset = neg_words_count[:len(y_pred)]
            compound_subset = compound_scores[:len(y_pred)]
            
            pos_corr = safe_correlation(np.array(pos_words_subset), y_pred)
            neg_corr = safe_correlation(np.array(neg_words_subset), y_pred)
            compound_corr = safe_correlation(np.array(compound_subset), y_pred)
            
            print(f"注意: 仅使用了测试集的子集 ({len(y_pred)}/{len(pos_words_count)}个样本) 计算相关系数")
        else:
            pos_corr = safe_correlation(np.array(pos_words_count), y_pred)
            neg_corr = safe_correlation(np.array(neg_words_count), y_pred)
            compound_corr = safe_correlation(np.array(compound_scores), y_pred)
        
        # 处理特殊情况的输出
        if np.isnan(pos_corr):
            print(f"正向词数量与预测的相关系数: 无法计算（预测值无变化或数据异常）")
        else:
            print(f"正向词数量与预测的相关系数: {pos_corr:.4f}")
            
        if np.isnan(neg_corr):
            print(f"负向词数量与预测的相关系数: 无法计算（预测值无变化或数据异常）")
        else:
            print(f"负向词数量与预测的相关系数: {neg_corr:.4f}")
            
        if np.isnan(compound_corr):
            print(f"VADER复合得分与预测的相关系数: 无法计算（预测值无变化或数据异常）")
        else:
            print(f"VADER复合得分与预测的相关系数: {compound_corr:.4f}")
        
        # 存储结果
        swn_analysis[model_name] = {
            'pos_correlation': pos_corr if not np.isnan(pos_corr) else 0,
            'neg_correlation': neg_corr if not np.isnan(neg_corr) else 0,
            'compound_correlation': compound_corr if not np.isnan(compound_corr) else 0,
            'pos_words_count': pos_words_count,
            'neg_words_count': neg_words_count,
            'compound_scores': compound_scores
        }
    
    # 可视化SentiWordNet分析结果
    if swn_analysis:  # 确保有数据可视化
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        models = list(swn_analysis.keys())
        pos_corrs = [swn_analysis[model]['pos_correlation'] for model in models]
        neg_corrs = [swn_analysis[model]['neg_correlation'] for model in models]
        compound_corrs = [swn_analysis[model]['compound_correlation'] for model in models]
        
        # 绘制条形图
        x = np.arange(len(models))
        width = 0.25
        
        plt.bar(x - width, pos_corrs, width, label='正向词相关性')
        plt.bar(x, neg_corrs, width, label='负向词相关性')
        plt.bar(x + width, compound_corrs, width, label='复合得分相关性')
        
        plt.xlabel('模型')
        plt.ylabel('相关系数')
        plt.title('各模型输出与SentiWordNet分析的相关性')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiwordnet_correlation.png'))
        plt.close()
        
        # 将结果保存为CSV
        swn_df = pd.DataFrame({
            'model': models,
            'pos_correlation': pos_corrs,
            'neg_correlation': neg_corrs,
            'compound_correlation': compound_corrs
        })
        swn_df.to_csv(os.path.join(output_dir, 'sentiwordnet_analysis.csv'), index=False)
    
    return swn_analysis

def analyze_llm_structured_output(structured_df, output_dir=None):
    """分析大语言模型的结构化输出 - 改进分析质量"""
    print("\n分析大语言模型的结构化输出...")
    
    # 设置输出目录
    output_dir = output_dir or DIRS['results']
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算情感分类准确率
    accuracy = accuracy_score(structured_df['true_label'], structured_df['sentiment'])
    print(f"结构化输出的情感分类准确率: {accuracy:.4f}")
    
    # 分析主题分布
    topic_counts = structured_df['topic'].value_counts()
    print("\n主题分布:")
    for topic, count in topic_counts.items():
        print(f"{topic}: {count}")
    
    # 分析情感分布
    emotion_counts = structured_df['emotion'].value_counts()
    print("\n情感分布:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")
    
    # 分析情感强度分布
    intensity_counts = structured_df['intensity'].value_counts().sort_index()
    print("\n情感强度分布:")
    for intensity, count in intensity_counts.items():
        print(f"{intensity}: {count}")
    
    # 分析情感强度与情感分类的关系
    intensity_by_sentiment = structured_df.groupby('sentiment')['intensity'].mean()
    print("\n不同情感分类的平均情感强度:")
    for sentiment, avg_intensity in intensity_by_sentiment.items():
        print(f"情感 {sentiment}: {avg_intensity:.2f}")
    
    # 分析情感与主题的关系
    topic_by_sentiment = pd.crosstab(structured_df['sentiment'], structured_df['topic'])
    print("\n主题与情感的关系:")
    print(topic_by_sentiment)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    topic_counts.plot(kind='bar')
    plt.title('主题分布')
    plt.xlabel('主题')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_topic_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar')
    plt.title('情感分布')
    plt.xlabel('情感')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_emotion_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    intensity_counts.plot(kind='bar')
    plt.title('情感强度分布')
    plt.xlabel('强度')
    plt.ylabel('数量')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_intensity_distribution.png'))
    plt.close()
    
    # 改进的总结发现
    pos_topics = topic_by_sentiment.iloc[1] if len(topic_by_sentiment) > 1 and not topic_by_sentiment.iloc[1].empty else pd.Series()
    neg_topics = topic_by_sentiment.iloc[0] if not topic_by_sentiment.iloc[0].empty else pd.Series()
    
    summary = f"""
    大语言模型结构化输出分析总结:

    1. 情感分类准确率: {accuracy * 100:.2f}%

    2. 样本统计:
    - 总样本数: {len(structured_df)}
    - 正面样本数: {sum(structured_df['true_label'] == 1)}
    - 负面样本数: {sum(structured_df['true_label'] == 0)}

    3. 主要讨论的电影主题:
    - 最常见主题: {topic_counts.index[0]}
    - 主题多样性: 共识别出{len(topic_counts)}种不同主题

    4. 主要表达的情感:
    - 最常见情感: {emotion_counts.index[0]}
    - 情感多样性: 共识别出{len(emotion_counts)}种不同情感

    5. 情感强度:
    - 平均强度: {structured_df['intensity'].mean():.2f}/5
    - 正面评论平均强度: {structured_df[structured_df['sentiment'] == 1]['intensity'].mean():.2f}/5
    - 负面评论平均强度: {structured_df[structured_df['sentiment'] == 0]['intensity'].mean():.2f}/5

    6. 主题与情感关系:
    - 最容易获得正面评价的主题: {pos_topics.idxmax() if not pos_topics.empty else 'N/A'}
    - 最容易获得负面评价的主题: {neg_topics.idxmax() if not neg_topics.empty else 'N/A'}
    
    7. 模型表现评估:
    - 分类准确率达到{accuracy * 100:.1f}%，表现{'良好' if accuracy > 0.8 else '一般' if accuracy > 0.6 else '较差'}
    - 主题识别多样性{'较好' if len(topic_counts) > 5 else '一般' if len(topic_counts) > 3 else '较低'}
    - 情感识别多样性{'较好' if len(emotion_counts) > 5 else '一般' if len(emotion_counts) > 3 else '较低'}
        """
    
    print(summary)
    
    # 保存总结
    with open(os.path.join(output_dir, 'llm_structured_analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return {
        'accuracy': accuracy,
        'topic_counts': topic_counts,
        'emotion_counts': emotion_counts,
        'intensity_counts': intensity_counts,
        'topic_by_sentiment': topic_by_sentiment,
        'summary': summary
    }

def run_evaluation(preprocessed_data=None, preprocessed_file=None, 
                  models_dir=None, llm_results_dir=None, 
                  structured_output_file=None, output_dir=None):
    """运行完整的评估流程"""
    print("开始模型评估...")
    
    # 设置目录
    models_dir = models_dir or DIRS['models']
    output_dir = output_dir or DIRS['results']
    llm_results_dir = llm_results_dir or DIRS['results']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预处理数据
    if preprocessed_data is None:
        preprocessed_file = preprocessed_file or os.path.join(DIRS['results'], 'preprocessed_data.pkl')
        preprocessed_data = load_preprocessed_data(preprocessed_file)
        
        if preprocessed_data is None:
            print("预处理数据加载失败，评估终止")
            return None
    
    # 获取测试数据
    test_data = preprocessed_data['test_data']
    X_test_bow = preprocessed_data['X_test_bow']
    X_test_tfidf = preprocessed_data['X_test_tfidf']
    y_test = preprocessed_data['y_test']
    
    # 加载模型
    ml_models = {}
    ml_model_names = ['naive_bayes', 'logistic_regression_bow', 'logistic_regression_tfidf']
    
    for model_name in ml_model_names:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        model = load_model(model_path)
        if model is not None:
            ml_models[model_name] = model
    
    # 加载特征选择模型
    chi2_models = {}
    mi_models = {}
    custom_feature_models = []
    custom_feature_names = []
    
    # Chi2模型
    for k in [200, 2000]:
        model_name = f"chi2_top{k}"
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        model = load_model(model_path)
        if model is not None:
            chi2_models[model_name] = model
    
    # MI模型
    for k in [200, 2000]:
        model_name = f"mi_top{k}"
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        model = load_model(model_path)
        if model is not None:
            mi_models[model_name] = model
    
    # 自定义特征模型
    for name in ['custom_features', 'custom_features_2']:
        model_path = os.path.join(models_dir, f"{name}.pkl")
        model = load_model(model_path)
        if model is not None:
            custom_feature_models.append(model)
            custom_feature_names.append(name)
    
    # 加载规则分类器
    rule_classifier_path = os.path.join(models_dir, 'rule_based_classifier.pkl')
    rule_result = load_model(rule_classifier_path)
    if rule_result is not None:
        rule_classifier, _ = rule_result
    else:
        rule_classifier = None
    
    # 加载LLM结果
    llm_results = {}
    for strategy in ["simple", "few_shot", "cot"]:
        result_path = os.path.join(llm_results_dir, f"llm_{strategy}_results.pkl")
        try:
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            llm_results[f"llm_{strategy}"] = result
        except:
            print(f"无法加载LLM结果: {result_path}")
    
    # 评估模型
    all_results = {}
    
    # 评估机器学习模型
    if ml_models:
        ml_results = evaluate_models(list(ml_models.values()), X_test_bow, y_test, list(ml_models.keys()))
        all_results.update(ml_results)
    
    # 评估Chi2特征选择模型
    if chi2_models:
        chi2_results = evaluate_models(list(chi2_models.values()), X_test_bow, y_test, list(chi2_models.keys()))
        all_results.update(chi2_results)
    
    # 评估MI特征选择模型
    if mi_models:
        mi_results = evaluate_models(list(mi_models.values()), X_test_bow, y_test, list(mi_models.keys()))
        all_results.update(mi_results)
    
    # 评估自定义特征模型
    if custom_feature_models:
        test_texts = test_data['text'].values
        custom_results = evaluate_custom_feature_models(custom_feature_models, test_texts, y_test, custom_feature_names)
        all_results.update(custom_results)
    
    # 评估规则分类器
    if rule_classifier is not None:
        rule_results = evaluate_rule_based_model(rule_classifier, test_data, y_test)
        all_results.update(rule_results)
    
    # 评估LLM模型
    if llm_results:
        llm_model_results = evaluate_llm_models(llm_results)
        all_results.update(llm_model_results)
    
    # 为所有结果添加测试集标签
    for model_name in all_results:
        all_results[model_name]['y_test'] = y_test
    
    # 可视化结果
    visualize_results(all_results, output_dir)
    
    # 使用SentiWordNet分析模型输出
    swn_analysis = analyze_with_sentiwordnet(test_data, all_results, output_dir)
    
    # 分析LLM结构化输出
    if structured_output_file:
        try:
            structured_df = pd.read_csv(structured_output_file)
            llm_structured_analysis = analyze_llm_structured_output(structured_df, output_dir)
        except:
            print(f"无法加载结构化输出文件: {structured_output_file}")
    
    print("\n模型评估完成。所有结果已保存到结果目录。")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估模型性能')
    parser.add_argument('--preprocessed', type=str, default=None, help='预处理数据文件路径')
    parser.add_argument('--models-dir', type=str, default=None, help='模型目录')
    parser.add_argument('--llm-results-dir', type=str, default=None, help='LLM结果目录')
    parser.add_argument('--structured-output', type=str, default=None, help='结构化输出文件路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 运行评估
    run_evaluation(
        preprocessed_file=args.preprocessed,
        models_dir=args.models_dir,
        llm_results_dir=args.llm_results_dir,
        structured_output_file=args.structured_output,
        output_dir=args.output_dir
    )