# rule_based_classifier.py
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import nltk
from tqdm import tqdm
import os
import pickle
import argparse
from CoreModules.utils import download_nltk_resources, load_data, get_wordnet_pos, load_preprocessed_data
from CoreModules.config import DATA_PATH, DIRS

class RuleBasedClassifier:
    def __init__(self):
        # 确保下载了NLTK必要资源
        download_nltk_resources()
        # 初始化VADER情感分析器
        self.vader = SentimentIntensityAnalyzer()
        # 设置阈值
        self.pos_threshold = 0.05  # 正向情感阈值
        # 正负向词汇列表
        self.pos_words = set(['good', 'great', 'excellent', 'best', 'love', 'amazing', 'enjoy', 'perfect', 'recommend', 'favorite'])
        self.neg_words = set(['bad', 'worst', 'terrible', 'awful', 'hate', 'poor', 'waste', 'boring', 'disappointed', 'fails'])
        # 否定词列表
        self.negation_words = set(['not', 'no', 'never', "don't", "doesn't", "didn't", "cannot", "can't", "won't", "wouldn't"])
        
    def get_sentiwordnet_score(self, text):
        """使用SentiWordNet计算文本的情感得分"""
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        pos_score = 0
        neg_score = 0
        
        for word, tag in tagged:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag not in [nltk.corpus.wordnet.NOUN, nltk.corpus.wordnet.ADJ, 
                              nltk.corpus.wordnet.ADV, nltk.corpus.wordnet.VERB]:
                continue
                
            synsets = list(swn.senti_synsets(word, wn_tag))
            if not synsets:
                continue
                
            # 取第一个同义词集的情感分数
            synset = synsets[0]
            pos_score += synset.pos_score()
            neg_score += synset.neg_score()
        
        return pos_score, neg_score
    
    def predict(self, text):
        """使用规则预测文本的情感"""
        # 使用VADER获取情感分数
        vader_scores = self.vader.polarity_scores(text)
        
        # 使用SentiWordNet获取情感分数
        swn_pos_score, swn_neg_score = self.get_sentiwordnet_score(text)
        
        # 检查是否包含正向词或负向词
        tokens = word_tokenize(text.lower())
        pos_count = sum(1 for word in tokens if word in self.pos_words)
        neg_count = sum(1 for word in tokens if word in self.neg_words)
        
        # 检查否定词
        negation_count = sum(1 for word in tokens if word in self.negation_words)
        
        # 基于规则做出预测
        vader_sentiment = 1 if vader_scores['compound'] > self.pos_threshold else 0
        swn_sentiment = 1 if swn_pos_score > swn_neg_score else 0
        word_count_sentiment = 1 if pos_count > neg_count else 0
        
        # 如果否定词数量为奇数，反转基于词汇的情感
        if negation_count % 2 == 1:
            word_count_sentiment = 1 - word_count_sentiment
        
        # 加权投票(VADER权重高)
        weights = [0.6, 0.2, 0.2]
        weighted_sum = (weights[0] * vader_sentiment + 
                        weights[1] * swn_sentiment + 
                        weights[2] * word_count_sentiment)
        
        # 四舍五入到最接近的整数(0或1)
        return round(weighted_sum)
    
    def evaluate(self, test_data):
        """评估规则系统在测试集上的表现"""
        predictions = []
        print("使用规则系统进行预测...")
        
        for text in tqdm(test_data['text'], desc="规则系统预测"):
            predictions.append(self.predict(text))
            
        return np.array(predictions)

def train_rule_based_classifier(preprocessed_data=None, test_data=None, preprocessed_file=None, output_file=None):
    """训练和评估规则系统"""
    print("训练规则系统分类器...")
    
    # 如果没有提供预处理数据，尝试加载
    if preprocessed_data is None:
        if preprocessed_file:
            preprocessed_data = load_preprocessed_data(preprocessed_file)
        else:
            preprocessed_file = os.path.join(DIRS['results'], 'preprocessed_data.pkl')
            preprocessed_data = load_preprocessed_data(preprocessed_file)
            
        if preprocessed_data is None:
            print("预处理数据加载失败，将使用原始测试数据")
            if test_data is None:
                test_data = load_data(DATA_PATH['test'])
                if test_data is None:
                    print("测试数据加载失败，规则系统训练终止")
                    return None, None
        else:
            test_data = preprocessed_data['test_data']
    else:
        test_data = preprocessed_data['test_data']
    
    # 初始化规则系统
    rule_classifier = RuleBasedClassifier()
    
    # 在测试集上评估
    y_pred_rule = rule_classifier.evaluate(test_data)
    
    # 保存模型
    if output_file:
        try:
            with open(output_file, 'wb') as f:
                pickle.dump((rule_classifier, y_pred_rule), f)
            print(f"规则分类器已保存到: {output_file}")
        except Exception as e:
            print(f"保存规则分类器失败: {str(e)}")
    else:
        output_file = os.path.join(DIRS['models'], 'rule_based_classifier.pkl')
        try:
            with open(output_file, 'wb') as f:
                pickle.dump((rule_classifier, y_pred_rule), f)
            print(f"规则分类器已保存到: {output_file}")
        except Exception as e:
            print(f"保存规则分类器失败: {str(e)}")
    
    return rule_classifier, y_pred_rule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练规则系统分类器')
    parser.add_argument('--preprocessed', type=str, default=None, help='预处理数据文件路径')
    parser.add_argument('--test', type=str, default=DATA_PATH['test'], help='测试集文件路径（如果不使用预处理数据）')
    parser.add_argument('--output', type=str, default=None, help='输出模型文件路径')
    
    args = parser.parse_args()
    
    # 训练规则系统
    train_rule_based_classifier(
        preprocessed_file=args.preprocessed,
        test_data=None if args.test == DATA_PATH['test'] else load_data(args.test),
        output_file=args.output
    )