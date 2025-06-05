# config.py
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径 - 指向Data_And_Results目录
DATA_PATH = {
    'train': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'Train.csv'),
    'valid': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'Valid.csv'),
    'test': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'Test.csv')
}

# 目录配置 - 指向Data_And_Results子目录
DIRS = {
    'models': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'models'),
    'results': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'results'),
    'figures': os.path.join(PROJECT_ROOT, 'Data_And_Results', 'figures')
}

# 创建必要的目录
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# 预处理配置
PREPROCESSING = {
    'remove_stopwords': True,
    'lemmatize': True,
    'max_features': 10000
}

# 特征选择配置
FEATURE_SELECTION = {
    'feature_sizes': [200, 2000]
}

# 大语言模型配置 - 增加结构化输出样本数量
LLM = {
    'sample_size': 200,
    'structured_sample_size': 20
}