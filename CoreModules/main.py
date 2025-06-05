# main.py
import argparse
import os
from config import DATA_PATH, DIRS, PREPROCESSING, FEATURE_SELECTION, LLM
from DataProcessing_And_Analysis.data_analysis import analyze_data
from DataProcessing_And_Analysis.data_preprocessing import prepare_datasets
from ClassificationMethods.rule_based_classifier import train_rule_based_classifier
from ClassificationMethods.ml_models import train_ml_models
from ClassificationMethods.feature_selection import train_feature_selection_models
from ClassificationMethods.llm_classifier import run_llm_classification
from Evaluation.evaluation import run_evaluation
from utils import download_nltk_resources

def run_full_pipeline(args):
    """运行完整的IMDB情感分类实验流程"""
    print("IMDB情感分类实验开始...")
    
    # 创建必要的目录
    for dir_name, dir_path in DIRS.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"目录已创建/确认: {dir_path}")
    
    # 确保下载了NLTK必要资源
    download_nltk_resources()
    
    # 1. 数据分析
    if args.run_analysis:
        print("\n========== 第一部分：数据分析 ==========")
        analysis_results = analyze_data(
            train_file=args.train,
            valid_file=args.valid,
            test_file=args.test
        )
    
    # 2. 数据预处理
    if args.run_preprocessing:
        print("\n========== 第二部分：数据预处理 ==========")
        preprocessed_data = prepare_datasets(
            train_file=args.train,
            valid_file=args.valid,
            test_file=args.test,
            remove_stopwords=PREPROCESSING['remove_stopwords'],
            lemmatize=PREPROCESSING['lemmatize'],
            max_features=PREPROCESSING['max_features']
        )
    else:
        # 如果跳过预处理，尝试加载已有的预处理数据
        from utils import load_preprocessed_data
        preprocessed_data = load_preprocessed_data(os.path.join(DIRS['results'], 'preprocessed_data.pkl'))
        if preprocessed_data is None and (args.run_rules or args.run_ml or args.run_features or args.run_evaluation):
            print("未找到预处理数据，请先运行预处理步骤")
            return
    
    # 3. 训练规则系统
    if args.run_rules:
        print("\n========== 第三部分：训练产生式系统 ==========")
        rule_classifier, _ = train_rule_based_classifier(preprocessed_data)
    
    # 4. 训练朴素贝叶斯和逻辑回归模型
    if args.run_ml:
        print("\n========== 第四部分：训练朴素贝叶斯和逻辑回归模型 ==========")
        ml_models = train_ml_models(preprocessed_data)
    
    # 5. 特征选择实验
    if args.run_features:
        print("\n========== 第五部分：特征选择实验 ==========")
        feature_selection_models = train_feature_selection_models(
            preprocessed_data,
            feature_sizes=FEATURE_SELECTION['feature_sizes']
        )
    
    # 6. 大语言模型实验
    if args.run_llm:
        print("\n========== 第六部分：大语言模型实验 ==========")
        llm_results, structured_df = run_llm_classification(
            preprocessed_data['test_data'] if preprocessed_data else None,
            test_file=args.test,
            sample_size=LLM['sample_size'],
            structured_sample_size=LLM['structured_sample_size']
        )
    
    # 7. 模型评估
    if args.run_evaluation:
        print("\n========== 第七部分：模型评估 ==========")
        evaluation_results = run_evaluation(
            preprocessed_data=preprocessed_data,
            structured_output_file=os.path.join(DIRS['results'], 'llm_structured_outputs.csv')
        )
    
    print("\n========== IMDB情感分类实验完成 ==========")
    print("所有结果和图表已保存到results和figures目录")

def main():
    parser = argparse.ArgumentParser(description='IMDB情感分类实验')
    
    # 数据文件参数
    parser.add_argument('--train', type=str, default=DATA_PATH['train'], help='训练集文件路径')
    parser.add_argument('--valid', type=str, default=DATA_PATH['valid'], help='验证集文件路径')
    parser.add_argument('--test', type=str, default=DATA_PATH['test'], help='测试集文件路径')
    
    # 运行特定步骤的参数
    parser.add_argument('--run-analysis', action='store_true', help='运行数据分析')
    parser.add_argument('--run-preprocessing', action='store_true', help='运行数据预处理')
    parser.add_argument('--run-rules', action='store_true', help='训练规则系统分类器')
    parser.add_argument('--run-ml', action='store_true', help='训练机器学习模型')
    parser.add_argument('--run-features', action='store_true', help='运行特征选择实验')
    parser.add_argument('--run-llm', action='store_true', help='运行大语言模型实验')
    parser.add_argument('--run-evaluation', action='store_true', help='评估模型性能')
    parser.add_argument('--run-all', action='store_true', help='运行所有步骤')
    
    args = parser.parse_args()
    
    # 如果没有指定任何步骤，或者使用了--run-all，则运行所有步骤
    if args.run_all or not any([args.run_analysis, args.run_preprocessing, args.run_rules, 
                              args.run_ml, args.run_features, args.run_llm, args.run_evaluation]):
        args.run_analysis = True
        args.run_preprocessing = True
        args.run_rules = True
        args.run_ml = True
        args.run_features = True
        args.run_llm = True
        args.run_evaluation = True
    
    run_full_pipeline(args)

if __name__ == "__main__":
    main()