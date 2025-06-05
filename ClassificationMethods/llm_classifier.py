# llm_classifier.py
import pickle
import numpy as np
import pandas as pd
import json
import os
import time
import re
import argparse
from tqdm import tqdm
from openai import OpenAI
from CoreModules.utils import load_data
from CoreModules.config import DATA_PATH, DIRS, LLM

def setup_deepseek_client():
    """设置DeepSeek客户端"""
    # 从系统环境变量读取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量，请先设置该变量")
    
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client

def classify_with_llm(client, texts, model="deepseek-chat", prompt_strategy="simple"):
    """使用大语言模型进行情感分类
    
    参数:
    - client: DeepSeek客户端
    - texts: 要分类的文本列表
    - model: 要使用的模型名称
    - prompt_strategy: 提示策略 ("simple"、"few_shot"或"cot")
    
    返回:
    - 预测结果列表
    """
    predictions = []
    
    # 定义不同的提示策略 - 修复CoT策略问题
    prompts = {
        "simple": "你是一个情感分析专家，你需要判断以下电影评论的情感是正面的还是负面的。\n\n评论: {text}\n\n请只回答0（负面情感）或1（正面情感）。",
        
        "few_shot": """你是一个情感分析专家，你需要判断以下电影评论的情感是正面的还是负面的。

以下是一些例子：

评论: "这部电影太棒了，我非常喜欢，强烈推荐！"
情感: 1（正面情感）

评论: "浪费时间和金钱，剧情无聊，演技差劲。"
情感: 0（负面情感）

评论: "演员的表演令人惊叹，但剧情有些拖沓。"
情感: 1（正面情感）

评论: "特效还行，但故事情节太烂了，完全不值得看。"
情感: 0（负面情感）

现在，请分析以下评论：
评论: {text}

请只回答0（负面情感）或1（正面情感）。""",
        
        "cot": """你是一个情感分析专家，请使用逐步分析的方法判断以下电影评论的情感倾向。

评论: {text}

请按以下步骤分析：
1. 识别关键情感词汇：找出评论中表达情感的关键词
2. 判断情感倾向：分析这些词汇是偏向正面还是负面
3. 考虑否定词：检查是否有否定词改变了情感倾向
4. 综合判断：基于以上分析得出最终结论

重要提示：请保持客观和平衡的判断，不要偏向任何一方。

最终答案（必须是0或1）："""
    }
    
    for text in tqdm(texts, desc="LLM情感分类"):
        # 选择提示策略并填充文本
        prompt = prompts[prompt_strategy].format(text=text)
        
        try:
            # 调用DeepSeek API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的情感分析助手。请客观分析，避免偏向性判断。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # 适当提高温度增加随机性，避免系统性偏向
            )
            
            # 提取回答
            answer = response.choices[0].message.content.strip()
            
            # 更严格的解析逻辑
            # 先查找明确的数字答案
            if re.search(r'\b1\b', answer) and not re.search(r'\b0\b', answer):
                predictions.append(1)
            elif re.search(r'\b0\b', answer) and not re.search(r'\b1\b', answer):
                predictions.append(0)
            elif "正面" in answer or "积极" in answer:
                predictions.append(1)
            elif "负面" in answer or "消极" in answer:
                predictions.append(0)
            else:
                # 如果无法明确解析，随机分配（避免系统性偏向）
                predictions.append(np.random.choice([0, 1]))
                
            # 添加延迟以避免API速率限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"API调用错误: {e}")
            # 出错时随机分配，避免偏向
            predictions.append(np.random.choice([0, 1]))
    
    return predictions

def parse_json_from_text(text):
    """从文本中提取并解析JSON"""
    # 方法1：寻找完整的JSON块
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # 方法2：如果找不到完整JSON，尝试手动提取字段
    try:
        # 使用正则表达式提取各个字段
        sentiment_match = re.search(r'"sentiment"\s*:\s*(\d+)', text)
        topic_match = re.search(r'"topic"\s*:\s*"([^"]*)"', text)
        emotion_match = re.search(r'"emotion"\s*:\s*"([^"]*)"', text)
        intensity_match = re.search(r'"intensity"\s*:\s*(\d+)', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        
        result = {
            "sentiment": int(sentiment_match.group(1)) if sentiment_match else 0,
            "topic": topic_match.group(1) if topic_match else "unknown",
            "emotion": emotion_match.group(1) if emotion_match else "unknown",
            "intensity": int(intensity_match.group(1)) if intensity_match else 1,
            "reasoning": reasoning_match.group(1) if reasoning_match else "Failed to parse"
        }
        return result
    except:
        pass
    
    # 方法3：如果都失败，返回默认值
    return {
        "sentiment": 0,
        "topic": "unknown",
        "emotion": "unknown",
        "intensity": 1,
        "reasoning": "Failed to parse"
    }

def classify_with_llm_structured(client, texts, model="deepseek-chat"):
    """使用大语言模型生成结构化输出
    
    参数:
    - client: DeepSeek客户端
    - texts: 要分类的文本列表
    - model: 要使用的模型名称
    
    返回:
    - 结构化输出列表
    """
    structured_outputs = []
    
    # 改进的提示，增加主题和情感的多样性
    prompt = """你是一个专业的电影评论分析师。请分析以下电影评论，并严格按照JSON格式提供输出。

评论: {text}

请提供以下JSON格式的分析结果（请确保输出只包含JSON，不要有其他文字）：
{{"sentiment": 0, "topic": "剧情", "emotion": "失望", "intensity": 3, "reasoning": "简短解释"}}

字段说明：
- sentiment: 0表示负面，1表示正面
- topic: 评论主要讨论的电影方面（如：剧情、演技、特效、音乐、导演、摄影、剪辑等）
- emotion: 评论者的主要情感（如：喜悦、失望、愤怒、惊喜、感动、无聊、兴奋等）
- intensity: 情感强度1-5（1=很轻微，5=非常强烈）
- reasoning: 30字以内的判断解释

请尽量识别多样化的主题和情感，不要总是选择相同的类别。"""
    
    for i, text in enumerate(tqdm(texts, desc="LLM结构化分析")):
        try:
            # 调用DeepSeek API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的电影评论分析助手。请严格按照要求的JSON格式回复，识别多样化的主题和情感，不要总是选择相同的类别。"},
                    {"role": "user", "content": prompt.format(text=text)}
                ],
                temperature=0.3  # 适当的温度保持一定随机性
            )
            
            # 提取回答
            answer = response.choices[0].message.content.strip()
            print(f"第{i+1}条回复原文: {answer}")  # 调试信息
            
            # 使用改进的JSON解析方法
            structured_output = parse_json_from_text(answer)
            structured_outputs.append(structured_output)
            
            # 添加延迟以避免API速率限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"API调用错误 (第{i+1}条): {e}")
            structured_outputs.append({
                "sentiment": 0,
                "topic": "unknown",
                "emotion": "unknown", 
                "intensity": 1,
                "reasoning": "API error"
            })
    
    return structured_outputs

def run_llm_classification(test_data=None, test_file=None, sample_size=None, structured_sample_size=None, output_dir=None, model="deepseek-chat"):
    """运行所有大语言模型实验"""
    print("开始大语言模型实验...")
    
    # 确保输出目录存在
    output_dir = output_dir or DIRS['results']
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    if test_data is None:
        if test_file:
            test_data = load_data(test_file)
        else:
            test_file = DATA_PATH['test']
            test_data = load_data(test_file)
            
        if test_data is None:
            print("测试数据加载失败，大语言模型实验终止")
            return None, None
    
    # 设置样本大小 - 增加结构化输出的样本数量
    sample_size = sample_size or LLM['sample_size']
    structured_sample_size = structured_sample_size or min(50, len(test_data))  # 增加到50个样本
    
    # 设置DeepSeek客户端
    try:
        client = setup_deepseek_client()
    except ValueError as e:
        print(f"设置DeepSeek客户端失败: {str(e)}")
        print("请确保已设置DEEPSEEK_API_KEY环境变量，您可以跳过这个步骤或稍后再试")
        return None, None
    
    # 从测试集随机抽取样本
    sample_size = min(sample_size, len(test_data))
    sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
    sample_texts = test_data.iloc[sample_indices]['text'].values
    sample_labels = test_data.iloc[sample_indices]['label'].values
    
    # 使用不同的提示策略进行分类
    results = {}
    
    for strategy in ["simple", "few_shot", "cot"]:
        print(f"\n使用{strategy}提示策略进行分类...")
        predictions = classify_with_llm(client, sample_texts, model=model, prompt_strategy=strategy)
        results[f"llm_{strategy}"] = (predictions, sample_labels)
        
        # 保存结果
        result_file = os.path.join(output_dir, f"llm_{strategy}_results.pkl")
        try:
            with open(result_file, 'wb') as f:
                pickle.dump((predictions, sample_labels), f)
            print(f"结果已保存到: {result_file}")
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
    
    # 结构化输出实验 - 确保正负样本平衡
    print("\n进行结构化输出实验...")
    
    # 分别从正负样本中抽取，确保平衡
    positive_samples = test_data[test_data['label'] == 1]
    negative_samples = test_data[test_data['label'] == 0]
    
    structured_size = min(structured_sample_size, len(test_data))
    pos_size = structured_size // 2
    neg_size = structured_size - pos_size
    
    pos_indices = np.random.choice(len(positive_samples), min(pos_size, len(positive_samples)), replace=False)
    neg_indices = np.random.choice(len(negative_samples), min(neg_size, len(negative_samples)), replace=False)
    
    structured_texts = np.concatenate([
        positive_samples.iloc[pos_indices]['text'].values,
        negative_samples.iloc[neg_indices]['text'].values
    ])
    structured_labels = np.concatenate([
        positive_samples.iloc[pos_indices]['label'].values,
        negative_samples.iloc[neg_indices]['label'].values
    ])
    
    # 打乱顺序
    shuffle_indices = np.random.permutation(len(structured_texts))
    structured_texts = structured_texts[shuffle_indices]
    structured_labels = structured_labels[shuffle_indices]
    
    structured_outputs = classify_with_llm_structured(client, structured_texts, model=model)
    
    # 将结构化输出和真实标签保存为DataFrame
    structured_df = pd.DataFrame(structured_outputs)
    structured_df['true_label'] = structured_labels
    structured_df['text'] = structured_texts
    
    # 保存结果
    structured_file = os.path.join(output_dir, 'llm_structured_outputs.csv')
    try:
        structured_df.to_csv(structured_file, index=False, encoding='utf-8')
        print(f"结构化输出已保存到: {structured_file}")
    except Exception as e:
        print(f"保存结构化输出失败: {str(e)}")
    
    return results, structured_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用大语言模型进行情感分类')
    parser.add_argument('--test', type=str, default=None, help='测试集文件路径')
    parser.add_argument('--sample-size', type=int, default=None, help='样本大小')
    parser.add_argument('--structured-sample-size', type=int, default=None, help='结构化输出样本大小')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--model', type=str, default="deepseek-chat", help='要使用的模型名称')
    
    args = parser.parse_args()
    
    # 运行大语言模型实验
    run_llm_classification(
        test_file=args.test,
        sample_size=args.sample_size,
        structured_sample_size=args.structured_sample_size,
        output_dir=args.output_dir,
        model=args.model
    )