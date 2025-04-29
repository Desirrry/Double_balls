import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import beta



def load_data(filename='Double_color_balls\data.json'):
    """读取data数据 并进行清理
    使之变成['4', '18', '19', '24', '27', '30', '16']等
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [n['开奖号码'] for n in data]


# 1. 基础统计分析
def basic_analysis(df):
    # 红球频率统计
    red_balls = df[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values.flatten()
    red_freq = pd.Series(red_balls).value_counts().sort_index()
    
    # 蓝球频率统计
    blue_freq = df['B'].value_counts().sort_index()
    
    return red_freq, blue_freq

# 2. 贝叶斯概率计算
class BayesianLotteryPredictor:
    def __init__(self, df):
        self.df = df
        self.red_prior = None
        self.blue_prior = None
        self.red_likelihood = defaultdict(dict)
        self.blue_likelihood = defaultdict(dict)
        
    def calculate_priors(self):
        # 计算先验概率（历史频率）
        red_balls = self.df[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values.flatten()
        self.red_prior = pd.Series(red_balls).value_counts(normalize=True)
        
        self.blue_prior = self.df['B'].value_counts(normalize=True)
    
    """window可调整参数"""
    def calculate_likelihoods(self, window=10):
        # 计算似然（近期趋势）
        recent = self.df.head(window)
        
        # 红球近期频率
        recent_red = recent[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values.flatten()
        recent_red_freq = pd.Series(recent_red).value_counts(normalize=True)
        
        # 蓝球近期频率
        recent_blue_freq = recent['B'].value_counts(normalize=True)
        
        # 计算调整因子
        for num in self.red_prior.index:
            if num in recent_red_freq:
                self.red_likelihood[num]['factor'] = recent_red_freq[num] / self.red_prior[num]
            else:
                self.red_likelihood[num]['factor'] = 0.5  # 未出现则降低权重
        
        for num in self.blue_prior.index:
            if num in recent_blue_freq:
                self.blue_likelihood[num]['factor'] = recent_blue_freq[num] / self.blue_prior[num]
            else:
                self.blue_likelihood[num]['factor'] = 0.5
    
    def predict(self):
        self.calculate_priors()
        self.calculate_likelihoods()
        
        # 计算后验概率
        red_posterior = self.red_prior.copy()
        for num in red_posterior.index:
            red_posterior[num] *= self.red_likelihood[num]['factor']
        
        blue_posterior = self.blue_prior.copy()
        for num in blue_posterior.index:
            blue_posterior[num] *= self.blue_likelihood[num]['factor']
        
        # 归一化
        red_posterior /= red_posterior.sum()
        blue_posterior /= blue_posterior.sum()
        
        return red_posterior.sort_values(ascending=False), blue_posterior.sort_values(ascending=False)

# 3. 生成推荐号码
def generate_recommendation(red_pred, blue_pred, n=6):
    # 选择概率最高的6个红球和1个蓝球
    recommended_red = red_pred.index[:n].tolist()
    recommended_blue = blue_pred.index[0]
    
    return recommended_red, recommended_blue

"""baseline参数可调整敏感度"""
# 4. 冷热号分析（基于贝叶斯）
def analyze_cold_hot(red_pred, blue_pred, baseline=0.5):
    print("\n冷热号分析:")
    # 红球冷号（后验概率 < 先验概率）
    red_cold = red_pred[red_pred < baseline*red_pred.mean()].index.tolist()
    print(f"红球冷号: {red_cold}")
    
    # 红球热号（后验概率 > 先验概率）
    red_hot = red_pred[red_pred > (2-baseline)*red_pred.mean()].index.tolist()
    print(f"红球热号: {red_hot}")
    
    # 蓝球冷热分析
    blue_cold = blue_pred[blue_pred < baseline*blue_pred.mean()].index.tolist()
    print(f"蓝球冷号: {blue_cold}")
    
    blue_hot = blue_pred[blue_pred > (2-baseline)*blue_pred.mean()].index.tolist()
    print(f"蓝球热号: {blue_hot}")


def main():
    # 数据读取
    data = load_data()
    # 转换为DataFrame
    df = pd.DataFrame(data, columns=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'B'])

    # 频率分析输出
    red_freq, blue_freq = basic_analysis(df)
    print("红球频率统计:\n", red_freq)
    print("\n蓝球频率统计:\n", blue_freq)


    # 使用贝叶斯预测
    predictor = BayesianLotteryPredictor(df)
    red_pred, blue_pred = predictor.predict()

    print("\n红球预测概率（从高到低）:")
    print(red_pred.head(10))
    print("\n蓝球预测概率（从高到低）:")
    print(blue_pred.head(5))

    # 生成推荐号码
    rec_red, rec_blue = generate_recommendation(red_pred, blue_pred)
    print("\n推荐号码:")
    print(f"红球: {', '.join(rec_red)}")
    print(f"蓝球: {rec_blue}")


    # 冷热号分析（基于贝叶斯）
    analyze_cold_hot(red_pred, blue_pred)

if __name__ == '__main__':
    main()