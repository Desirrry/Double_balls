import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from typing import List, Tuple

def load_data(filename='Double_color_balls/data.json') -> List[List[str]]:
    """读取并清洗双色球数据
    
    Args:
        filename: JSON数据文件路径
        
    Returns:
        清洗后的开奖号码列表，如[['4','18','19','24','27','30','16'], ...]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [n['开奖号码'] for n in data]

def build_markov_matrix(df: pd.DataFrame, 
                       ball_type: str = 'red', 
                       order: int = 1) -> Tuple[np.ndarray, List[int]]:
    """构建马尔科夫转移矩阵
    
    Args:
        df: 包含开奖数据的DataFrame
        ball_type: 'red'或'blue'
        order: 马尔科夫阶数
        
    Returns:
        trans_matrix: 转移概率矩阵
        states: 状态(号码)列表
    """
    if ball_type == 'red':
        balls = df[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values.flatten()
    else:
        balls = df['B'].values
    
    states = sorted(set(balls))
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    # 对于高阶马尔科夫链，状态是顺序相关的
    if order > 1:
        # 生成所有可能的状态序列
        from itertools import product
        state_sequences = list(product(states, repeat=order))
        n_sequences = len(state_sequences)
        seq_to_idx = {seq: i for i, seq in enumerate(state_sequences)}
        
        trans_counts = np.zeros((n_sequences, n_states))
        
        # 统计转移次数
        for i in range(len(balls)-order):
            prev_seq = tuple(balls[i:i+order])
            next_state = balls[i+order]
            
            if prev_seq in seq_to_idx and next_state in state_to_idx:
                trans_counts[seq_to_idx[prev_seq]][state_to_idx[next_state]] += 1
    else:
        # 一阶马尔科夫链
        trans_counts = np.zeros((n_states, n_states))
        for i in range(len(balls)-1):
            prev_state = balls[i]
            next_state = balls[i+1]
            trans_counts[state_to_idx[prev_state]][state_to_idx[next_state]] += 1
    
    # 转换为概率矩阵
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(trans_counts, row_sums, 
                           out=np.zeros_like(trans_counts), 
                           where=row_sums!=0)
    
    return trans_matrix, states

def plot_transition_matrix(matrix: np.ndarray, 
                         states: List[int], 
                         title: str,
                         order: int = 1,
                         min_prob: float = 0.05) -> None:
    """可视化转移矩阵
    
    Args:
        matrix: 转移概率矩阵
        states: 状态(号码)列表
        title: 图表标题
        order: 马尔科夫阶数
        min_prob: 显示的最小概率阈值
        min_prob 可调整参数
    """
    plt.figure(figsize=(20, 12))
    
    # 过滤低概率转移
    filtered_matrix = np.where(matrix >= min_prob, matrix, 0)
    
    if order == 1:
        # 一阶转移矩阵
        sns.heatmap(filtered_matrix,
                   xticklabels=states,
                   yticklabels=states,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.1%',
                   annot_kws={'size': 8},
                   linewidths=0.2,
                   cbar_kws={'shrink': 0.7})
        plt.xlabel('Next State')
        plt.ylabel('Current State')
    else:
        # 高阶转移矩阵需要特殊处理标签
        plt.imshow(filtered_matrix, cmap='YlOrRd')
        plt.colorbar(shrink=0.7)
        plt.title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def get_top_transitions(matrix: np.ndarray, 
                      states: List[int], 
                      order: int = 1,
                      top_n: int = 10) -> List[Tuple]:
    """获取概率最高的转移对
    
    Args:
        matrix: 转移概率矩阵
        states: 状态(号码)列表
        order: 马尔科夫阶数
        top_n: 返回的前N个转移
        
    Returns:
        按概率排序的转移对列表
    """
    transitions = []
    
    if order == 1:
        # 一阶转移
        for i in range(len(states)):
            for j in range(len(states)):
                prob = matrix[i,j]
                if prob > 0:
                    transitions.append((states[i], states[j], prob))
    else:
        # 高阶转移
        from itertools import product
        state_sequences = list(product(states, repeat=order))
        
        for seq_idx in range(len(state_sequences)):
            for state_idx in range(len(states)):
                prob = matrix[seq_idx, state_idx]
                if prob > 0:
                    seq = state_sequences[seq_idx]
                    transitions.append((*seq, states[state_idx], prob))
    
    # 按概率降序排序
    transitions.sort(key=lambda x: x[-1], reverse=True)
    return transitions[:top_n]

def analyze_blue_ball(df: pd.DataFrame, order: int = 2) -> None:
    """专门分析蓝球的转移模式
    
    Args:
        df: 包含开奖数据的DataFrame
        order: 马尔科夫阶数
    """
    # 构建转移矩阵
    matrix, states = build_markov_matrix(df, 'blue', order)
    
    # 获取高频转移
    top_transitions = get_top_transitions(matrix, states, order, 5)
    
    print(f"\nTop {len(top_transitions)} Blue Ball Transitions (Order {order}):")
    for transition in top_transitions:
        *prev_states, next_state, prob = transition
        arrow = " → ".join(map(str, prev_states))
        print(f"{arrow} → {next_state}: {prob:.2%}")
        
        # 验证历史出现次数
        count = 0
        blue_balls = df['B'].values
        for i in range(len(blue_balls)-order):
            if all(blue_balls[i+k] == prev_states[k] for k in range(order)):
                if blue_balls[i+order] == next_state:
                    count += 1
        print(f"  历史出现次数: {count}次")

def main():
    # 加载数据
    data = load_data()
    df = pd.DataFrame(data, columns=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'B'])
    df = df.apply(pd.to_numeric)
    
    # 红球一阶马尔科夫分析
    print("=== 红球一阶马尔科夫分析 ===")
    red_matrix, red_states = build_markov_matrix(df, 'red', 1)
    plot_transition_matrix(red_matrix, red_states, 'Red Ball Transition Matrix (Order 1)')
    
    top_red = get_top_transitions(red_matrix, red_states, 1, 10)
    print("\nTop 10 Red Ball Transitions (Order 1):")
    for from_, to_, prob in top_red:
        print(f"{from_} → {to_}: {prob:.2%}")
    
    # 蓝球二阶马尔科夫分析
    print("\n=== 蓝球二阶马尔科夫分析 ===")
    analyze_blue_ball(df, 2)
    
    # 预测示例
    print("\n=== 预测示例 ===")
    current_red = 4
    predictions = predict_next_red(current_red, red_matrix, red_states)
    print(f"红球{current_red}之后最可能出现的号码:")
    for num, prob in predictions:
        print(f"{num}: {prob:.2%}")

def predict_next_red(current_red: int, 
                   matrix: np.ndarray, 
                   states: List[int]) -> List[Tuple[int, float]]:
    """预测下一个可能出现的红球
    
    Args:
        current_red: 当前红球号码
        matrix: 转移概率矩阵
        states: 状态列表
        
    Returns:
        预测结果列表(号码, 概率)
    """
    state_to_idx = {s: i for i, s in enumerate(states)}
    if current_red not in state_to_idx:
        return []
    
    idx = state_to_idx[current_red]
    probs = matrix[idx]
    sorted_indices = np.argsort(probs)[::-1]
    
    return [(states[i], probs[i]) for i in sorted_indices[:5] if probs[i] > 0]

if __name__ == "__main__":
    main()