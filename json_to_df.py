import pandas as pd
import json


with open('Double_color_balls\data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(data)

# 拆分开奖号码为R1-R6和B1
for i in range(6):
    df[f"R{i+1}"] = df["开奖号码"].apply(lambda x: x[i] if len(x) > i else None)

df["B1"] = df["开奖号码"].apply(lambda x: x[6] if len(x) > 6 else None)

# 删除原始的开奖号码列
df = df.drop("开奖号码", axis=1)

# 重新排列列顺序
columns_order = [
    "期号",
    "开奖日期",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "B1",
    "一等奖注数",
    "一等奖金额",
    "二等奖注数",
    "二等奖金额",
    "销售额",
    "奖池金额",
    "开奖公告",
]
# 保存为Excel文件
excel_file = "lottery_results.xlsx"  
df.to_excel(excel_file, index=False)  
print(f"Excel 文件已保存为 {excel_file}")

# 保存为CSV文件（解决中文乱码问题）
csv_file = "lottery_results.csv"  # CSV文件名
df.to_csv(csv_file, index=False, encoding="utf_8_sig")  
print(f"CSV 文件已保存为 {csv_file}")
