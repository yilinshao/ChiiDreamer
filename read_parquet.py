import pandas as pd

df = pd.read_parquet('/data/datasets/svg/train-00000-of-00001.parquet')
rows = df.values.tolist()

# 查看数据
print(df.head())  # 显示前几行
print(df.info())  # 显示数据信息