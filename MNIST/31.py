import pandas as pd

# CSVファイルの読み込み
file_path = 'testrevkrum_result.csv'  # CSVファイルのパス
df = pd.read_csv(file_path)

# ラウンド31のデータを抽出
round_31_data = df[df['round'] == 7]

# 結果を表示
print(round_31_data)

# 必要に応じて結果をCSVファイルとして保存
output_path = 'round11__data.csv'
round_31_data.to_csv(output_path, index=False)
