import pandas as pd

p = r"data/raw/CICIDS-2017/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
df = pd.read_csv(p, encoding="cp1252", low_memory=False)
print(df.columns[:10], df.columns[-5:])
print(df[df.columns[-1]].value_counts(dropna=False).head(20))  # последний столбец должен быть Label
