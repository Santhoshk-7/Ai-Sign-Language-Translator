import pandas as pd

# load data
data = pd.read_csv("data.csv", header=None)

# keep only rows that are NOT B
data = data[data.iloc[:, -1] != 'B']

# save cleaned data
data.to_csv("data.csv", index=False, header=False)

print("All B data removed successfully!")