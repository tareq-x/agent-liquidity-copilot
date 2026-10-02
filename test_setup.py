import pandas as pd

df = pd.DataFrame({"hour": [9, 10, 11], "cashout": [1200, 3400, 2100]})
print(df)
print("Total:", df["cashout"].sum())