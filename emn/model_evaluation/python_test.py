import pandas as pd
import os
import sys



df = pd.DataFrame()

df['ten'] = pd.Series(range(10, 50, 5), index=range(1,10, 1))

print(df)
