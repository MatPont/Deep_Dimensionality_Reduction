import sys
import pandas as pd
from statistics import mean, stdev

df = pd.read_csv(sys.argv[1], header=None)
print(df.shape)
for i in range(df.shape[1]):
    print("=====================")
    print(mean(df.iloc[:, i]))
    print(stdev(df.iloc[:, i]))
