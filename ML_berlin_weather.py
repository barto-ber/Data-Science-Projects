import pandas as pd
import numpy as np
pd.options.display.width = 0
pd.options.display.max_rows = None

data = pd.read_csv('berlin_klima_1948_2019_en.txt', sep=';')
less_columns = ["Station_ID", "QN_3", "QN_4", "VPM", "eor"]
data.drop(less_columns, inplace=True, axis=1)
print(data.info())
# print(data.head())
print("Check -999:\n", data[(data['Max_Wind_Speed'] > -999)])