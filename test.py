import pandas as pd

df1= pd.read_csv("datasets/3_512_x_main.csv")
df2= pd.read_csv("datasets/3_512_y_main.csv")
df3 = pd.concat([df1, df2] , axis = 1)
df3.to_csv("datasets/xy.csv")