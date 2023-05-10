import pandas as pd

for i in range(1000):
    Requests=pd.read_csv('./req_%d.csv'%i)
    print(len(Requests.loc[Requests['request']==1]))
