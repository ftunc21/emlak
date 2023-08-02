import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("multilinearregression.csv",sep=";")
reg = linear_model.LinearRegression()
reg.fit(df[['alan','odasayisi','binayasi']],df['fiyat'])#ilk sırada yazılanlar bağımsız değişken virgülden sonra yazılanlar bağımlı değişken
prices=[[230,4,10],[230,6,0],[355,3,20]]
reg.predict(prices)