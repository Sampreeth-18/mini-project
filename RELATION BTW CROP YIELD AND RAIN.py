import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import numpy as np  
df=pd.read_csv("C:/AI and ML/Datasets/Crop_recommendation.csv") 
print(df.head())
x=df[['CROP YIELD']] 
y=df[['RAINFALL RATE']]
plt.plot(x,y)
plt.show()
X=x.astype(float) 
Y=y.astype(float) 
model=sm.OLS(Y,X).fit() 
print(model.summary())
plt.plot(x,y)
plt.show()
