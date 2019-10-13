import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('../data/train.csv',delimiter=',',header=None)

df_train,df_val=train_test_split(df.values,test_size=0.25)

#df_final,df_val=train_test_split(df_train,test_size=0.25)

pd.DataFrame(df_train).to_csv('../data/final.csv',header=False,index=False)
#pd.DataFrame(df_test).to_csv('./data/test.csv',header=False,index=False)
pd.DataFrame(df_val).to_csv('../data/val.csv',header=False,index=False)

print(df_train.size)
#print df_test.size
print(df_val.size)