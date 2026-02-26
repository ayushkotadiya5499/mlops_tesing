import pandas as pd
import os 
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def split_data(df,test_size=0.2,random_state=42):
    train_df,test_df=train_test_split(df,test_size=test_size,random_state=random_state)   
    return train_df,test_df

df=load_data('cars.csv')
train_df,test_df=split_data(df)

os.makedirs('data', exist_ok=True)
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)