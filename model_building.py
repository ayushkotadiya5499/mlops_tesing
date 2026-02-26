import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import joblib

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def split_data(df:pd.DataFrame,target_column):
    x=df.drop(target_column,axis=1)
    y=df[target_column]
    return x,y

def model_building(x,y):

    # model=RandomForestRegressor(n_estimators=595, max_depth=17, min_samples_split=40)
    model=RandomForestRegressor(max_depth=100, n_estimators=10, random_state=42)
    model.fit(x,y)
    return model

def evaluate_model(model,x,y):
    y_pred=model.predict(x)
    r2=r2_score(y,y_pred)
    return r2

df=load_data('preprocessed_data/preprocessed_train.csv')
x,y=split_data(df,'selling_price')
model=model_building(x,y)
r2=evaluate_model(model,x,y)
print(f'R2 Score: {r2}')


os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/ml_model.pkl')

