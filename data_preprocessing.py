import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import joblib

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def preprocess_data(df:pd.DataFrame, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Define transformers for both types of data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_cols)
        ],remainder='passthrough',verbose_feature_names_out=False)
    

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    x_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
    y=pd.DataFrame(y)
    processed_df=pd.concat([x_processed_df,y],axis=1)

    os.makedirs('preprocesser', exist_ok=True)
    joblib.dump(preprocessor, 'preprocesser/preprocessor.pkl')
    
    return processed_df

train_df=load_data('data/train_data.csv')
# test_df=load_data('data/test_data.csv')

target_column='selling_price'

preprocessed_train_df=preprocess_data(train_df,target_column)
# preprocessed_test_df=preprocess_data(test_df,target_column)



os.makedirs('preprocessed_data', exist_ok=True)
preprocessed_train_df.to_csv('preprocessed_data/preprocessed_train.csv', index=False)
# preprocessed_test_df.to_csv('preprocessed_data/preprocessed_test.csv', index=False)