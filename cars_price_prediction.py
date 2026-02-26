import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer   
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import mlflow
import optuna
from sklearn.model_selection import train_test_split

df = pd.read_csv('cars.csv')

x=df.drop('selling_price',axis=1)
y=df['selling_price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


num_col=['km_driven']
cat_col=['brand','fuel','owner']

num_pipe=Pipeline(steps=[
    ('scaler',StandardScaler())
])

cat_pipe=Pipeline(steps=[
    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])

compose=ColumnTransformer(transformers=[
    ('num',num_pipe,num_col),
    ('cat',cat_pipe,cat_col)
],remainder='passthrough',verbose_feature_names_out=False)

pipe=Pipeline(steps=[
    ('compose',compose),
    ('model',RandomForestRegressor())
])

pipe.fit(x_train,y_train)   

# y_pred=pipe.predict(x_test) 

# r2=r2_score(y_test,y_pred)

# print("R2 Score:",r2)


def objective(trial:optuna.trial.Trial):
    n_estimators=trial.suggest_int('n_estimators',50,2000)
    max_depth=trial.suggest_int('max_depth',5,20)
    min_samples_split=trial.suggest_int('min_samples_split',2,100)
    
    pipe.set_params(model__n_estimators=n_estimators,
                    model__max_depth=max_depth,
                    model__min_samples_split=min_samples_split)
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        # pipe.fit(x_train,y_train)
        y_pred=pipe.predict(x_test)
        r2=r2_score(y_test,y_pred)
        mlflow.log_metric("r2_score",r2)
    
    return r2




with mlflow.start_run(run_name="Optuna_Parent_Run"):
    study=optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=100)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_r2_score",study.best_value)

    print("Best Hyperparameters:",study.best_params)
    print("Best R2 Score:",study.best_value)