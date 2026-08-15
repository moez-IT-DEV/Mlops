import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.pyplot import figure

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from imblearn.over_sampling import SMOTE
import  mlflow
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

EXPERIMENTNAME="ModelingV2"
version="1"
MLFLOW_TRACKING_URI="https://dagshub.com/moezkchaoumail/Mlops.mlflow"
MLFLOW_TRACKING_USERNAME="moezkchaoumail"
MLFLOW_TRACKING_PASSWORD="moez123."
os.environ['MLFLOW_TRACKING_USERNAME'] =MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENTNAME)

df = pd.read_csv("./backend/src/fraud_v1.csv")

def transform_obbject_todate(df,col_names):
    for col in col_names:
        #convert trans_date_trans_time , dob to datetime
        df[col] = pd.to_datetime(df[col])
    return df
col_todate=["trans_date_trans_time","dob"]
df = transform_obbject_todate(df,col_todate)

df["year_trans"]=df["trans_date_trans_time"].dt.year
df["month_trans"]=df["trans_date_trans_time"].dt.month
df["day_trans"]=df["trans_date_trans_time"].dt.day
df['age']=dt.date.today().year-pd.to_datetime(df['dob']).dt.year


#dropping variables
df.drop(['trans_date_trans_time','first', 'last', 'dob','trans_num','cc_num','unix_time','merchant', 'street', 'city', 'state', 'job'] , axis=1, inplace=True)


#Only for gender
encode_dict = {  # Encoding dictionary
    'F': 0, 'M': 1}
df['gender'] = df['gender'].map(encode_dict)
dummy_cols = ['category']
#For other categorical columns
df = pd.get_dummies(df, columns=dummy_cols)


X = df.drop('is_fraud', axis=1)  # Select features
y = df['is_fraud']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)  # Split data 80/20


num_features= df.select_dtypes(include=['integer']).columns.tolist()
num_features.remove("is_fraud")
scaler = MinMaxScaler()  # Normalize train & test features
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

method= SMOTE()
x_resampled, y_resampled = method.fit_resample(X_train, y_train)
scaler = MinMaxScaler()  # Normalize train & test features


mlflow.sklearn.autolog(registered_model_name="LogisticRegression")

RUN_NAME="LogisticRegression1"
with mlflow.start_run(run_name=RUN_NAME):
    print("test")
    mlflow.log_param("data_version",version)
    mlflow.log_param("input_rows",df.shape[0])
    mlflow.log_param("input_cols",df.shape[1])
    #model fitting and training
    model=LogisticRegression(solver='lbfgs' ,max_iter=10000)
    model.fit(x_resampled,y_resampled)
    predicted=model.predict(X_test)
    precision,recall,fscore,support=score(y_test,predicted,average='macro')
    mlflow.log_metric("Precision_metric",precision)
    mlflow.log_metric("Recall_metric",recall)
    mlflow.log_metric("F1_score_metric",fscore)

mlflow.sklearn.autolog(registered_model_name="RandomForest")
RUN_NAME="RandomForest1"
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("data_version",version)
    mlflow.log_param("input_rows",df.shape[0])
    mlflow.log_param("input_cols",df.shape[1])
    #model fitting and training
    model2 = RandomForestClassifier(random_state=5)
    model2.fit(x_resampled,y_resampled)
    predicted=model2.predict(X_test)
    precision,recall,fscore,support=score(y_test,predicted,average='macro')
    mlflow.log_metric("Precision_metric",precision)
    mlflow.log_metric("Recall_metric",recall)
    mlflow.log_metric("F1_score_metric",fscore)

#Reading Pandas Dataframe from mlflow
df=mlflow.search_runs(filter_string="metrics.F1_score_metric < 1")
run_id = df.loc[df['metrics.F1_score_metric'].idxmax()]['run_id']
print(run_id)
#run_name = df.loc[df['metrics.F1_score_metric'].idxmax()]['tags.mlflow.runName']
#print(run_name)
model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
with open('./backend/models/best_model_2.pkl','wb') as f:
    pickle.dump(model,f)
    
