import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#Reads vehicle conditions that are stored in app_data_input.csv
X_test = pd.read_csv("vehicle_conditions.csv",index_col=0)

#Preprocess the data such that they can be used by the model. 

def preprocess_data(df):
    
    #convert last service date and warranty expiry date into date time
    df["Last_Service_Date"] = pd.to_datetime(df["Last_Service_Date"],format="%d/%m/%Y")
    df["Warranty_Expiry_Date"] = pd.to_datetime(df["Warranty_Expiry_Date"],format="%d/%m/%Y")

    #Preprocess data such that it is express as warranty days remaining/days since last service. 
    df["Days_since_last_service"] = (pd.Timestamp.today() - df["Last_Service_Date"]).dt.days
    df["Warranty Days remaining"] = (df["Warranty_Expiry_Date"] - pd.Timestamp.today()).dt.days

    #Drop original last_service_date and warranty_expiry date cols
    #Drop Vehicle Model as all truck
    df = df.drop(columns = ['Last_Service_Date','Warranty_Expiry_Date','Vehicle_Model','Fuel_Type'], axis = 1)


    #Select categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include = ['int','float']).columns


    oneHotEncoder = OneHotEncoder()

    #Convert ordinal data into numerical categories
    # pd.Categorical(df["Tire_Condition"],categories=["Worn Out","Good","New"],ordered=True) adds ordering to data Worn Out < Good < New but does not convert into binary
    df["Tire_Condition"] = pd.Categorical(df["Tire_Condition"],categories=["Worn Out","Good","New"],ordered=True).codes
    df["Brake_Condition"] = pd.Categorical(df["Brake_Condition"],categories=["Worn Out","Good","New"],ordered=True).codes
    # X_train["Battery_Status"] = pd.Categorical(X_train["Battery_Status"],categories=["Weak","Good","New"],ordered=True).codes
    df["Owner_Type"] = pd.Categorical(df["Owner_Type"],categories=["First","Second","Third"],ordered=True).codes
    df["Maintenance_History"] = pd.Categorical(df["Maintenance_History"],categories=["Poor","Average","Good"],ordered=True).codes


    # "Battery_Status"
    labelEncodedCols = ["Tire_Condition", "Brake_Condition","Owner_Type","Maintenance_History"]

    #Essentially Transmission type and fuel type.
    oneHotEncodedCols = list(set(categorical_cols) - set(labelEncodedCols))
    df = pd.get_dummies(df,columns=oneHotEncodedCols,drop_first=True)


    numerical_df = df[numerical_cols]
    numerical_data = df[numerical_cols]


    df_cols = df.columns

    df = pd.DataFrame(data=df,columns=df_cols)

    return df



X_test = preprocess_data(X_test)

#Load pre-trained model.
import pickle
with open('rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


#Predicts whether maintenance is needed for vehicle. 
y_pred_rf = loaded_model.predict(X_test)


#add truck IDS to vehicle. 
truck_IDs = X_test.index
X_test.insert(0, 'Truck_ID', truck_IDs)
#Add predictions to data
X_test["Need_Maintenance"] = y_pred_rf


X_test.to_csv("truck_chat_data.csv")

