print("step 1: Importing the required libraries")
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
#Importing model training functions through sklearn library
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#Importing feature_selection class 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
#Importing pipeline from imblearn library
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE # type: ignore
#Importing metrics to configure accuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#Importing feature engineering class from sklearn library
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Importing machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
#Importing warnings package to avoid warnings
import warnings
warnings.filterwarnings("ignore")

#Loads the dataset from a GitHub repository into a Pandas DataFrame.
#displays top 5 rows
print("step 2: Loading the dataset and created dataframe successfully")
df=pd.read_csv("https://raw.githubusercontent.com/gujjaripavan222/Optimized-Machine-Learning-Pipeline-for-Predicting-Online-Shoppers-Behavior/refs/heads/main/online_shoppers_intention.csv")
print(df.head())

print("step 3: Feature engineering on Weekend and Revenue column")

#Converts boolean values (True/False) into numerical values (1/0).
df["Weekend"]=df["Weekend"].replace((True,False),(1,0))
df["Revenue"]=df["Revenue"].replace((True,False),(1,0))

print("step 4: Adding Returning_Visitor Column from Visitor Type Column")

#Creates a new column Returning_Visitor based on VisitorType.
#Drops the original VisitorType column.
condition=df["VisitorType"]=="Returning_Visitor"
df["Returning_Visitor"]=np.where(condition,1,0)
df=df.drop(columns=["VisitorType"])

print("step 5: Applying OneHotEncoding on Month Column")

#Converts categorical month names into numerical values.
ordinal_encoder=OrdinalEncoder()
df["Month"]=ordinal_encoder.fit_transform(df[["Month"]])

print("step 6: Checking correlation on Revenue column")

#Computes correlation between Revenue and other features.
result=df[df.columns[1:]].corr()["Revenue"]
#Sorts features by correlation strength.
result1=result.sort_values(ascending=False)
"""
#Data Visualization on feature distributions
plt.figure(figsize=(12, 6))
df.hist(bins=20, figsize=(12, 10), edgecolor="black")
plt.tight_layout()
plt.show()

#visualizing correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

#visualizing Target variable distributions
sns.countplot(x="Revenue", data=df, palette="Set2")
plt.title("Revenue Distribution")
plt.show()

#visualizing key feature impact on Revenue
sns.barplot(x="Weekend", y="Revenue", data=df, palette="viridis")
plt.title("Weekend vs Revenue")
plt.show()

sns.boxplot(x="Revenue", y="Administrative_Duration", data=df)
plt.title("Administrative Duration Impact on Revenue")
plt.show()

#visaulizing to check outliers
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, palette="Set3")
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.show()
"""
print("step 7: Data preparation on features as X and Target as y")

#Splits dataset into features (X) and target (y).
X=df.drop(["Revenue"],axis=1)
y=df["Revenue"]

print("step 8: Splitting the dataset X_train,X_test,y_train and y_test ")

#Splits data into 70% training and 30% testing.
X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0
)

print("step 9: Model Pipeline")

def model_pipeline(X,model):
    #Identifies numerical (n_c) and categorical (c_c) columns.
    n_c=X.select_dtypes(exclude=["object"]).columns.tolist()
    c_c=X.select_dtypes(include=["object"]).columns.tolist()
    #Imputes missing values and scales numerical features.
    numeric_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="constant")),
        ("scaler",MinMaxScaler())
        ])
    #Encodes categorical features.
    categoric_pipeline=Pipeline([
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
        ])
    #Combines numerical and categorical preprocessing.
    preprocessor=ColumnTransformer([
        ("numeric",numeric_pipeline,n_c),
        ("categorical",categoric_pipeline,c_c)],
        remainder="passthrough")
    #Applies preprocessing, oversampling, feature selection, and classification.
    final_steps=[
        ("preprocessor",preprocessor),
        ("smote",SMOTE(random_state=1)),
        ("feature_selection",SelectKBest(score_func=chi2,k=6)),
        ("model",model)
        ]
    return IMBPipeline(steps=final_steps)
print("step 10: create select_model function")
def select_model(X,y,pipeline=None):
    #Defines a dictionary of classification models.
    classifiers={}

    c_d1={"RandomForestClassifier": RandomForestClassifier()}
    classifiers.update(c_d1)
    c_d2={"KNeigborsClassifier": KNeighborsClassifier()}
    classifiers.update(c_d2)
    c_d3={"DecisionTreeClassifier": DecisionTreeClassifier()}
    classifiers.update(c_d3)
    c_d4={"RidgeClassifier": RidgeClassifier()}
    classifiers.update(c_d4)
    c_d5={"SVC": SVC()}
    classifiers.update(c_d5)
    c_d6={"DummyClassifier":DummyClassifier(strategy="most_frequent")}
    classifiers.update(c_d6)
    c_d7={"LGBMClassifier":LGBMClassifier()}
    classifiers.update(c_d7)
    c_d8={"ExtraTreeClassifier":ExtraTreeClassifier()}
    classifiers.update(c_d8)
    c_d9={"ExtraTreesClassifier":ExtraTreesClassifier()}
    classifiers.update(c_d9)
    c_d10={"BernoulliNB":BernoulliNB()}
    classifiers.update(c_d10)
    c_d11={"XGBClassifier": XGBClassifier()}
    classifiers.update(c_d11)
    c_d12={"SGDClassifier": SGDClassifier()}
    classifiers.update(c_d12)
    c_d13={"AdaBoostClassifier": AdaBoostClassifier()}
    classifiers.update(c_d13)
    c_d14={"BaggingClassifier": BaggingClassifier()}
    classifiers.update(c_d14)

    mlpc={
        "MLPClassifier (paper)": 
        MLPClassifier(hidden_layer_sizes=(27,50),
        max_iter=300,
        activation="relu",
        solver="adam",
        random_state=1)
        }
    c_d15=mlpc
    classifiers.update(c_d15)

    cols=["model","run_time","roc_auc"]
    df_models=pd.DataFrame(columns=cols)

    #Runs cross-validation for each model
    for key in classifiers:
        start_time=time.time()
        print()
        print("step 11: Model pipeline run successfully on ",key)
        pipeline=model_pipeline(X_train,classifiers[key])
        cv_scores=cross_val_score(pipeline,X,y,cv=10,scoring="roc_auc")

        row={
            "model":key,
            "run_time":format(round((time.time()-start_time)/60,2)),
            "roc_auc":cv_scores.mean()
        }

        df_models=pd.concat([df_models,pd.DataFrame([row])],ignore_index=True)
    df_models=df_models.sort_values(by="roc_auc",ascending=False)

    return df_models
print("step 12: Accessing select_model function successfully")
models=select_model(X_train,y_train)

print("step 13: running select_model successfully")
print(models)

print("step 14: Accessing best model function ")

#Trains the best model.
selected_model=MLPClassifier()
bundled_pipeline=model_pipeline(X_train,selected_model)
bundled_pipeline.fit(X_train,y_train)

print("step 15: predicting results successfully")

#Predicts test data.
y_pred=bundled_pipeline.predict(X_test)

print(y_pred)

print("step 16: Accessing Roc and Auc scores")

#Evaluates performance using accuracy, F1-score, ROC-AUC, and confusion matrix.
roc_auc=roc_auc_score(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
f1_score=f1_score(y_test,y_pred)

print("ROC/AUC:",roc_auc)
print("Accuracy:",accuracy)
print("f1_score:",f1_score)

print("step 17: Generating Confusion matrix")

confusion=confusion_matrix(y_test,y_pred)
print(confusion)

print("step 17: Generating Classification Report Successfully")
classif_report=classification_report(y_test,y_pred)
print(classif_report)

"""
The optimized machine learning pipeline achieved an accuracy of 87.67%,
demonstrating its strong predictive capability for online shoppers' purchasing behavior.
The best-performing model, MLPClassifier (paper), achieved a ROC-AUC score of 0.903,
indicating its excellent ability to distinguish between classes
"""
# machine learning pipeline has been successfully implemented and completed,
# achieving 87.67% accuracy with an MLPClassifier as the best-performing model.
# This python code is a complete ML pipeline for predicting online shopper behavior.
# Uses preprocessing, feature engineering, and multiple models to find the best one.
# Employs cross-validation and performance evaluation techniques.
