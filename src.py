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
print("libraries imported successfully")

print("step 2: Loading the dataset and created dataframe successfully")
url="https://drive.google.com/file/d/1TfyII57-mXyc6HCLi9yZGIHC85ncg42O/view?usp=drive_link"
df=pd.read_csv(url)
print(df.head())

