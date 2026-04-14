import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from preprocess import preprocess
from train import RFC_model
from evaluate import pred
 
filepath = "./data/Loan Prediction.csv"
X,y = preprocess(filepath)
model , X_test ,y_test = RFC_model(X,y)
prediction = pred(model , X_test , y_test)
