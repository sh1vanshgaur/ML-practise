import pandas as pd

def preprocess(filepath):
    df = pd.read_csv(filepath)

    df = df.drop(["Id" , "CITY" , "STATE", "Profession"] , axis=1)
    df["Married/Single" ] = df["Married/Single"].map({"single" : 0 , "married" :1})
    df["Car_Ownership"] = df["Car_Ownership"].map({"no" : 0 , "yes" :1})
    df = pd.get_dummies(df, columns=["House_Ownership"] , dtype=int)
    df = df.fillna(df.mean())
    X = df.drop("Risk_Flag", axis=1)
    y=df["Risk_Flag"]

    return X,y