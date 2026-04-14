from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
def RFC_model(X , y):
    model1 = RandomForestClassifier()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42, shuffle=True)
    sm=SMOTE(random_state=42)
    X_train_sm , y_train_sm = sm.fit_resample(X_train,y_train)

    
    model1.fit(X_train_sm,y_train_sm)


    return model1 , X_test , y_test