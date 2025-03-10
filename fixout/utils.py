import sklearn
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,RandomForestClassifier
from fixout.interface.ttypes import ModelType

def getModelType(model):

    if model.__class__ is sklearn.linear_model.LogisticRegression : 
        return ModelType.LR
    elif model.__class__ is AdaBoostClassifier:
        return ModelType.ADA
    elif model.__class__ is BaggingClassifier : 
        return ModelType.BAG
    elif model.__class__ is RandomForestClassifier :
        return ModelType.RF
    
    return None
