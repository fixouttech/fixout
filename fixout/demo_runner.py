"""
This is a demo of how to use the open-source version of FixOut
"""
from demo_data import importGermanData

from artifact import FixOutArtifact
from helper import FixOutHelper

def demo():
    
    model, X_train, X_test, y_train, y_test, features_name, dic = importGermanData()

    fixout = FixOutHelper("Credit Risk Assessment (German bank)") 

    sensitive_features = [(19,0,"foreignworker"), 
                          (18,1,"telephone"), 
                          (8,2,"statussex")] # (no), (yes), (male single) 

    fxa = FixOutArtifact(model=model,
                         training_data=(X_train,y_train), 
                         testing_data=[(X_test,y_test,"1")],
                         features_name=features_name,
                         sensitive_features=sensitive_features,
                         dictionary=dic)
    
    fixout.run(fxa, dic) 

    

if __name__ == '__main__':

    demo()