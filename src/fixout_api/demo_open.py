"""
This is a demo of how to use the open-source version of FixOut
"""
from basic_ml import mainClient

from artifact import FixOutArtifact
from helper import FixOutHelper

def main():
    
    model, X_train, X_test, y_train, y_test, features_name = mainClient()

    print(X_train)

    fixout = FixOutHelper("Credit Risk Assessment") 

    sensitive_features = [(19,0,"foreignworker"), 
                          (18,1,"telephone"), 
                          (8,2,"statussex")] # (no), (yes), (male single) 

    fxa = FixOutArtifact(model=model,
                         X_train=X_train.tolist(), 
                         y_train=y_train,
                         X_test=X_test.tolist(),
                         y_test=y_test,
                         features_name=features_name,
                         sensitive_features=sensitive_features)
    
    fixout.run(fxa) 

if __name__ == '__main__':

    main()
    print("Finished")