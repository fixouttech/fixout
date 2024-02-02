"""
This is a demo of how to connect and use the FixOut API
"""
from basic_ml import mainClient

from connector import FixOutConnector, FixOutArtifact
from fixout.ttypes import FairMetricEnum


def main():
    
    model, X_train, X_test, y_train, y_test, features_name = mainClient()
    X_train, X_test = X_train.tolist(), X_test.tolist()

    y_pred = model.predict(X_test)

    # ###########
    # FixOut 
    # ###########

    fx = FixOutConnector('localhost', 9090)
    fx.connect("3f6ba732c46a9a0e6255f245247508cc","D13 Credit Risk")

    sensitive_features = [(19,0,"foreignworker"), 
                          (18,1,"telephone"), 
                          (8,2,"statussex")] # (no), (yes), (male single) 

    fxa = FixOutArtifact(model=model,
                         X_train=X_train, 
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         features_name=features_name,
                         sensitive_features=sensitive_features,
                         nonnumeric_features=[])
    
    fx.load(fxa)  
    fx.inspect([FairMetricEnum.DP, FairMetricEnum.EO, FairMetricEnum.PE])
    fx.explain()
    newModel = fx.debias()
    fx.disconnect(sendMail=True)

    # ###########
    # ###########

if __name__ == '__main__':

    main()
    print("Finished")