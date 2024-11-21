import numpy as np
import pickle
import datetime
import copy

from sklearn.preprocessing import LabelEncoder

from artifact import FixOutArtifact
from fairness import computeFairnessMetrics
from fixout.ttypes import SensitiveFeature, FairMetricEnum

import web.app2 as interface

class FixOutHelper:
    
    def __init__(self,repport_name=""):
         self.repport_name = repport_name

    def common(self, fxa):
        self.model = fxa.model
        self.X_train = fxa.X_train
        self.y_train = fxa.y_train
        self.X_test = fxa.X_test
        self.y_test = fxa.y_test
        self.f_names = fxa.features_name
        self.nonnumeric_features = fxa.nonnumeric_features

        if self.model is None and fxa.y_pred is None:
            raise

        if fxa.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        else:
            self.y_pred = fxa.y_pred
        self.prob_y_pred = fxa.prob_y_pred

        sens_f_indexes = [u for u,_,_ in fxa.sensfeatList]
        sens_f_unprivPops = [v for _,v,_ in fxa.sensfeatList]
        sens_f_unprivPops_discretes = []
        self.sens_f_names = [w for _,_,w in fxa.sensfeatList]

        encoders = []

        transformed_data = copy.deepcopy(self.X_test)
        
        for i in range(len(self.f_names)):
            
            le = None
            
            if i in self.nonnumeric_features:
                le = LabelEncoder( )
                le.fit(self.X_test[:,i])
                transformed_data[:,i] = le.transform(self.X_test[:,i]).astype(float)

            encoders.append(le)

        ######
        # for each column
        for i in range(len(sens_f_indexes)):
            
            sens_f_index = sens_f_indexes[i]

            if sens_f_index in self.nonnumeric_features: 

                le = encoders[sens_f_index]
                sens_f_unprivPops_discreted = int(le.transform([sens_f_unprivPops[i]])[0])
                
                new_array = [1 if x == str(float(sens_f_unprivPops_discreted)) else 0 for x in transformed_data[:,sens_f_index]]
                transformed_data[:,sens_f_index] = np.array(new_array)
            
            else:
                sens_f_unprivPops_discreted = int(sens_f_unprivPops[i])
                    
            sens_f_unprivPops_discretes.append(sens_f_unprivPops_discreted)
        
        
        self.sensitivefeatureslist = []
        
        self.sens_f_indexes = sens_f_indexes
        # for each sensitive feature
        for i in range(len(sens_f_indexes)):

            aSensitiveFeature = SensitiveFeature()
            aSensitiveFeature.featureIndex = sens_f_indexes[i] 
            aSensitiveFeature.unprivPop = sens_f_unprivPops_discretes[i]
            aSensitiveFeature.unprivPop_original = sens_f_unprivPops[i]
            aSensitiveFeature.name = self.sens_f_names[i]
            aSensitiveFeature.description = ""
            aSensitiveFeature.type = 1 if sens_f_indexes[i] in self.nonnumeric_features else 0
            self.sensitivefeatureslist.append(aSensitiveFeature)
        
        ######

        transformed_data = transformed_data.astype(float)
        self.X_test = transformed_data

        self.metricsToBeCalculated = [FairMetricEnum.DP, FairMetricEnum.EO, FairMetricEnum.PE, FairMetricEnum.EOD]

        self.result = []
        for sensitiveFeature in self.sensitivefeatureslist:
            r = computeFairnessMetrics(True, 
                                        self.metricsToBeCalculated, 
                                        sensitiveFeature, 
                                        self.X_test.tolist(), 
                                        self.y_test.tolist(),
                                        self.y_pred)
            self.result.append((sensitiveFeature,r))        

    def run(self, fxa):
        self.common(fxa)

        self.repport = {
            "report_details":{
                "repport_name": self.repport_name,
                "generated": datetime.datetime.now().date()
            },
            "model_availability": self.model is not None,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "f_names": self.f_names,
            "sens_f_index": self.sens_f_indexes,
            "sens_f_names": self.sens_f_names,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "prob_y_pred": None, # Fix it
            "y_pred": self.y_pred,
            "sens_f_unpriv": [x.unprivPop for x in self.sensitivefeatureslist],
            "sens_f_unpriv_original": [x.unprivPop_original for x in self.sensitivefeatureslist],
            "sens_f_type": [1 if x in self.nonnumeric_features else 0 for x in self.sensitivefeatureslist],
            "sens_f_pair": [(x.featureIndex, x.name) for x in self.sensitivefeatureslist],
            "fair_metrics": self.result,
            "metrics_list": self.metricsToBeCalculated
        }
        

        pickle.dump(self.repport,open(str("repport_output.fixout"),"wb"))

        interface.app.run()
        return self.repport