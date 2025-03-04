import numpy as np
import pickle
import datetime
import copy

from sklearn.preprocessing import LabelEncoder

from artifact import FixOutArtifact
from fairness import computeFairnessMetrics
from fixout.ttypes import SensitiveFeature, FairMetricEnum

from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import web.webapp as interface

clazzes = [(linear_model.RidgeClassifier,"linear reg"),
            (tree.DecisionTreeClassifier,"tree"),
            (svm.SVC,"svm"),
            (GaussianNB,"gaussian"),
            (RandomForestClassifier,"rd forest"),
            (MLPClassifier,"neural"),
            (GradientBoostingClassifier,"grad boost")]

class FixOutHelper:
    
    def __init__(self,repport_name=""):
        self.input = {}
        self.input["report_details"] = {}
        self.input["report_details"]["repport_name"] = repport_name
        self.input["report_details"]["generated"] = datetime.datetime.now().date()

        self.output = {}

    def common(self, fxa):
         
        self.input["model"] = fxa.model
        self.input["X"] = fxa.X
        self.input["y"] = fxa.y
        self.input["f_names"] = fxa.features_name
        self.input["nonnumeric_features"] = fxa.nonnumeric_features

        self.input["testing_data"] = fxa.test_data
        
        self.input["dictionary"] = fxa.dictionary 

        if self.input["model"] is None and fxa.y_pred is None:
            raise

        if fxa.y_pred is None:
            self.output["y_pred"] = self.input["model"].predict(self.input["X"])
        else:
            self.output["y_pred"] = fxa.y_pred
        self.prob_y_pred = fxa.prob_y_pred

        sens_f_indexes = [u for u,_,_ in fxa.sensfeatList]
        sens_f_unprivPops = [v for _,v,_ in fxa.sensfeatList]
        sens_f_unprivPops_discretes = []
        self.input["sens_f_names"] = [w for _,_,w in fxa.sensfeatList]

        encoders = []

        transformed_data = copy.deepcopy(self.input["X"])
        
        for i in range(len(self.input["f_names"])):
            
            le = None
            
            if i in self.input["nonnumeric_features"]:
                le = LabelEncoder( )
                le.fit(self.input["X"][:,i])
                transformed_data[:,i] = le.transform(self.input["X"][:,i]).astype(float)

            encoders.append(le)

        self.input["sens_f_index"] = sens_f_indexes
        
        ######
        # for each column
        for i in range(len(self.input["sens_f_index"])):
            
            sens_f_index = self.input["sens_f_index"][i]

            if sens_f_index in self.input["nonnumeric_features"]: 

                le = encoders[sens_f_index]
                sens_f_unprivPops_discreted = int(le.transform([sens_f_unprivPops[i]])[0])
                
                new_array = [1 if x == str(float(sens_f_unprivPops_discreted)) else 0 for x in transformed_data[:,sens_f_index]]
                transformed_data[:,sens_f_index] = np.array(new_array)
            
            else:
                sens_f_unprivPops_discreted = int(sens_f_unprivPops[i])
                    
            sens_f_unprivPops_discretes.append(sens_f_unprivPops_discreted)
        
        
        self.sensitivefeatureslist = []
        
        # for each sensitive feature
        for i in range(len(self.input["sens_f_index"])):

            aSensitiveFeature = SensitiveFeature()
            aSensitiveFeature.featureIndex = self.input["sens_f_index"][i] 
            aSensitiveFeature.unprivPop = sens_f_unprivPops_discretes[i]
            aSensitiveFeature.unprivPop_original = sens_f_unprivPops[i]
            aSensitiveFeature.name = self.input["sens_f_names"][i]
            aSensitiveFeature.description = ""
            aSensitiveFeature.type = 1 if self.input["sens_f_index"][i] in self.input["nonnumeric_features"] else 0
            self.sensitivefeatureslist.append(aSensitiveFeature)
        
        ######

        transformed_data = transformed_data.astype(float)
        self.input["X"] = transformed_data
        
        self.input["model_availability"] = self.input["model"] is not None
        self.input["sens_f_unpriv"] = [x.unprivPop for x in self.sensitivefeatureslist],
        self.input["sens_f_unpriv_original"] = [x.unprivPop_original for x in self.sensitivefeatureslist],
        self.input["sens_f_type"] = [1 if x in self.input["nonnumeric_features"] else 0 for x in self.sensitivefeatureslist],
        self.input["sens_f_pair"] = [(x.featureIndex, x.name) for x in self.sensitivefeatureslist]
        
        self.output["prob_y_pred"] = None, # Fix it
        
        rev_fairness = ReverseFairness()
        self.input["reversed_models"] = []
        rev_train = rev_fairness.build_reversed_models(self.input["X"], self.input["y"], self.sensitivefeatureslist)
        self.input["reversed_models"].append(rev_train)
        for X_test,y_test,_ in self.input["testing_data"]:
            rev_test = rev_fairness.build_reversed_models(self.input["X"], self.input["y"], self.sensitivefeatureslist, X_test, y_test)
            self.input["reversed_models"].append(rev_test)
    
        unfair_model = UnfairModel()
        self.input["unfair_model"] = []
        unfair_train =  unfair_model.build_unfair_model(self.input["X"], self.input["y"], self.sensitivefeatureslist)
        self.input["unfair_model"].append(unfair_train)
        for X_test,y_test,_ in self.input["testing_data"]:
            runfair_test = unfair_model.build_unfair_model(self.input["X"], self.input["y"], self.sensitivefeatureslist, X_test, y_test)
            self.input["unfair_model"].append(runfair_test)


        self.assess_fairness()


    def assess_fairness(self):

        self.output["metrics_list"] = [FairMetricEnum.DP, FairMetricEnum.EO, FairMetricEnum.PE, FairMetricEnum.EOD]
        self.output["nonStandardMetricsToBeCalculated"] = [FairMetricEnum.PP, FairMetricEnum.CEA]

        self.output["result"] = self.eval_fairness(self.output["metrics_list"],
                                                   self.sensitivefeatureslist,
                                                   self.input["X"].tolist(),
                                                   self.input["y"].tolist(),
                                                   self.output["y_pred"],
                                                   "original")
        
        self.output["nonstandardResults"] = self.eval_fairness(self.output["nonStandardMetricsToBeCalculated"],
                                                               self.sensitivefeatureslist,
                                                               self.input["X"].tolist(),
                                                               self.input["y"].tolist(),
                                                               self.output["y_pred"],
                                                               "original")
        self.baselines()

    def eval_fairness(self,metrics,sensFeatures,X,y,y_pred,txtIndicator):
        
        results = []
        for sensitiveFeature in sensFeatures:
            r = computeFairnessMetrics(metrics,
                                       sensitiveFeature, 
                                       X, 
                                       y,
                                       y_pred)
            results.append((sensitiveFeature,r,txtIndicator))
        
        return results

    
    def run(self, fxa, dic=None):
        self.common(fxa)
                
        pickle.dump((self.input, self.output),open(str("repport_output.fixout"),"wb"))

        interface.app.run()
        return self.output, None
    

    def baselines(self):

        predictions_list=[]

        for clazz,clazz_name in clazzes:
            self.build_model(clazz,clazz_name,predictions_list)

        for name_method, preditions in predictions_list:
            for sensitiveFeature in self.sensitivefeatureslist:
                r = computeFairnessMetrics(self.output["metrics_list"],
                                        sensitiveFeature, 
                                        self.input["X"].tolist(), 
                                        self.input["y"].tolist(),
                                        preditions)
                self.output["result"].append((sensitiveFeature,r,name_method))

                nonStandardR = computeFairnessMetrics(self.output["nonStandardMetricsToBeCalculated"],
                                       sensitiveFeature, 
                                       self.input["X"].tolist(), 
                                       self.input["y"].tolist(),
                                       preditions)
                self.output["nonstandardResults"].append((sensitiveFeature,nonStandardR,name_method))

    def build_model(self, clazz, name_method, predictions_list):

        clf = clazz()
        clf.fit(self.input["X"], self.input["y"])
        y_pred = clf.predict(self.input["X"])
        predictions_list.append((name_method, y_pred))


# definir funcao para predizer Y somente com as variaveis sensiveis (e correlacionadas)


class ReverseFairness():

    def build_reversed_models(self, X, y, sensitivefeatureslist, X_test=None, y_test=None):

        results = {}
        _X_test = None
        _y_test = None

        _y = [[i] for i in y]
        if y_test is not None:
            _y_test = [[i] for i in y_test]
        
        for sens_feature in sensitivefeatureslist:
            
            results[sens_feature.name] = []
            _X = X[:,sens_feature.featureIndex]
            if X_test is not None:
                _X_test = X_test[:,sens_feature.featureIndex]

            for clazz,clazz_name in clazzes:
                r = self.build_reverse_model(clazz, _X, _y, _X_test, _y_test) 
                results[sens_feature.name].append((clazz_name,r))

        print(results)

        return results


    def build_reverse_model(self, clazz, X, y, X_test=None, y_test=None):

        clf = clazz()
        clf.fit(y, X)
        #X_pred = clf.predict(y)
        
        #baccuracy = balanced_accuracy_score(X,X_pred)
        #precision = precision_score(X, X_pred, average='weighted')
        #recall = recall_score(X, X_pred, average='weighted')

        if X_test is not None and y_test is not None:
            return self.eval_perf(clf, y_test, X_test)
        else:
            return self.eval_perf(clf, y, X)
        #return baccuracy, precision, recall

    def eval_perf(self, clf, X_test, y_test):

        y_pred = clf.predict(X_test)
        
        baccuracy = balanced_accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        return baccuracy, precision, recall

    

class UnfairModel():

    def build_unfair_model(self, X, y, sensitivefeatureslist, X_test=None, y_test=None):

        results = []

        indexes = [x.featureIndex for x in sensitivefeatureslist]
        _X = X[:,indexes]
        
        for clazz,clazz_name in clazzes:
            
            clf = clazz()
            clf.fit(_X, y)
            #y_pred = clf.predict(_X)
            
            #baccuracy = balanced_accuracy_score(y,y_pred)
            #precision = precision_score(y, y_pred)
            #recall = recall_score(y, y_pred)

            if X_test is not None and y_test is not None:
                _X_test = X_test[:,indexes]
                results.append((clazz_name,self.eval_perf(clf, _X_test, y_test)))
            else:
                results.append((clazz_name,self.eval_perf(clf, _X, y)))

        return results

    def eval_perf(self, clf, X_test, y_test):

        y_pred = clf.predict(X_test)
        
        baccuracy = balanced_accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        return baccuracy, precision, recall
