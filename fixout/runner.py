import numpy as np
import pickle
import datetime
import copy

from sklearn.preprocessing import LabelEncoder

from fixout import fairness
from fixout.helper import ReverseFairness, UnfairModel, clazzes
from fixout.interface.ttypes import SensitiveFeature, FairMetricEnum

import warnings
warnings.filterwarnings('ignore')

import fixout.web.webapp as webinterface

from IPython.display import IFrame
import threading

httpaddress = "http://localhost:5000/"

class FixOutRunner:
    """
    A class to process and evaluate fairness in machine learning models, given a FixOut artifact.

    See Also
    --------
    fixout.artifact.FixOutArtifact : A FixOut artifact.
    """
    
    def __init__(self,report_name=""):
        """
        Initializes the FixOutRunner with a report name.
        
        Parameters:
        -----------
        report_name : str, optional
            Name of the report. Defaults to an empty string.
        """
        self.input = {}
        self.input["report_details"] = {}
        self.input["report_details"]["report_name"] = report_name
        self.input["report_details"]["generated"] = datetime.datetime.now().date()

        self.output = {}

    def _common(self, fxa):
         
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


        if len(fxa.sensfeatList) > 0 and type(fxa.sensfeatList[0]) is int: 
            sens_f_indexes = fxa.sensfeatList #[u for u,_,_ in fxa.sensfeatList]
            self.input["sens_f_names"] = [self.input["f_names"][u] for u in fxa.sensfeatList]
        elif len(fxa.sensfeatList) > 0 and type(fxa.sensfeatList[0]) is str:
            sens_f_indexes = [self.input["f_names"].index(w) for w in fxa.sensfeatList]
            self.input["sens_f_names"] = fxa.sensfeatList #[w for _,_,w in fxa.sensfeatList
        else:
            sens_f_indexes = []
            self.input["sens_f_names"] = []

        #sens_f_unprivPops = [v for _,v,_ in fxa.sensfeatList]
        #sens_f_unprivPops_discretes = []
        #self.input["sens_f_names"] = [w for _,w in fxa.sensfeatList] #[w for _,_,w in fxa.sensfeatList]

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
        
        '''
        ######
        # for each column
        for i in range(len(self.input["sens_f_index"])):
            
            sens_f_index = self.input["sens_f_index"][i]

            if sens_f_index in self.input["nonnumeric_features"]: 

                le = encoders[sens_f_index]
                #sens_f_unprivPops_discreted = int(le.transform([sens_f_unprivPops[i]])[0])
                
                #new_array = [1 if x == str(float(sens_f_unprivPops_discreted)) else 0 for x in transformed_data[:,sens_f_index]]
                #transformed_data[:,sens_f_index] = np.array(new_array)
            
            else:
                sens_f_unprivPops_discreted = int(sens_f_unprivPops[i])
                    
            #sens_f_unprivPops_discretes.append(sens_f_unprivPops_discreted)
        '''
        
        self.sensitivefeatureslist = []
        
        # for each sensitive feature
        for i in range(len(self.input["sens_f_index"])):

            aSensitiveFeature = SensitiveFeature()
            aSensitiveFeature.featureIndex = self.input["sens_f_index"][i] 
            #aSensitiveFeature.unprivPop = None #sens_f_unprivPops_discretes[i]
            #aSensitiveFeature.unprivPop_original = None #sens_f_unprivPops[i]
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


        self._assess_fairness()


    def _assess_fairness(self):

        self.output["metrics_list"] = [FairMetricEnum.DP, FairMetricEnum.EO, FairMetricEnum.PE, FairMetricEnum.EOD]
        self.output["nonStandardMetricsToBeCalculated"] = [FairMetricEnum.PP, FairMetricEnum.CEA]

        self.output["result"] = self._eval_fairness(self.output["metrics_list"],
                                                   self.sensitivefeatureslist,
                                                   self.input["X"].tolist(),
                                                   self.input["y"].tolist(),
                                                   self.output["y_pred"],
                                                   "original")
        
        self.output["nonstandardResults"] = self._eval_fairness(self.output["nonStandardMetricsToBeCalculated"],
                                                               self.sensitivefeatureslist,
                                                               self.input["X"].tolist(),
                                                               self.input["y"].tolist(),
                                                               self.output["y_pred"],
                                                               "original")
        self._baselines()

    def _eval_fairness(self,metrics,sensFeatures,X,y,y_pred,txtIndicator):
        
        results = []
        for sensitiveFeature in sensFeatures:
            r = fairness.computeFairnessMetrics(metrics,
                                       sensitiveFeature, 
                                       X, 
                                       y,
                                       y_pred)
            results.append((sensitiveFeature,r,txtIndicator))
        
        return results
    

    def _run(self, fxa, webserver):
        self._common(fxa)
        if webserver :
            pickle.dump((self.input, self.output),open(str("repport_output.fixout"),"wb"))
                

    def run(self, fxa, show=True):
        """
        Runs FixOut with a given artifact.

        Parameters:
        -----------
        fxa : FixOutArtifact
            Original model, training and/or testing data to process.
        show : bool, optional
            If True (default), the output will be shown using a web interface, 
            otherwise only the evaluation will be executed and the results returned.

        Returns:
        --------
        tuple or None
            If `show` is True, returns None and actives a web interface.
            Otherwise, returns the computed evaluation results.

        """
        self._run(fxa, show)
        if show :
            print("Initializing the web interface.\nResults available at http://localhost:5000")
            app = webinterface.create_app()
            app.run()
        return self.output
    
    def runJ(self, fxa, show=True):
        """
        Runs FixOut with a given artifact in a Jupyter environment.

        Parameters:
        -----------
        fxa : FixOutArtifact
            Original model, training and/or testing data to process.
        show : bool, optional
            If True (default), the output will be shown using a web interface, 
            otherwise None is returned.

        Returns:
        --------
        IFrame
            Displays a frame with all the results obtained using FixOut.
        """
        self._run(fxa, True)
        app = webinterface.create_app()
        self.wserver = threading.Thread(target=app.run,args=())
        self.wserver.start()
        if show:
            return IFrame(httpaddress, 1200,800)
        else:
            return

    
    #def close(self):
    #    self.wserver.join()
    #    return 
    
    def _buildURL(self,service_name,dataset_reference):
        address = httpaddress + "d/" + service_name + "/"
        
        if dataset_reference is not None: # checking testing data
            if type(dataset_reference) is int:
                return address + str(dataset_reference+1)
            else:
                # searching for index based on label dataset
                for i in range(len(self.input["testing_data"])):
                    _,_,label = self.input["testing_data"][i] # X, y, label
                    if label == dataset_reference:
                        return address + str(i+1)
                return None # not found and is not training data
        else: # training data
            return address + str(0)
    
    def show(self):
        return IFrame(httpaddress, 1200,800)

    def data_distribution(self, dataset_reference=None):
        """
        Shows histograms to analyze the distribution of data centered on sensitive features.

        Returns:
        --------
        IFrame
            A frame with the requested results 
        """
        address = self._buildURL("histo",dataset_reference)
        if address is None:
            return 
        return IFrame(address, 1200,400)
    
    def data_visualization(self, dataset_reference=None):
        """
        Shows histograms to analyze the distribution of data centered on sensitive features.

        Returns:
        --------
        IFrame
            A frame with the requested results 
        """
        address = self._buildURL("visu",dataset_reference)
        if address is None:
            return 
        return IFrame(address, 1200,500)
    
    def correlation(self, dataset_reference=None):
        """
        Shows correlation matrices and rankings of features ordered according to a correlation coefficient. 
        The rankings are generated centered on sensitive features.

        Returns:
        --------
        IFrame
            A frame with the requested results 
        """
        address = self._buildURL("corr",dataset_reference)
        if address is None:
            return 
        return IFrame(address, 1200,400)
    
    def reverse(self, dataset_reference=None):
        """
        Shows the performance metrics for reversed models. 
        These models attempt to predict sensitive features from target labels.

        Returns:
        --------
        IFrame
            A frame with the requested results 
        """
        address = self._buildURL("rev",dataset_reference)
        if address is None:
            return 
        return IFrame(address, 1200,400)
    
    def discriminatory(self, dataset_reference=None):
        address = self._buildURL("corr",dataset_reference)
        if address is None:
            return 
        return IFrame(address, 1200,800)
    
    def fairness(self,dataset_reference=None):
        """
        Shows the calculated fairness metrics for the provided model. 

        Returns:
        --------
        IFrame
            A frame with the requested results 
        """
        address = self._buildURL("metrics",dataset_reference)
        return IFrame("http://localhost:5000/d/metrics/1", 1200,800)


    def get_fairness(self,
                     model="original",
                     sensitivefeature=None):
        """
        Retrieves calculated fairness metrics for a given model and sensitive feature.

        Parameters:
        -----------
        model : str, optional
            The model label to filter results (default: "original").
        sensitivefeature : str, optional
            The specific sensitive feature to filter results (default: None).

        Returns:
        --------
        dict
            A dictionary where keys are model labels and values are the calculated fairness metrics.
        """
        
        result = {}

        for sensf, calculated_metrics, model_label in self.output["result"]:#, self.output["nonstandardResults"]]:
            if sensitivefeature is not None and sensitivefeature == sensf.name:
                result[model_label] = calculated_metrics
            elif sensitivefeature is None:
                result[model_label] = calculated_metrics

        return result        

    def _baselines(self):

        predictions_list=[]

        for clazz,clazz_name in clazzes:
            self.__build_model(clazz,clazz_name,predictions_list)

        for name_method, preditions in predictions_list:
            for sensitiveFeature in self.sensitivefeatureslist:
                r = fairness.computeFairnessMetrics(self.output["metrics_list"],
                                        sensitiveFeature, 
                                        self.input["X"].tolist(), 
                                        self.input["y"].tolist(),
                                        preditions)
                self.output["result"].append((sensitiveFeature,r,name_method))

                nonStandardR = fairness.computeFairnessMetrics(self.output["nonStandardMetricsToBeCalculated"],
                                       sensitiveFeature, 
                                       self.input["X"].tolist(), 
                                       self.input["y"].tolist(),
                                       preditions)
                self.output["nonstandardResults"].append((sensitiveFeature,nonStandardR,name_method))

    def __build_model(self, clazz, name_method, predictions_list):

        clf = clazz()
        clf.fit(self.input["X"], self.input["y"])
        y_pred = clf.predict(self.input["X"])
        predictions_list.append((name_method, y_pred))

