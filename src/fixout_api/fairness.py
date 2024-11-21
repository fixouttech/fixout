from fixout.ttypes import FairMetric, FairMetricEnum

import numpy as np
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix


def equalized_odds(cm0,cm1):
    """
    Calculates the equalized odds.
    
    Parameters
    ----------
    cm0 : 
        The confusion matrix of the unprivileged subpopulation.
    
    cm1 : 
        The confusion matrix of the privileged subpopulation.

    Returns
    -------
    metric : FairMetric
        The calculated equalized odds metric.
    """
    tn0, fp0, fn0, tp0 = cm0 # unprivileged 
    tn1, fp1, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.EOD
    metric.value = ( (tp0/(tp0+fn0)) + (fp0/(fp0+tn0)) ) - ( (tp1/(tp1+fn1)) + (fp1/(fp1+tn1)) )
    return metric

def demographic_parity(cm0,cm1):
    """
    Calculates the demographic parity metric.
    
    Parameters
    ----------
    cm0 : 
        The confusion matrix of the unprivileged subpopulation.
    
    cm1 : 
        The confusion matrix of the privileged subpopulation.

    Returns
    -------
    metric : FairMetric
        The calculated demographic parity metric.
    """
    tn0, fp0, fn0, tp0 = cm0 # unprivileged 
    tn1, fp1, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.DP
    metric.value = ((tp0+tn0)/(tn0+fp0+fn0+tp0)) - ((tp1+tn1)/(tn1+fp1+fn1+tp1))
    return metric

def equal_opportunity(cm0,cm1):
    """
    Calculates the equal opportunity metric.
    
    Parameters
    ----------
    cm0 : 
        The confusion matrix of the unprivileged subpopulation.
    
    cm1 : 
        The confusion matrix of the privileged subpopulation.

    Returns
    -------
    metric : FairMetric
        The calculated equal opportunity metric.
    """
    _, _, fn0, tp0 = cm0 # unprivileged 
    _, _, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.EO
    metric.value = (tp0/(tp0+fn0)) - (tp1/(tp1+fn1)) 
    return metric

def predictive_equality(cm0,cm1):
    """
    Calculates the equal predictive equality metric.

    Parameters
    ----------
    cm0 : 
        The confusion matrix of the unprivileged subpopulation.
    
    cm1 : 
        The confusion matrix of the privileged subpopulation.

    Returns
    -------
    metric : FairMetric
        The calculated predictive equality metric.
    """
    _, fp0, _, tp0 = cm0 # unprivileged 
    _, fp1, _, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.PE
    metric.value = (fp0/(fp0+tp0)) - (fp1/(fp1+tp1)) 
    return metric

    
def computeFairnessMetrics(isOriginalModel, metrics, sFeature, X_test, y_test, y_pred):
    """
    Calculate fariness metrics for a given ensemble of instances. 
    
    Supported fariness metrics: Demographic Parity, Equal Opportunity, Predictive Equality.

    Check out the enum types : 'FairMetricEnum'
    
    Parameters
    ----------
    metrics : list
        List of metrics to be calculated.

    sFeature : SensitiveFeature 
        Sensitive feature that will be take into account to calculate fairness metrics.

    X_test : array-like of shape 
        Dataset

    y_test : array-like of shape 
        Predicted outcomes related to X_test.

    Returns
    -------
    results : list of FairMetric
        List with all requested metrics calculated.
        Returns an empty list if no metric is informed or a problem when calculating the confusion matrix is found.

    """ 
    
    y_test_array = np.array(y_test)
    X_test_array = np.array(X_test)

    results = {}
    for value in set(X_test_array[:, sFeature.featureIndex]): # default = sFeature.unprivPop

        results[str(value)] = []
        print(value)

        if sFeature.type == 0: # numeric sensitive feature
            sens_array = X_test_array[:, sFeature.featureIndex] 
            sens_array = binarize([sens_array.astype(float)],threshold=value)
        
        indexes0 = np.where(X_test_array[:, sFeature.featureIndex] == value)
        indexes1 = np.where(X_test_array[:, sFeature.featureIndex] != value) #sFeature.unprivPop
        
        y_test_0 = y_test_array[indexes0]
        y_test_1 = y_test_array[indexes1]
        y_pred_0 = y_pred[indexes0]
        y_pred_1 = y_pred[indexes1]

        # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cm0 = confusion_matrix(y_test_0, y_pred_0).ravel()
        cm1 = confusion_matrix(y_test_1, y_pred_1).ravel()

        if (len(cm0) == 4) and (len(cm1) == 4) :
            
            for i in metrics:
                if i == FairMetricEnum.DP:  
                    results[str(value)].append(demographic_parity(cm0,cm1))
                elif i == FairMetricEnum.EO:
                    results[str(value)].append(equal_opportunity(cm0,cm1))
                elif i == FairMetricEnum.PE:
                    results[str(value)].append(predictive_equality(cm0,cm1))
                elif i == FairMetricEnum.EOD:
                    results[str(value)].append(equalized_odds(cm0,cm1))
        
    return results
