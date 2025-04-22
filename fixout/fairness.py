"""This module provides a suite of fairness metrics for evaluating machine learning models. 
It includes measures like Conditional Accuracy Equality, Predictive Parity, Equal Opportunity, and more. 
Each metric is implemented using a generic evaluation wrapper and is based on protected attributes or sensitive features to measure fairness across demographic groups."""

from fixout.interface.ttypes import FairMetric, FairMetricEnum

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix

def conditional_accuracy_equality(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Computes the Conditional Accuracy Equality (CEA) fairness metric.

    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    FairMetric
        The calculated conditional accuracy equality metric.

    """
    return __genericFairnessMetric(_conditional_accuracy_equality, sFeature, X_test, y_test, model, y_pred)

def predictive_parity(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Computes the Predictive Parity (PP) fairness metric.

    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    FairMetric
        The calculated predictive parity metric.
    """
    return __genericFairnessMetric(_predictive_parity, sFeature, X_test, y_test, model, y_pred)

def predictive_equality(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Calculates the equal predictive equality metric.

    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    metric : FairMetric
        The calculated predictive equality metric.
    """
    return __genericFairnessMetric(_predictive_equality, sFeature, X_test, y_test, model, y_pred)

def equal_opportunity(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Calculates the equal opportunity (EO) fairness metric.
    
    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    metric : FairMetric
        The calculated equal opportunity metric.

    """
    return __genericFairnessMetric(_equal_opportunity, sFeature, X_test, y_test, model, y_pred)

def demographic_parity(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Calculates the demographic parity (DP) fairness metric.
    
    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    metric : FairMetric
        The calculated demographic parity metric.

    """
    return __genericFairnessMetric(_demographic_parity, sFeature, X_test, y_test, model, y_pred)

def equalized_odds(sFeature, X_test, y_test, model=None, y_pred=None):
    """
    Calculates the equalized odds (EOD) fairness metric.
    
    Parameters
    ----------
    sFeature : SensitiveFeature
        Sensitive feature that will be taken into account to calculate fairness metrics.
    X_test : array-like
        Feature set for testing the model.
    y_test : array-like
        True labels for the test data.
    model : sklearn-like estimator, optional
        Trained model to generate predictions. If `y_pred` is not provided, this model will be used.
    y_pred : array-like, optional
        Predicted labels for `X_test`. If provided, `model` is ignored.

    Returns
    -------
    metric : FairMetric
        The calculated equalized odds metric.
    
    """
    return __genericFairnessMetric(_equalized_odds, sFeature, X_test, y_test, model, y_pred)

def __genericFairnessMetric(fairMetricFunc, sFeature, X_test, y_test, model=None, y_pred=None):

    # check model and y_pred
    _X_test, _y_test, _y_pred = __check(X_test, y_test, model, y_pred)
    possible_values = __getPossibleValues(_X_test, sFeature)

    results = {}
    results[str(value)] = []
    
    for value in possible_values: 
        
        y_test_0, y_test_1, y_pred_0, y_pred_1 = __dividePop(sFeature, value, _X_test, _y_pred, _y_test)
        cm0 = confusion_matrix(y_test_0, y_pred_0).ravel()
        cm1 = confusion_matrix(y_test_1, y_pred_1).ravel()

        if (len(cm0) == 4) and (len(cm1) == 4) :
            calculated_metrics = fairMetricFunc(cm0,cm1)
            results[str(value)].append(calculated_metrics)

    return results

def __check(X_test, y_test, model=None, y_pred=None):
    # check model and y_pred
    if model == None and y_pred == None:
        return None
    
    if y_pred == None:
        _y_pred = model.predict(X_test)
    else:
        _y_pred = y_pred  

    return np.array(X_test), np.array(y_test), _y_pred

def __getPossibleValues(sFeature, X):
    possible_values = list(set(X[:, sFeature.featureIndex])) # default = sFeature.unprivPop
    if len(possible_values) == 2:
        possible_values = possible_values[:-1]
    return possible_values

def __dividePop(sFeature, value, X, y_pred, y_test):
    
    if sFeature.type == 0: # numeric sensitive feature
        sens_array = X[:, sFeature.featureIndex] 
        sens_array = binarize([sens_array.astype(float)],threshold=value)
    
    indexes0 = np.where(X[:, sFeature.featureIndex] == value)
    indexes1 = np.where(X[:, sFeature.featureIndex] != value) #sFeature.unprivPop

    return y_test[indexes0], y_test[indexes1], y_pred[indexes0], y_pred[indexes1]

def _equalized_odds(cm0,cm1):
    tn0, fp0, fn0, tp0 = cm0 # unprivileged 
    tn1, fp1, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.EOD
    metric.value = ( (tp0/(tp0+fn0)) + (fp0/(fp0+tn0)) ) - ( (tp1/(tp1+fn1)) + (fp1/(fp1+tn1)) )
    return metric

def _demographic_parity(cm0,cm1):
    tn0, fp0, fn0, tp0 = cm0 # unprivileged 
    tn1, fp1, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.DP
    metric.value = ((tp0+tn0)/(tn0+fp0+fn0+tp0)) - ((tp1+tn1)/(tn1+fp1+fn1+tp1))
    return metric

def _equal_opportunity(cm0,cm1):
    _, _, fn0, tp0 = cm0 # unprivileged 
    _, _, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.EO
    metric.value = (tp0/(tp0+fn0)) - (tp1/(tp1+fn1)) 
    return metric

def _predictive_equality(cm0,cm1): 
    _, fp0, _, tp0 = cm0 # unprivileged 
    _, fp1, _, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.PE
    metric.value = (fp0/(fp0+tp0)) - (fp1/(fp1+tp1)) 
    return metric

def _predictive_parity(cm0,cm1):
    _, fp0, _, tp0 = cm0 # unprivileged 
    _, fp1, _, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.PP
    metric.value = (tp0/(tp0+fp0)) - (tp1/(tp1+fp1))
    return metric

def _conditional_accuracy_equality(cm0,cm1):
    tn0, fp0, fn0, tp0 = cm0 # unprivileged 
    tn1, fp1, fn1, tp1 = cm1 # privileged
    metric = FairMetric()
    metric.type = FairMetricEnum.CEA
    metric.value = ((tp0/(tp0+fp0))+(tn0/(tn0+fn0))) - ((tp1/(tp1+fp1))+(tn1/(tn1+fn1)))
    return metric
    
def computeFairnessMetrics(metrics, sFeature, X_test, y_test, y_pred):
    """
    Calculate fariness metrics for a given ensemble of instances. 
    
    Supported fariness metrics: Demographic Parity, Equal Opportunity, Predictive Equality.

    Check out the enum types : 'FairMetricEnum'
    
    Parameters
    ----------
    metrics : list
        List of fairness metric types to be calculated.

    sFeature : SensitiveFeature 
        Sensitive feature that will be taken into account to calculate fairness metrics.

    X_test : array-like
        Feature matrix of the test set.
    
    y_test : array-like
        True labels of the test set.
    
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    results : list of FairMetric
        List with all requested metrics calculated.
        Returns an empty list if no metric is informed or a problem when calculating the confusion matrix is found.

    """ 
    
    y_test_array = np.array(y_test)
    X_test_array = np.array(X_test)

    results = {}
    
    possible_values = list(set(X_test_array[:, sFeature.featureIndex])) # default = sFeature.unprivPop
    if len(possible_values) == 2:
        possible_values = possible_values[:-1]

    for value in possible_values: 

        results[str(value)] = []

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
                    results[str(value)].append(_demographic_parity(cm0,cm1))
                elif i == FairMetricEnum.EO:
                    results[str(value)].append(_equal_opportunity(cm0,cm1))
                elif i == FairMetricEnum.PE:
                    results[str(value)].append(_predictive_equality(cm0,cm1))
                elif i == FairMetricEnum.EOD:
                    results[str(value)].append(_equalized_odds(cm0,cm1))
                elif i == FairMetricEnum.PP:
                    results[str(value)].append(_predictive_parity(cm0,cm1))
                elif i == FairMetricEnum.CEA:
                    results[str(value)].append(_conditional_accuracy_equality(cm0,cm1))
        
    return results
