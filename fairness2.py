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
    value = ((tp0+tn0)/(tn0+fp0+fn0+tp0)) - ((tp1+tn1)/(tn1+fp1+fn1+tp1))
    return value

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
    value = (tp0/(tp0+fn0)) - (tp1/(tp1+fn1)) 
    return value

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
    value = (fp0/(fp0+tp0)) - (fp1/(fp1+tp1)) 
    return value
