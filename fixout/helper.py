from sklearn.exceptions import UndefinedMetricWarning

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

import warnings
warnings.filterwarnings('ignore')


clazzes = [(linear_model.RidgeClassifier,"linear reg"),
            (tree.DecisionTreeClassifier,"tree"),
            (svm.SVC,"svm"),
            (GaussianNB,"gaussian"),
            (RandomForestClassifier,"rd forest"),
            (MLPClassifier,"neural"),
            (GradientBoostingClassifier,"grad boost")]

class ReverseFairness():
    """
    A class for building and evaluating reversed fairness models. 
    These models attempt to predict sensitive features from target labels.
    """

    def build_reversed_models(self, X, y, sensitivefeatureslist, X_test=None, y_test=None):
        """
        Constructs reversed models to predict sensitive features from target labels.

        Parameters
        ----------
        X : array-like
            Feature matrix (excluding sensitive attributes).
        y : array-like
            Target variable labels.
        sensitivefeatureslist : list
            List of sensitive feature objects, each having a `name` and `featureIndex` attribute.
        X_test : array-like, optional
            Test set feature matrix (default is None).
        y_test : array-like, optional
            Test set target labels (default is None).

        Returns
        -------
        dict
            A dictionary mapping each sensitive feature name to a classifier and its list of calculated performance metrics.
        """
    
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

        return results


    def build_reverse_model(self, clazz, X, y, X_test=None, y_test=None):
        """
        Trains a classifier to predict sensitive attributes based on the target variable.

        Parameters
        ----------
        clazz : Classifier class
            The classifier to be trained (e.g., LogisticRegression, DecisionTree).
        X : array-like
            Training data (sensitive feature values).
        y : array-like
            Target labels.
        X_test : array-like, optional
            Test set sensitive feature values (default is None).
        y_test : array-like, optional
            Test set target labels (default is None).

        Returns
        -------
        tuple
            Calculated performance metrics (balanced accuracy, precision, and recall).
        
        Notes
        -----
        - The classifier is trained with `y` as input and `X` as output (reverse prediction).
        """
        clf = clazz()
        clf.fit(y, X)

        if X_test is not None and y_test is not None:
            return self.__eval_perf(clf, y_test, X_test)
        else:
            return self.__eval_perf(clf, y, X)

    def __eval_perf(self, clf, X_test, y_test):

        y_pred = clf.predict(X_test)
        
        try:
            baccuracy = balanced_accuracy_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
        except UndefinedMetricWarning:
            print("except UndefinedMetricWarning")

        return baccuracy, precision, recall

    

class UnfairModel():
    """
    A class for building and evaluating unfair models, which predict the target 
    variable using only sensitive attributes. This helps analyze bias in models.
    """

    def build_unfair_model(self, X, y, sensitivefeatureslist, X_test=None, y_test=None):
        """
        Trains a classifier to predict target variable based on sensitive attributes only.

        Parameters
        ----------
        X : array-like
            Training data (sensitive feature values).
        y : array-like
            Target labels.
        sensitivefeatureslist : list
            List of sensitive features
        X_test : array-like, optional
            Test set sensitive feature values (default is None).
        y_test : array-like, optional
            Test set target labels (default is None).

        Returns
        -------
        tuple
            Calculated performance metrics (balanced accuracy, precision, and recall).
        
        """

        results = []

        indexes = [x.featureIndex for x in sensitivefeatureslist]
        _X = X[:,indexes]
        
        for clazz,clazz_name in clazzes:
            
            clf = clazz()
            clf.fit(_X, y)

            if X_test is not None and y_test is not None:
                _X_test = X_test[:,indexes]
                results.append((clazz_name,self.__eval_perf(clf, _X_test, y_test)))
            else:
                results.append((clazz_name,self.__eval_perf(clf, _X, y)))

        return results

    def __eval_perf(self, clf, X_test, y_test):

        y_pred = clf.predict(X_test)
        
        try:
            baccuracy = balanced_accuracy_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
        except UndefinedMetricWarning:
            print("except UndefinedMetricWarning:")


        return baccuracy, precision, recall
