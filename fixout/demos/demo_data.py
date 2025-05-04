from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from importlib.resources import files

def importLSACData():
    csv_path = files('fixout').joinpath('demos/data/lsac.data')
    df = pd.read_csv(csv_path,sep=";", header=0)
    y = df['pass_bar'].to_numpy()
    encY = LabelEncoder()
    y = encY.fit_transform(y)
    _df = df.drop(['pass_bar'],axis=1)
    dataset = _df.to_numpy()

    enc = OrdinalEncoder(encoded_missing_value=-1)
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.5, random_state=42)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    sen_features_indexes = [_df.columns.get_loc(x) for x in ["sex","race","family_income"]]

    return model, X_train, X_test, y_train, y_test, _df.columns.values.tolist(), generate_dictionary(enc,sen_features_indexes)

def importBankData():
    
    #csv_path = files('fixout').joinpath('demos/data/bank.data')
    df = pd.read_csv('fixout/demos/data/bank.data',sep=";", header=0)
    y = df['y'].to_numpy()
    encY = LabelEncoder()
    y = encY.fit_transform(y)
    _df = df.drop(['y'],axis=1)
    dataset = _df.to_numpy()


    enc = OrdinalEncoder(encoded_missing_value=-1)
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.5, random_state=42)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    sen_features_indexes = [_df.columns.get_loc(x) for x in ["age","marital","contact"]]

    return model, X_train, X_test, y_train, y_test, _df.columns.values.tolist(), generate_dictionary(enc,sen_features_indexes)


def importCompasData():
    
    #df = pd.read_csv('fixout/demos/data/compas.data',sep=",", header=0)
    csv_path = files('fixout').joinpath('demos/data/compas.data')
    df = pd.read_csv(csv_path,sep=",", header=0)
    y = df['two_year_recid'].to_numpy()
    _df = df.drop(['two_year_recid','id','name','c_case_number','type_of_assessment','v_type_of_assessment','violent_recid'],axis=1)#,'r_charge_desc','vr_charge_degree','decile_score.1','end','event'],axis=1)
    dataset = _df.to_numpy()

    enc = OrdinalEncoder(encoded_missing_value=-1)
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    sen_features_indexes = [_df.columns.get_loc(x) for x in ["sex","age","race"]]

    return model, X_train, X_test, y_train, y_test, _df.columns.values.tolist(), generate_dictionary(enc,sen_features_indexes)

def importAdultData():
    
    csv_path = files('fixout').joinpath('demos/data/adult.data')
    df = pd.read_csv(csv_path,sep=", ", header=0)
    dataset = df.to_numpy()

    enc = OrdinalEncoder()
    _dataset = enc.fit_transform(dataset)

    X_new = _dataset[:,0:13] # without y
    y = _dataset[:,14]

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.50, random_state=42)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df.columns.values.tolist(), generate_dictionary(enc,[5,8,9])

def generate_dictionary(encoder, sens_feature_indexes):
    dictionary = {}
    for i in sens_feature_indexes:
        dictionary[i] = {}
        for j in range(len(encoder.categories_[i])):
            dictionary[i][j] = encoder.categories_[i][j]
    return dictionary


def importGermanData():

    csv_path = files('fixout').joinpath('demos/data/german.data')
    
    df = pd.read_csv(csv_path,sep=" ", header=0)
    y = df['classification'].to_numpy()
    df = df.drop(['classification'],axis=1)
    dataset = df.to_numpy()

    enc = OrdinalEncoder()
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.75, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df.columns.values.tolist(), german_dictionary #generate_dictionary(enc,[19,18,8])


german_dictionary = {
    19 : { 
        0 : "yes",
        1 : "no"
    },
    18 : {
        0 : "none",
        1 : "yes", # registered under the customers name  
    },
    8 : {
        0 : "male divorced",
        1 : "female divorced",
        2 : "male single",
        3 : "male married",
        4 : "female single"
    }
}

if __name__ == '__main__':
    importGermanData()