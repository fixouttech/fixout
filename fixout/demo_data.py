from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

def importAdultData():
    
    df = pd.read_csv('data/adult.data',sep=", ", header=0)
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
    
    df = pd.read_csv('data/german.data',sep=" ", header=0)
    y = df['classification'].to_numpy()
    df = df.drop(['classification'],axis=1)
    dataset = df.to_numpy()

    enc = OrdinalEncoder()
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.75, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df.columns.values.tolist(), german_dictionary #generate_dictionary(enc,[19,18,8])


# checar o dicionario !! <<--- TODO 
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