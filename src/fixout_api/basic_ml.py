from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

def mainClient():
    
    df = pd.read_csv('data/german.data',sep=" ", header=0)
    y = df['classification'].to_numpy()
    df = df.drop(['classification'],axis=1)
    dataset = df.to_numpy()

    enc = OrdinalEncoder()
    X_new = enc.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.75, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df.columns.values.tolist()

if __name__ == '__main__':
    mainClient()
    print("End")