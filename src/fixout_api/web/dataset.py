import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import plotly
import plotly.graph_objs as go

import json

from scipy.stats import spearmanr
from scipy.stats import pearsonr


def vis_tsne(X,label,sensitive_feature_index, sensitive_feature_names):

    if X is None or label is None:
        return None
    
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(X)
    #tsne_result.shape

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': label})

    graphJSON_list = []
    for i in range(len(sensitive_feature_index)):
        sens_f_index = [i]
        data = [
            go.Scatter(
                x=tsne_result_df['tsne_1'],
                y=tsne_result_df['tsne_2'],
                mode = 'markers',
                marker = dict(
                    color = tsne_result_df['label'],
                    symbol=pd.DataFrame(X)[sens_f_index]
                    ),
                name=""            
            )
        ]
        
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON_list.append((sensitive_feature_names[i],sens_f_index,graphJSON))

    return graphJSON_list

def vis_pca(X,label,sensitive_feature_index, sensitive_feature_names):
    
    if X is None or label is None:
        return None
    
    pca = PCA(n_components=2)    
    components = pca.fit_transform(X)

    pca_components_df = pd.DataFrame({'component_1': components[:,0], 
                                      'component_2': components[:,1], 
                                      'label': label})

    graphJSON_list = []
    for i in range(len(sensitive_feature_index)):
        sens_f_index = [i]
        data = [
            go.Scatter(
                x=pca_components_df['component_1'],
                y=pca_components_df['component_2'],
                mode = 'markers',
                marker = dict(
                    color = pca_components_df['label'],
                    symbol=pd.DataFrame(X)[sens_f_index]
                    ),
                name="",
                showlegend=True
            )
        ]
        
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON_list.append((sensitive_feature_names[i],sens_f_index,graphJSON))

    return graphJSON_list

def vis_lda(X,label,sensitive_feature_index, sensitive_feature_names):
    
    if X is None or label is None:
        return None

    lda = LinearDiscriminantAnalysis(n_components=2)    
    components = lda.fit_transform(X)

    lda_components_df = pd.DataFrame({'component_1': components[:,0], 'component_2': components[:,1], 'label': label})

    graphJSON_list = []
    for i in range(len(sensitive_feature_index)):
        sens_f_index = [i]
        data = [
            go.Scatter(
                x=lda_components_df['component_1'],
                y=lda_components_df['component_2'],
                mode = 'markers',
                marker = dict(
                    color = lda_components_df['label'],
                    symbol=pd.DataFrame(X)[sens_f_index]
                    ),
                name=""            
            )
        ]
        
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON_list.append((sensitive_feature_names[i],sens_f_index,graphJSON))

    return graphJSON_list

def correlation_analysis(X,
                         sensitive_feature_index, 
                         sensitive_feature_names, 
                         f_names,
                         func_corr=pearsonr):

    if X is None:
        return (None, None)

    n_features = len(X[0])
    corr_matrix = np.zeros((n_features,n_features))
    for i in range(n_features):
        corr_matrix[i] = [func_corr(X[:,i],X[:,j])[0] for j in range(n_features)]

    #corr_matrix = pearsonr(X,X)#,rowvar=False)
    #corr_matrix1 = spearmanr(X,X)

    data = [
        go.Heatmap(
            z=corr_matrix,
            x=f_names,
            y=f_names
        )
    ]
    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    #upper_matrix = np.triu(corr_matrix,k=1) # above the diagonal
    
    rankings = []

    for i in range(len(sensitive_feature_index)):
        index_sens_feature = sensitive_feature_index[i]
        name_sens_feature = sensitive_feature_names[i]
        ranking = corr_matrix[index_sens_feature]
        tuple = []
        for r in range(len(ranking)):
            if f_names[r] != name_sens_feature:
                if abs(ranking[r]) >= 0.1:
                    tuple.append((f_names[r],ranking[r]))

        tuple.sort(reverse=True,key=lambda x: abs(x[1]))
        rankings.append((name_sens_feature,tuple))


    return (graphJSON, rankings)


def statsNonThreshold(X,
                      y,
                      sensitive_feature_index, 
                      sensitive_feature_names, 
                      sens_f_type, 
                      sens_f_unpriv, 
                      sens_f_unpriv_original):

    if X is None or y is None:
        return None

    labels = pd.DataFrame(y)
    possible_labels = list(set([int(x) for x in y]))
    df = pd.DataFrame(np.array(X).astype(int))
    
    graphJSON_dic = []
    for i in range(len(sensitive_feature_names)): # for each sensitive feature
        sens_f_index = sensitive_feature_index[i]
                
        #print(df.iloc[:,sens_f_index].to_numpy().flatten())
        #print(labels)

        for target_label in possible_labels:
            target_indexes =  labels[0] == target_label
            target_indexes_comp =  labels[0] != target_label

            data = [
                go.Histogram(
                    #x=unprivPop,
                    x= df.iloc[:,sens_f_index][target_indexes],#.to_numpy().flatten(),
                    name="True",
                    showlegend=False,
                    opacity=0.6
                ),
                go.Histogram(
                    #x=unprivPop,
                    x= df.iloc[:,sens_f_index][target_indexes_comp],#.to_numpy().flatten(),
                    name="False",
                    showlegend=False,
                    opacity=0.6
                )
            ]
        
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        sens_f_name = sensitive_feature_names[i]
        graphJSON_dic.append((sens_f_name, graphJSON))

    return graphJSON_dic

def stats(X,
          y,
          sensitive_feature_index, 
          sensitive_feature_names, 
          sens_f_type, 
          sens_f_unpriv, 
          sens_f_unpriv_original):

    if X is None or y is None:
        return None

    labels = pd.DataFrame(y)
    df = pd.DataFrame(np.array(X).astype(int))
    
    graphJSON_dic = []
    for i in range(len(sensitive_feature_names)): # for each sensitive feature
        sens_f_index = sensitive_feature_index[i]
                
        print(df.iloc[:,sens_f_index])
        print(labels)
        if sens_f_type[i] == 1: # nonnumeric
            indexes_unpriv =  df.iloc[:,sens_f_index] == sens_f_unpriv[i]
            indexes_priv = df.iloc[:,sens_f_index] != sens_f_unpriv[i]
            legend_unpriv = "=" + str(sens_f_unpriv_original[i])
            legend_priv = "!=" + str(sens_f_unpriv_original[i])
        else: # numeric
            indexes_unpriv = df.iloc[:,sens_f_index] < sens_f_unpriv[i]
            indexes_priv = df.iloc[:,sens_f_index] >= sens_f_unpriv[i]
            legend_unpriv = "<" + str(sens_f_unpriv_original[i])
            legend_priv = ">=" + str(sens_f_unpriv_original[i])

        unprivPop_labels = labels.loc[indexes_unpriv]
        privPop_labels = labels.loc[indexes_priv]

        data = [
            go.Histogram(
                #x=unprivPop,
                x=unprivPop_labels.to_numpy().flatten(),
                name="Unpriv. (" + legend_unpriv + ")",
                showlegend=False
            ),
            go.Histogram(
                #x=privPop,
                x=privPop_labels.to_numpy().flatten(),
                name="Priv. (" + legend_priv + ")",
                showlegend=False
            )
        ]
        
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        sens_f_name = sensitive_feature_names[i]
        graphJSON_dic.append((sens_f_name, graphJSON))

    return graphJSON_dic
