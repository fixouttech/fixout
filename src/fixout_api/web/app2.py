from flask import Flask, render_template
from flask import jsonify
from werkzeug.exceptions import abort

import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import pickle

import web.dataset as dat
import web.comparison as comp
import web.fair_exp as fair_exp

from scipy.stats import spearmanr
from scipy.stats import pearsonr

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

storage_path = "storage/"

html_starter = 'iopen.html'

@app.route('/')
def index():
    #return render_template('index2.html', posts=None)

#@app.route('/<string:report_id>')
#def get_post(report_id):
        
    f = open("repport_output.fixout",'rb')
    data = pickle.load(f)

    model_availability = data["model_availability"]
    y_prob_predictions = data["prob_y_pred"]  
    y_predictions = data["y_pred"]
    
    f_names = data["f_names"] 
    sens_f_index = data["sens_f_index"]
    sens_f_names = data["sens_f_names"]
    metrics_list = data["metrics_list"]
    if "sens_f_unpriv" in data.keys():
        sens_f_unpriv = data["sens_f_unpriv"]
        sens_f_unpriv_original = data["sens_f_unpriv_original"]
        sens_f_type = data["sens_f_type"]
        print("sens_f_unpriv",sens_f_unpriv)
        print("sens_f_unpriv_original",sens_f_unpriv_original)
        print("sens_f_type",sens_f_type)
    
#    if X_train is None:
#        X_train = X_test
#    if y_train is None:
#        y_train = y_test

    X_train = data["X_train"]
    y_train = data["y_train"]
    
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Data
    train_histograms = dat.stats(X_train, y_train, sens_f_index, sens_f_names, sens_f_type, sens_f_unpriv, sens_f_unpriv_original)
    train_corr_heatmap, train_corr_rankings = dat.correlation_analysis(X_train, sens_f_index, sens_f_names, f_names)
    train_visu = dat.vis_pca(X_train, y_train, sens_f_index, sens_f_names)

    test_histograms = dat.statsNonThreshold(X_test, y_test, sens_f_index, sens_f_names, sens_f_type, sens_f_unpriv, sens_f_unpriv_original)
    #test_histograms = dat.stats(X_test, y_test, sens_f_index, sens_f_names, sens_f_type, sens_f_unpriv, sens_f_unpriv_original)
    test_PearCorr_heatmap, test_PearCorr_rankings = dat.correlation_analysis(X_test, sens_f_index, sens_f_names, f_names, func_corr=pearsonr)
    test_SpeaCorr_heatmap, test_SpeaCorr_rankings = dat.correlation_analysis(X_test, sens_f_index, sens_f_names, f_names, func_corr=spearmanr)
    test_visu = dat.vis_pca(X_test, y_test, sens_f_index, sens_f_names)
    
    # Explanations
    #plots = fair_exp.create_plot(cursor, report_id, sensitive_features)
    full_fairness_plots = fair_exp.full_fairness_metrics2(data["fair_metrics"],metrics_list)

    if train_histograms is not None and test_histograms is not None:
        return render_template(html_starter, 
                           report_details=data["report_details"], 
                           sensitive_features=data["sens_f_names"],
                           train_visu=train_visu, # train
                           train_corr_heatmap=train_corr_heatmap, 
                           train_corr_rankings=train_corr_rankings, 
                           train_histograms=train_histograms, 
                           test_visu=test_visu, # test
                           test_PearCorr_heatmap=test_PearCorr_heatmap, 
                           test_PearCorr_rankings=test_PearCorr_rankings, 
                           test_SpeaCorr_heatmap=test_SpeaCorr_heatmap,
                           test_SpeaCorr_rankings=test_SpeaCorr_rankings,
                           test_histograms=test_histograms,  
                           full_fairness_plots=full_fairness_plots)
    
    elif train_histograms is not None:
        return render_template(html_starter, 
                           report_details=data["report_details"], 
                           sensitive_features=data["sens_f_names"],
                           train_visu=train_visu, # train
                           train_corr_heatmap=train_corr_heatmap, 
                           train_corr_rankings=train_corr_rankings, 
                           train_histograms=train_histograms,  
                           full_fairness_plots=full_fairness_plots)
    
    elif test_histograms is not None:
        return render_template(html_starter, 
                           report_details=data["report_details"], 
                           sensitive_features=data["sens_f_names"],
                           test_visu=test_visu, # test
                           test_PearCorr_heatmap=test_PearCorr_heatmap, 
                           test_PearCorr_rankings=test_PearCorr_rankings, 
                           test_SpeaCorr_heatmap=test_SpeaCorr_heatmap,
                           test_SpeaCorr_rankings=test_SpeaCorr_rankings,
                           test_histograms=test_histograms,  
                           full_fairness_plots=full_fairness_plots)
    else:
        return render_template(html_starter, 
                           report_details=data["report_details"], 
                           sensitive_features=data["sens_f_names"])

if __name__ == "__main__":
    app.run(debug=True)