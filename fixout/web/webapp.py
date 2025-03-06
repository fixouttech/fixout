from flask import Flask, redirect, render_template, url_for
from flask import request
from flask import jsonify

import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import pickle

import web.dataset as dat
import web.fair_exp as fair_exp

from scipy.stats import spearmanr
from scipy.stats import pearsonr

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

storage_path = "storage/"

html_starter = 'index.html'

class WebData():
    def __init__(self):
        f = open("repport_output.fixout",'rb')
        self.input, self.output = pickle.load(f)

    def data(self):
        return self.input["X"], self.input["y"]
    
    def testing_data(self):
        return self.input["testing_data"]
    
    def meta_data(self):
        f_names = self.input["f_names"] 
        sens_f_index = self.input["sens_f_index"]
        sens_f_names = self.input["sens_f_names"]
        dictionary = self.input["dictionary"]

        return dictionary, f_names, sens_f_index, sens_f_names

    def fairness_metrics(self):
        metrics_list = self.output["metrics_list"]
        other_metrics_list = self.output['nonStandardMetricsToBeCalculated']

        return metrics_list, other_metrics_list


@app.route('/')
def index():

    wdata = WebData()

    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()
    metrics_list, other_metrics_list = wdata.fairness_metrics()

    X, y = wdata.data()
    dataslices = wdata.testing_data()
    

    return render_template(html_starter, 
                           report_details=wdata.input["report_details"],
                           num_data_slices=len(dataslices), 
                           sensitive_features=wdata.input["sens_f_names"],
                           dataslices=dataslices)

@app.route('/d/metrics/<int:data>')
def d_metrics(data_slice_index):
    return redirect(url_for('metrics', data=data_slice_index, debug=True))

@app.route('/metrics/<int:data>')
def metrics(data):
    wdata = WebData()
    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()
    if data == 0:
        X, y = wdata.data()
    else:
        X, y, _ = wdata.testing_data()[data-1]

    full_fairness_plots = fair_exp.generate_fairness_plots(wdata.output["result"],dictionary=dictionary)
    other_fairness_plots = fair_exp.generate_fairness_plots(wdata.output["nonstandardResults"],dictionary=dictionary)

    return render_template("metrics.html",
                           debug=request.args.get('debug', None),
                           sensitive_features=wdata.input["sens_f_names"],
                           full_fairness_plots=full_fairness_plots,
                           other_fairness_plots=other_fairness_plots) # train 


@app.route('/d/visu/<int:data_slice_index>')
def d_visu(data_slice_index):
    return redirect(url_for('visu', data_slice_index=data_slice_index, debug=True))

@app.route('/visu/<int:data_slice_index>')
def visu(data_slice_index):
    wdata = WebData()
    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()
    if data_slice_index == 0:
        X, y = wdata.data()
    else:
        X, y, _ = wdata.testing_data()[data_slice_index-1]

    plots_visu = dat.generate_visu(X, y, sens_f_index, sens_f_names)

    return render_template("visualization.html",
                           debug=request.args.get('debug', None),
                           index_data=data_slice_index,
                           sensitive_features=wdata.input["sens_f_names"],
                           plots_visu=plots_visu) # train 

@app.route('/d/histo/<int:data_slice_index>')
def d_histo(data_slice_index):
    return redirect(url_for('histo', data_slice_index=data_slice_index, debug=True))

@app.route('/histo/<int:data_slice_index>')
def histo(data_slice_index):
    wdata = WebData()
    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()
    if data_slice_index == 0: # indicates training data
        X, y = wdata.data()
    else: # otherwise, testing data is used
        X, y, _ = wdata.testing_data()[data_slice_index-1]

    histograms = dat.statsNonThreshold(X, y, sens_f_index, sens_f_names, dictionary)
    inter_histograms = dat.statsInter(X, y, sens_f_index, sens_f_names, dictionary)

    return render_template("histograms.html",
                           debug=request.args.get('debug', None),
                           index_data=data_slice_index,
                           histograms=histograms,
                           inter_histograms=inter_histograms) 



@app.route('/d/corr/<int:data_slice_index>')
def d_corr(data_slice_index):
    return redirect(url_for('corr', data_slice_index=data_slice_index, debug=True))

@app.route('/corr/<int:data_slice_index>')
def corr(data_slice_index):
    wdata = WebData()
    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()
    if data_slice_index == 0:
        X, _ = wdata.data()
    else:
        X, _, _ = wdata.testing_data()[data_slice_index-1]

    pearCorr_heatmap, pearCorr_rankings = dat.correlation_analysis(X, sens_f_index, sens_f_names, f_names, func_corr=pearsonr)
    speaCorr_heatmap, speaCorr_rankings = dat.correlation_analysis(X, sens_f_index, sens_f_names, f_names, func_corr=spearmanr)

    return render_template("correlation.html",
                           debug=request.args.get('debug', None),
                           sensitive_features=wdata.input["sens_f_names"],
                           index_data=data_slice_index,
                           pearCorr_heatmap=pearCorr_heatmap, 
                           pearCorr_rankings=pearCorr_rankings, 
                           speaCorr_heatmap=speaCorr_heatmap,
                           speaCorr_rankings=speaCorr_rankings) 


@app.route('/d/rev/<int:data_slice_index>')
def d_rev(data_slice_index):
    return redirect(url_for('rev', data_slice_index=data_slice_index, debug=True))

@app.route('/rev/<int:data_slice_index>')
def rev(data_slice_index):
    wdata = WebData()
    dictionary, f_names, sens_f_index, sens_f_names = wdata.meta_data()

    rmodels_perf_plors = dat.reversed_models_plot(perf_metrics=wdata.input["reversed_models"][data_slice_index], dictionary=dictionary)
    unfairmodel_perf_plots = dat.unfair_model_plot(perf_metrics=wdata.input["unfair_model"][data_slice_index], dictionary=dictionary)

    return render_template("reversed.html",
                           debug=request.args.get('debug', None),
                           sensitive_features=wdata.input["sens_f_names"],
                           index_data=data_slice_index,
                           rmodels_perf_plors = rmodels_perf_plors,
                           unfairmodel_perf_plots = unfairmodel_perf_plots) 

if __name__ == "__main__":
    app.run(debug=True)