from flask import Flask, render_template
from flask import jsonify
from werkzeug.exceptions import abort
from flask_mysqldb import MySQL

import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import pickle

import dataset as dat
import comparison as comp
import fair_exp

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'fixout'

app.config["TEMPLATES_AUTO_RELOAD"] = True
mysql = MySQL(app)

storage_path = "storage/"

@app.route('/')
def index():
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM explanations WHERE token_fk = %s',("000000",))
    posts = cursor.fetchall()
    cursor.close()
    return render_template('index2.html', posts=posts)

@app.route('/<string:report_id>')
def get_post(report_id):
    cursor = mysql.connection.cursor()
    
    cursor.execute('SELECT * FROM reports INNER JOIN tokens ON tokens.token = reports.token_fk WHERE  report_id = %s',(report_id,))
    report_details = cursor.fetchone()
    
    f = open(storage_path+report_details[0]+"_"+report_details[2]+"_Xy_train",'rb')
    data = pickle.load(f)

    if isinstance(data,tuple): 
        X_train, y_train, f_names, sens_f_index, sens_f_names = data
        model_availability = True
    elif isinstance(data,dict):
        model_availability = data["model_availability"]
        X_train = data["X_train"]
        y_train = data["y_train"]
        f_names = data["f_names"] 
        sens_f_index = data["sens_f_index"]
        sens_f_names = data["sens_f_names"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_prob_predictions = data["prob_y_pred"]  
        y_predictions = data["y_pred"]
        if "sens_f_unpriv" in data.keys():
            sens_f_unpriv = data["sens_f_unpriv"]
            sens_f_unpriv_original = data["sens_f_unpriv_original"]
            sens_f_type = data["sens_f_type"]
            print("sens_f_unpriv",sens_f_unpriv)
            print("sens_f_unpriv_original",sens_f_unpriv_original)
            print("sens_f_type",sens_f_type)
        
        if X_train == None:
            X_train = X_test
        if y_train == None:
            y_train = y_test

    if model_availability :
        # explanations
        cursor.execute('SELECT * FROM explanations WHERE token_fk = %s AND original_model = "True" ORDER BY ABS(contrib) DESC',(report_id,))
        postsOriginal = cursor.fetchall()
        
        if len(postsOriginal) == 0:
            return render_template('index2.html', posts=postsOriginal)
        
        cursor.execute('SELECT * FROM explanations WHERE token_fk = %s AND original_model = "False" ORDER BY ABS(contrib) DESC',(report_id,))
        postsEnsemble = cursor.fetchall()
        # end explanations
    else :
        postsOriginal = []
        postsEnsemble = []

    cursor.execute('SELECT DISTINCT indexf, sensitive_f_name FROM calculated_metrics WHERE reportid_fk= %s AND original_model=\'True\'',(report_id,))
    sensitive_features = cursor.fetchall()

    # Data
    histograms = dat.stats(X_train, y_train, sens_f_index, sens_f_names, sens_f_type, sens_f_unpriv, sens_f_unpriv_original)
    corr_heatmap, corr_rankings = dat.correlation_analysis(X_train, sens_f_index, sens_f_names, f_names)
    train_visu = dat.vis_pca(X_train, y_train, sens_f_index, sens_f_names)
    
    # Explanations
    plots = fair_exp.create_plot(cursor, report_id, sensitive_features)
    full_fairness_plots = fair_exp.full_fairness_metrics(cursor, report_id, sensitive_features)

    # Comparison
    temporal_plot = comp.comparison_plot(cursor, report_details[0], sensitive_features)
    
    cursor.close()
    
    return render_template('index.html', 
                           report_details=report_details, 
                           postsOriginal=postsOriginal, 
                           postsEnsemble=postsEnsemble,
                           sensitive_features=sensitive_features, 
                           plot=plots,
                           temporal_plot=temporal_plot,
                           train_visu=train_visu,
                           corr_heatmap=corr_heatmap,
                           corr_rankings=corr_rankings,
                           histograms=histograms,
                           full_fairness_plots=full_fairness_plots)

if __name__ == "__main__":
    app.run(debug=True)