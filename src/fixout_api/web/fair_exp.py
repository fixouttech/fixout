import pandas as pd
import numpy as np

import plotly
import plotly.graph_objs as go

import json


def create_plot(cursor, report_id, sensitive_features):

    graphJSON_dic = []
    for sensitive_f, name in sensitive_features:

        cursor.execute('SELECT * FROM calculated_metrics WHERE original_model = \'True\' AND reportid_fk = %s AND indexf = %s',(report_id, sensitive_f,))
        metrics_original = cursor.fetchall()

        cursor.execute('SELECT * FROM calculated_metrics WHERE original_model = \'False\' AND reportid_fk = %s AND indexf = %s',(report_id, sensitive_f,))
        metrics_ensemble = cursor.fetchall()

        #print (metrics_original)
        y_original = [metric[2] for metric in metrics_original]
        y_ensemble = [metric[2] for metric in metrics_ensemble]

        x = ['DP','EO','PE']
        df_original = None
        df_ensemble = None
        if (len(y_original) == len(x)) :
            df_original = pd.DataFrame({'x': x, 'y': y_original}) # creating a sample dataframe
        if (len(y_ensemble) == len(x)) :
            df_ensemble = pd.DataFrame({'x': x, 'y': y_ensemble}) # creating a sample dataframe

        data = []
        if df_original is not None:
            data.append(
                go.Bar(
                    name='Original',
                    x=df_original['x'], # assign x as the dataframe column 'x'
                    y=df_original['y']
                )
            )
        if df_ensemble is not None:
            data.append(
                go.Bar(
                    name='FixOut',
                    x=df_ensemble['x'], # assign x as the dataframe column 'x'
                    y=df_ensemble['y']
                )
            )

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON_dic.append((name, graphJSON))

    return graphJSON_dic


def full_fairness_metrics(cursor, reportId, sensitive_features):

    graphJSON_list = []
    alerts = []

    cursor.execute('SELECT DISTINCT metric_name_fk FROM calculated_metrics WHERE reportid_fk = %s',(reportId,))
    metrics = cursor.fetchall()

    for sensitive_f in sensitive_features:
        
        sensitive_f_name = sensitive_f[1]

        for metric_name in metrics:

            alert_flag = False

            cursor.execute('SELECT m_value, threshold_value, report_name FROM calculated_metrics INNER JOIN reports ON reportid_fk = report_id  WHERE reportid_fk = %s AND metric_name_fk = %s AND sensitive_f_name = %s AND original_model = \'True\'',(reportId, metric_name, sensitive_f_name,))
            metrics_original = cursor.fetchall()

            x_original = []
            y_original = []
            for metric in metrics_original:
                y_original.append(metric[0])
                x_original.append(metric[1])

            df_original = pd.DataFrame({'x': x_original, 'y': y_original}) # creating a sample dataframe

            data = [
                go.Bar(
                    #name='Original',
                    x=df_original['x'], # assign x as the dataframe column 'x'
                    y=df_original['y']
                )
            ]

            alert_array = abs(df_original['y']) >= 0.25
            if alert_array.any():
                alert_flag = True

            graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
            graphJSON_list.append((sensitive_f_name,metric_name,alert_flag,graphJSON))
            
    return graphJSON_list



def full_fairness_metrics2(fair_metrics, num_metrics):

    graphJSON_list = []

    for sensitive_feature, calculated_metrics in fair_metrics: # for each sensitive feature
        dat = {}
        
        for value, metrics in calculated_metrics.items(): # for each value in the sensitive feature
            
            str_value = str(value)

            for metric in metrics: # for each calculated metric 

                str_metric = str(metric.type)
                if str_metric not in dat:
                    dat[str_metric] = {}
                dat[str_metric][str_value] = metric.value

        for i in range(len(num_metrics)): # for each calculated metric
            
            x_ = []
            y_ = []

            if str(i) in dat:
                for threshold, calc_metric in dat[str(i)].items():
                    x_.append(float(threshold))
                    y_.append(calc_metric)
        
            df = pd.DataFrame({'x': x_, 'y': y_}) # creating a sample dataframe

            data = [
                go.Bar(
                    x=df['x'], # assign x as the dataframe column 'x'
                    y=df['y']
                )
            ]

            alert_flag = False
            alert_array = abs(df['y']) >= 0.25
            if alert_array.any():
                alert_flag = True

            graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
            graphJSON_list.append((sensitive_feature.name, str(i), alert_flag, graphJSON))
            
    return graphJSON_list
