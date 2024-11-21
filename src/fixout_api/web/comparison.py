import pandas as pd
import numpy as np

import plotly
import plotly.graph_objs as go

import json

def comparison_plot(cursor, token, sensitive_features):

    graphJSON_list = []

    cursor.execute('SELECT DISTINCT metric_name_fk FROM calculated_metrics WHERE reportid_fk IN (SELECT report_id FROM reports WHERE token_fk = %s)',(token,))
    metrics = cursor.fetchall()

    for sensitive_f in sensitive_features:

        sensitive_f_name = sensitive_f[1]

        for metric_name in metrics:

            cursor.execute('SELECT m_value, threshold_value, report_name FROM calculated_metrics INNER JOIN reports ON reportid_fk = report_id  WHERE reportid_fk IN (SELECT report_id FROM reports WHERE token_fk = %s) AND metric_name_fk = %s AND sensitive_f_name = %s AND original_model = \'True\'',(token, metric_name, sensitive_f_name,))
            metrics_original = cursor.fetchall()

            cursor.execute('SELECT m_value, threshold_value, report_name FROM calculated_metrics INNER JOIN reports ON reportid_fk = report_id  WHERE reportid_fk IN (SELECT report_id FROM reports WHERE token_fk = %s) AND metric_name_fk = %s AND sensitive_f_name = %s AND original_model = \'False\'',(token, metric_name, sensitive_f_name,))
            metrics_ensemble = cursor.fetchall()

            x_original = []
            x_ensemble = []
            y_original = []
            y_ensemble = []
            for metric in metrics_original:
                y_original.append(metric[0])
                x_original.append(metric[1])
            for metric in metrics_ensemble:
                y_ensemble.append(metric[0])
                x_ensemble.append(metric[1])


            df_original = pd.DataFrame({'x': x_original, 'y': y_original}) # creating a sample dataframe
            df_ensemble = pd.DataFrame({'x': x_ensemble, 'y': y_ensemble}) # creating a sample dataframe

            data = [
                go.Bar(
                    name='Original',
                    x=df_original['x'], # assign x as the dataframe column 'x'
                    y=df_original['y']
                ),
                go.Bar(
                    name='FixOut',
                    x=df_ensemble['x'], # assign x as the dataframe column 'x'
                    y=df_ensemble['y']
                )
            ]

            graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
            graphJSON_list.append((sensitive_f_name,metric_name,graphJSON))

    return graphJSON_list
