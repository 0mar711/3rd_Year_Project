from explainerdashboard.custom import *
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from explainerdashboard.explainers import BaseExplainer


class ThresholdAdjustmentComponent(ExplainerComponent):
    def __init__(self, explainer, error_normal, error_anomalies, title="Threshold Adjustment", **kwargs):
        super().__init__(explainer, title=title)
        self.error_normal = error_normal
        self.error_anomalies = error_anomalies
        # You can ignore or use kwargs as needed

    def layout(self):
        combined_errors = np.concatenate([self.error_normal, self.error_anomalies])
        return html.Div([
            dcc.Graph(id='accuracy_graph'),
            dcc.Graph(id='metrics_graph'),
            dcc.Slider(
                id='threshold_slider',
                min=0,
                max=100,
                value=50,
                marks={i: f'{np.percentile(combined_errors, i):.2f}' for i in range(0, 101, 10)},
                step=1,
            ),
            dcc.Graph(id='roc_curve_graph'),  # Placeholder for ROC curve graph
            dcc.Graph(id='pr_curve_graph'),  # Placeholder for Precision-Recall curve graph
            html.Pre(id='classification_report')
        ])

    def register_callbacks(self, app):
        @app.callback(
            [Output('accuracy_graph', 'figure'),
             Output('metrics_graph', 'figure'),
             Output('roc_curve_graph', 'figure'),  # New output
             Output('pr_curve_graph', 'figure'),  # New output
             Output('classification_report', 'children')],
            [Input('threshold_slider', 'value')]
        )
        def update_graphs(slider_percentile):
            errors = np.concatenate([self.error_normal, self.error_anomalies])
            y_true = np.concatenate([np.zeros(len(self.error_normal)), np.ones(len(self.error_anomalies))])
        
            # Compute binary predictions based on the chosen threshold for classification report
            threshold = np.percentile(errors, slider_percentile)
            y_pred = errors > threshold
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision_val, recall_val, f1_val = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
            classification_report_text = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])
        
            # Generate metrics and confusion figures based on binary classification
            metrics_fig, confusion_fig = self.create_figures(tn, fp, fn, tp, precision_val, recall_val, f1_val)
        
            # Compute ROC and PR curves using errors as scores
            fpr, tpr, roc_thresholds = roc_curve(y_true, errors)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, errors)
        
            # Find closest threshold points on ROC and PR curves
            closest_roc_index = np.argmin(np.abs(roc_thresholds - threshold))
            closest_pr_index = np.argmin(np.abs(pr_thresholds - threshold))
        
            roc_curve_fig = go.Figure()
            roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            # Add marker for the selected threshold
            roc_curve_fig.add_trace(go.Scatter(x=[fpr[closest_roc_index]], y=[tpr[closest_roc_index]], mode='markers', marker_symbol='circle', marker_size=10, name='Selected Threshold'))
        
            pr_curve_fig = go.Figure()
            pr_curve_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve'))
            # Add marker for the selected threshold
            pr_curve_fig.add_trace(go.Scatter(x=[recall[closest_pr_index]], y=[precision[closest_pr_index]], mode='markers', marker_symbol='circle', marker_size=10, name='Selected Threshold'))
        
            roc_curve_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            pr_curve_fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
        
            # Return the updated figures
            return metrics_fig, confusion_fig, roc_curve_fig, pr_curve_fig, classification_report_text


    def calculate_accuracy(self, threshold):
        # This method can be used if you need to calculate accuracy separately
        pass
    
    def create_figures(self, tn, fp, fn, tp, precision, recall, f1):
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Accuracy, Precision, Recall, F1 Score Indicator
        metrics_fig = go.Figure()
    
        gauges_layout = [
            {"value": precision, "title": "Precision", "domain": {'x': [0, 0.24], 'y': [0, 1]}, "min": 0, "max": 1},
            {"value": recall, "title": "Recall", "domain": {'x': [0.26, 0.49], 'y': [0, 1]}, "min": 0, "max": 1},
            {"value": f1, "title": "F1 Score", "domain": {'x': [0.51, 0.74], 'y': [0, 1]}, "min": 0, "max": 1},
            {"value": accuracy, "title": "Accuracy", "domain": {'x': [0.76, 1], 'y': [0, 1]}, "min": 0, "max": 1}
        ]
    
        for gauge in gauges_layout:
            metrics_fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=gauge["value"],
                domain=gauge["domain"],
                title={'text': gauge["title"]},
                gauge={'axis': {'range': [gauge["min"], gauge["max"]]}, 'bar': {'color': "darkblue"}}
            ))
    
        metrics_fig.update_layout(height=400, title="Precision, Recall, F1 Score, and Accuracy")
      
    
        
        # Confusion Matrix Components Graph remains unchanged
        confusion_fig = go.Figure(data=[
            go.Bar(name='True Positives', x=['Metrics'], y=[tp]),
            go.Bar(name='False Positives', x=['Metrics'], y=[fp]),
            go.Bar(name='True Negatives', x=['Metrics'], y=[tn]),
            go.Bar(name='False Negatives', x=['Metrics'], y=[fn])
        ])
        confusion_fig.update_layout(title="Confusion Matrix Components",
                                    barmode='group',
                                    height=400)
        
        return metrics_fig, confusion_fig
