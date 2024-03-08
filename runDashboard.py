import numpy as np
import pandas as pd
import tensorflow as tf
import json
import dill
from joblib import load
import plotly
# Dash related imports
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ExplainerDashboard related imports
from explainerdashboard import  ExplainerDashboard, ExplainerHub
from explainerdashboard.custom import *

import shap
import xgboost as xgb
# Sklearn metrics

# Flask for any additional web server operations
from flask import redirect

from custom_components import ThresholdAdjustmentComponent


class DirectKerasModelWrapper:
    def __init__(self, keras_model):
        self.model = keras_model
    
    def predict(self, X):
        # Convert DataFrame to NumPy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Ensure input X is reshaped to the expected format by the LSTM model
        X_reshaped = X.reshape(-1, 1, X.shape[-1]) if len(X.shape) == 2 else X
        predictions = self.model.predict(X_reshaped)
        # Assuming your model outputs probabilities and you need to convert these to class labels
        return predictions.argmax(axis=-1)
    
    def predict_proba(self, X):
        # Convert DataFrame to NumPy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Reshape input X to the expected 3D format [samples, timesteps, features]
        X_reshaped = X.reshape(-1, 1, X.shape[-1]) if len(X.shape) == 2 else X
        return self.model.predict(X_reshaped)

class RLModelWrapper:
    def __init__(self, keras_model):
        self.model = keras_model

    def predict(self, X):
        """Predicts the class/action with the highest Q-value for each sample."""
        # Convert DataFrame to NumPy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Predict the Q-values for the input
        q_values = self.model.predict(X)
        # Select the action with the highest Q-value for each sample
        predicted_actions = np.argmax(q_values, axis=1)
        return predicted_actions
    
    def predict_proba(self, X):
        """Predicts the Q-values for each action, normalized to sum to 1 (like probabilities)."""
        # Convert DataFrame to NumPy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Predict the Q-values for the input
        q_values = self.model.predict(X)
        # Normalize Q-values to sum to 1, so they mimic probabilities
        q_values_normalized = q_values / q_values.sum(axis=1, keepdims=True)
        return q_values_normalized
    

# Load LSTMUNSW explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/LSTMUNSW_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/LSTMUNSW_explainer_without_model.dill', 'rb') as file:
    LSTMexplainerUNSW = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerUNSW.model = wrapped_model

# Load XGBoostUNSW explainer
xgb_explainerUNSW = load('datasets/explainers/XGBoostUNSW_explainer.joblib')

# Load RFUNSW explainer
rf_explainerUNSW = load('datasets/explainers/RFUNSW_explainer.joblib')

#Load AEUNSW
with open('datasets/explainers/AutoencoderUNSWfiles.dill', 'rb') as file:
    loaded_data = dill.load(file)

loaded_explainerAEUNSW = loaded_data['explainer']
loaded_threshold_componentAEUNSW = loaded_data['threshold_component']

# Load RLDDQNUNSW explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/RLDDQNUNSW_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/RLDDQNUNSW_explainer_without_model.dill', 'rb') as file:
    RLDDQNexplainerUNSW = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerUNSW.model = wrapped_model

# Load LSTMUNSW explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/LSTMCIC_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/LSTMCIC_explainer_without_model.dill', 'rb') as file:
    LSTMexplainerCIC = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerCIC.model = wrapped_model

xgb_explainerCIC = load('datasets/explainers/XGBoostCIC_explainer.joblib')

rf_explainerCIC = load('datasets/explainers/RFCIC_explainer.joblib')

with open('datasets/explainers/AutoencoderCICfiles.dill', 'rb') as file:
    loaded_data = dill.load(file)

loaded_explainerAECIC = loaded_data['explainer']
loaded_threshold_componentAECIC = loaded_data['threshold_component']

# Load RLDDQNCIC explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/RLDDQNCIC_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/RLDDQNCIC_explainer_without_model.dill', 'rb') as file:
    RLDDQNexplainerCIC = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerCIC.model = wrapped_model

# Load LSTMUNSW explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/LSTMINSDN_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/LSTMINSDN_explainer_without_model.dill', 'rb') as file:
    LSTMexplainerINSDN = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerINSDN.model = wrapped_model

xgb_explainerINSDN = load('datasets/explainers/XGBoostINSDN_explainer.joblib')

rf_explainerINSDN = load('datasets/explainers/RFINSDN_explainer.joblib')

with open('datasets/explainers/AutoencoderINSDNfiles.dill', 'rb') as file:
    loaded_data = dill.load(file)

loaded_explainerAEINSDN = loaded_data['explainer']
loaded_threshold_componentAEINSDN = loaded_data['threshold_component']

# Load RLDDQNINSDN explainer
# Load the TensorFlow model
loaded_model = tf.keras.models.load_model('datasets/explainers/RLDDQNINSDN_modelFinal')

# Wrap the loaded model with your DirectKerasModelWrapper
wrapped_model = DirectKerasModelWrapper(loaded_model)

# Load the rest of the explainer
with open('datasets/explainers/RLDDQNINSDN_explainer_without_model.dill', 'rb') as file:
    RLDDQNexplainerINSDN = dill.load(file)

# Re-attach the wrapped model to the explainer
LSTMexplainerINSDN.model = wrapped_model


dashboard1 = ExplainerDashboard(rf_explainerUNSW, mode='external', title="Random Forest UNSW", name="RFUNSW",
            description="The Random Forest classifier with the UNSW-NB15 Dataset", no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
   , bootstrap=dbc.themes.PULSE                           
                        
)
dashboard2 = ExplainerDashboard(xgb_explainerUNSW, title="XGBoost UNSW", name="XGBUNSW",
            description="The XGboost classifier with the UNSW-NB15 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard3 = ExplainerDashboard(LSTMexplainerUNSW, title="LSTM UNSW", name="LSTMUNSW",
            description="The LSTM classifier with the UNSW-NB15 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard4 = ExplainerDashboard(loaded_explainerAEUNSW, [loaded_threshold_componentAEUNSW], title="AE UNSW", name="AEUNSW"   , bootstrap=dbc.themes.PULSE                           
)

dashboard5 = ExplainerDashboard(xgb_explainerCIC, title="XGBoost CIC-IDS 2017", name="XGBCIC",
            description="The XGboost classifier with the CIC-IDS 2017 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard6 = ExplainerDashboard(rf_explainerCIC, title="Random Forest CIC-IDS 2017", name="RFCIC",
            description="The Random Forest classifier with the CIC-IDS 2017 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
       , bootstrap=dbc.themes.PULSE                           
                          
)
dashboard7 = ExplainerDashboard(LSTMexplainerCIC, title="LSTM CIC-IDS 2017", name="LSTMCIC",
            description="The LSTM classifier with the CIC-IDS 2017 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
          , bootstrap=dbc.themes.PULSE                           
                       
)
dashboard8 = ExplainerDashboard(loaded_explainerAECIC, [loaded_threshold_componentAECIC], title="AE CIC", name="AECIC"   , bootstrap=dbc.themes.PULSE )

dashboard9 = ExplainerDashboard(xgb_explainerINSDN, title="XGBoost INSDN", name="XGBINSDN",
            description="The XGboost classifier with the INSDN Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard10 = ExplainerDashboard(rf_explainerINSDN, title="Random Forest INSDN", name="RFINSDN",
            description="The Random Forest classifier with the INSDN Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=True,
   model_summary=True,
   contributions=True,
   whatif=True,
   shap_dependence=True,
   shap_interaction=False,
   decision_trees=False
       , bootstrap=dbc.themes.PULSE                           
                          
)
dashboard11 = ExplainerDashboard(LSTMexplainerINSDN, title="LSTM INSDN", name="LSTMINSDN",
            description="The LSTM classifier with the INSDN Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
                                    , bootstrap=dbc.themes.PULSE                           

)
dashboard12 = ExplainerDashboard(loaded_explainerAEINSDN, [loaded_threshold_componentAEINSDN], title="AE INSDN", name="AEINSDN"   , bootstrap=dbc.themes.PULSE     )

dashboard13 = ExplainerDashboard(RLDDQNexplainerUNSW, title="RL DDQN UNSW", name="RLDDQNUNSW",
            description="The RL DDQN classifier with the UNSW-NB15 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard14 = ExplainerDashboard(RLDDQNexplainerCIC, title="RL DDQN CIC-IDS 2017", name="RLDDQNCIC",
            description="The RL DDQN classifier with the CIC-IDS 2017 Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)
dashboard15 = ExplainerDashboard(RLDDQNexplainerINSDN, title="RL DDQN INSDN", name="RLDDQNINSDN",
            description="The RL DDQN classifier with the INSDN Dataset",
                               mode='external', no_permutations=True, hide_poweredby=True,
   importances=False,
   model_summary=True,
   contributions=False,
   whatif=False,
   shap_dependence=False,
   shap_interaction=False,
   decision_trees=False
      , bootstrap=dbc.themes.PULSE                           
                           
)

hub = ExplainerHub([dashboard1,dashboard2, dashboard3, dashboard4, dashboard5,dashboard6, dashboard7, dashboard8, dashboard9, dashboard10, dashboard11, dashboard12,dashboard13,dashboard14,dashboard15])


# Directly access the Flask application
flask_app = hub.flask_server()

#Add custom routes to the flask_app
@flask_app.route('/')
def home():
    return redirect('/custom-hub')

# Load the JSON string from the file
with open('datasets/explainers/unsw_attack_distribution.json', 'r') as f:
    fig_json = json.load(f)
# Convert the JSON string back into a Plotly figure object
attackUNSWfig = plotly.graph_objs.Figure(json.loads(fig_json))

with open('datasets/explainers/cic-ids_attack_distribution.json', 'r') as f:
    fig_json = json.load(f)

attackCICfig = plotly.graph_objs.Figure(json.loads(fig_json))

with open('datasets/explainers/INSDN_attack_distribution.json', 'r') as f:
    fig_json = json.load(f)

attackINSDNfig = plotly.graph_objs.Figure(json.loads(fig_json))

# Create a separate Dash app for customization
external_stylesheets = [
    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',  # Bootstrap CSS
    'https://use.fontawesome.com/releases/v5.8.1/css/all.css'  # FontAwesome for icons
]

# Create Dash app with external stylesheets
dash_app = Dash(__name__, server=flask_app, url_base_pathname='/custom-hub/', external_stylesheets=external_stylesheets)

# Define the layout with a navigation bar and a content container
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav(className='navbar navbar-expand-lg navbar-dark bg-dark', children=[
        html.Div(className='container-fluid', children=[
            html.A('Network Anomaly Detection', href='/custom-hub/', className='navbar-brand'),
            html.Button(html.Span(className='navbar-toggler-icon'), className='navbar-toggler', type='button',
                        **{'data-bs-toggle': 'collapse', 'data-bs-target': '#navbarNav',
                           'aria-controls': 'navbarNav', 'aria-expanded': 'false',
                           'aria-label': 'Toggle navigation'}),
            html.Div(className='collapse navbar-collapse', id='navbarNav', children=[
                html.Ul(className='navbar-nav', children=[
                    html.Li(html.A('Home', href='/custom-hub/', className='nav-link')),
                    html.Li(html.A('UNSW-NB15', href='/custom-hub/unsw', className='nav-link')),
                    html.Li(html.A('CIC-IDS 2017', href='/custom-hub/cic', className='nav-link')),
                    html.Li(html.A('INSDN', href='/custom-hub/insdn', className='nav-link')),

                    # Add more datasets as list items here
                ])
            ])
        ])
    ]),
    html.Div(id='page-content', className='container mt-5')
])

# Callback to switch page content based on the URL
@dash_app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/custom-hub/':
        return html.Div([
            html.H1("Network Anomaly Detection Project", className='text-center mb-5', style={'color': '#6f42c1'}),
            
            dcc.Markdown("""
                ## Introduction
                This project explores network anomaly detection using various machine learning algorithms.
                Here, you'll find interactive dashboards for different datasets and the machine learning models applied to them. &#9989;  <!-- Checkmark -->
            """, className='mb-5', dangerously_allow_html=True),  # Enable HTML rendering
            
            html.Div([
                html.H2("Aims and Objectives", className='mb-4', style={'color': '#6f42c1'}),
                html.Ul([
                    html.Li([
                        html.Strong("Aim 1:"),
                        " To develop an effective machine learning model capable of detecting network anomalies with high accuracy."
                    ], className='mb-2'),
                    html.Li([
                        html.Strong("Aim 2:"),
                        " To compare the performance of different machine learning methods (supervised, unsupervised, and reinforcement learning) and its models in the context of network security."
                    ], className='mb-4'),
                ], style={'listStyleType': 'none'}),
                
                html.H4("Objectives", className='mb-3', style={'color': '#6f42c1'}),
                html.Ul([
                    html.Li([
                        html.I(className="fas fa-check-circle", style={'color': 'green'}),  # Font Awesome icon
                        " Implement and evaluate various machine learning algorithms for the different methods for anomaly detection."
                    ], className='mb-2'),
                    html.Li([
                        html.I(className="fas fa-check-circle", style={'color': 'green'}),
                        " Analyze the datasets to identify patterns and features significant for detecting network threats."
                    ], className='mb-2'),
                    html.Li([
                        html.I(className="fas fa-check-circle", style={'color': 'green'}),
                        " Enhance the interpretability of machine learning models to provide insights into the decision-making process."
                    ], className='mb-2'),
                ], style={'listStyleType': 'none'}),
            ], className='aims-objectives')
        ], className='container')
    elif pathname == '/custom-hub/unsw':
        return html.Div([
            html.H2("Dataset: UNSW-NB15", className='mb-4', style={'color': '#6f42c1'}),
            dcc.Markdown("""
                **Description**: The UNSW-NB15 dataset features synthetic and real network traffic,
                including normal activities and attack behaviors, for network intrusion detection analysis. The dataset has nine types of attacks, including Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. 
                """, className='mb-5'),
            dcc.Graph(figure=attackUNSWfig),
            html.Div(className='d-grid gap-2', children=[
                html.A('Random Forest Full Analysis', href='/dashboards/RFUNSW/', target='_blank', className='btn btn-primary'),
                html.A('XGBoost Full Analysis', href='/dashboards/XGBUNSW/', target='_blank', className='btn btn-secondary'),
                html.A('LSTM Classification Stats', href='/dashboards/LSTMUNSW/', target='_blank', className='btn btn-secondary'),
                html.A('AutoEncoder Classification Stats', href='/dashboards/AEUNSW/', target='_blank', className='btn btn-secondary'),
                html.A('Reinforcement Learning DDQN Classification Stats', href='/dashboards/RLDDQNUNSW/', target='_blank', className='btn btn-secondary')

            ])
        ])
    # Add more elif statements for other datasets
    elif pathname == '/custom-hub/cic':
        return html.Div([
            html.H2("Dataset: CIC-IDS 2017", className='mb-4', style={'color': '#6f42c1'}),
            dcc.Markdown("""
                **Description**: The CIC-IDS 2017 dataset features a comprehensive set of network traffic data, 
                including a wide variety of intrusions simulated in a military network environment. It includes common attack scenarios such as DDoS, DoS, Brute Force, Heartbleed, and more, designed to benchmark intrusion detection systems.
                """, className='mb-5'),
            dcc.Graph(figure=attackCICfig),  
            html.Div(className='d-grid gap-2', children=[
                html.A('Random Forest Full Analysis', href='/dashboards/RFCIC/', target='_blank', className='btn btn-primary'),
                html.A('XGBoost Full Analysis', href='/dashboards/XGBCIC/', target='_blank', className='btn btn-secondary'),
                html.A('LSTM Classification Stats', href='/dashboards/LSTMCIC/', target='_blank', className='btn btn-secondary'),
                html.A('AutoEncoder Classification Stats', href='/dashboards/AECIC/', target='_blank', className='btn btn-secondary'),
                html.A('Reinforcement Learning DDQN Classification Stats', href='/dashboards/RLDDQNCIC/', target='_blank', className='btn btn-secondary')

            ])
        ])
        
    elif pathname == '/custom-hub/insdn':
        return html.Div([
            html.H2("Dataset: inSDN", className='mb-4', style={'color': '#6f42c1'}),
            dcc.Markdown("""
                **Description**: The inSDN dataset features a comprehensive set of network traffic data,
                tailored specifically for evaluating security mechanisms within Software Defined Networking (SDN) environments. It includes various attack scenarios relevant to SDN infrastructures, designed to test the efficacy of intrusion detection systems in these modern networking setups.
                """, className='mb-5'),
            dcc.Graph(figure=attackINSDNfig),  
            html.Div(className='d-grid gap-2', children=[
                html.A('Random Forest Full Analysis', href='/dashboards/RFINSDN/', target='_blank', className='btn btn-primary'),
                html.A('XGBoost Full Analysis', href='/dashboards/XGBINSDN/', target='_blank', className='btn btn-secondary'),
                html.A('LSTM Classification Stats', href='/dashboards/LSTMINSDN/', target='_blank', className='btn btn-secondary'),
                html.A('AutoEncoder Classification Stats', href='/dashboards/AEINSDN/', target='_blank', className='btn btn-secondary'),
                html.A('Reinforcement Learning DDQN Classification Stats', href='/dashboards/RLDDQNINSDN/', target='_blank', className='btn btn-secondary')

            ])
        ])
    else:
        return '404'

# Assuming you want to redirect from the root to your custom hub
@flask_app.route('/')
def redirect_to_hub():
    return redirect('/custom-hub')

if __name__ == "__main__":
    hub.run(debug=True, use_reloader=False)