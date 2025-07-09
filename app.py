"""
Healthcare ML Dashboard - Deployment Version
Emergency Room Visit Prediction using Machine Learning

This version is optimized for cloud deployment (Heroku, Render, Railway)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
import os

# Generate sample data if CSV doesn't exist (for deployment)
def generate_sample_data():
    np.random.seed(42)
    n_patients = 1000
    
    # Generate synthetic patient data
    patient_ids = [f"P{i:04d}" for i in range(1, n_patients + 1)]
    
    # Create various features
    ages = np.random.randint(18, 85, n_patients)
    
    # Disease types
    disease_types = np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease', 'Respiratory', 'Other'], 
                                    n_patients, p=[0.3, 0.25, 0.15, 0.15, 0.15])
    
    # Medication dispense frequency (related to ER visits)
    med_freq = np.random.randint(1, 20, n_patients)
    
    # Insurance types
    insurance = np.random.choice(['Private', 'Medicare', 'Medicaid', 'None'], 
                                n_patients, p=[0.4, 0.3, 0.2, 0.1])
    
    # Generate target variable (Emergency Room Visits) with some logic
    er_visit_prob = (ages > 60) * 0.3 + (med_freq > 10) * 0.4 + (insurance == 'None') * 0.3
    er_visit_prob = np.clip(er_visit_prob, 0, 1)
    er_visits = np.random.binomial(1, er_visit_prob, n_patients)
    
    # Create DataFrame
    data = pd.DataFrame({
        'PatientID': patient_ids,
        'Age': ages,
        'DiseaseType': disease_types,
        'MedicationDispenseFrequency': med_freq,
        'InsuranceType': insurance,
        'EmergencyRoomVisit': er_visits
    })
    
    return data

# Load or generate dataset
def load_data():
    if os.path.exists("medication_dispense_data.csv"):
        try:
            data = pd.read_csv("medication_dispense_data.csv")
            print("Loaded existing dataset")
        except:
            print("Error loading CSV, generating sample data")
            data = generate_sample_data()
    else:
        print("CSV not found, generating sample data for demonstration")
        data = generate_sample_data()
    
    return data

# Load data
data = load_data()

# Validate required columns
required_columns = ['PatientID', 'EmergencyRoomVisit']
if not all(col in data.columns for col in required_columns):
    print("Missing required columns, using sample data")
    data = generate_sample_data()

# Define features and target
features = data.drop(['PatientID', 'EmergencyRoomVisit'], axis=1)
target = data['EmergencyRoomVisit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Identify column types
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = features.select_dtypes(include=['number']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

# Create results DataFrame
results_df = X_test.copy()
results_df['PatientID'] = data.loc[X_test.index, 'PatientID'].values
results_df['ActualERVisit'] = y_test.values
results_df['PredictedERVisit'] = y_pred
results_df['ERVisitProbability'] = y_pred_proba

# Create visualizations
def create_visualizations(df):
    # Distribution of predictions
    pred_counts = df['PredictedERVisit'].value_counts()
    pie_fig = px.pie(values=pred_counts.values, names=['No ER Visit', 'ER Visit'], 
                     title='Distribution of Predicted Emergency Room Visits',
                     color_discrete_map={'No ER Visit': '#4983C3', 'ER Visit': '#ff6b6b'})
    
    # Probability distribution
    hist_fig = px.histogram(df, x='ERVisitProbability', nbins=20,
                           title='Distribution of ER Visit Probabilities',
                           labels={'ERVisitProbability': 'ER Visit Probability'})
    hist_fig.update_layout(xaxis_title='Probability', yaxis_title='Count')
    
    # Feature analysis (if available)
    if 'DiseaseType' in df.columns:
        disease_analysis = df.groupby('DiseaseType')['ERVisitProbability'].mean().reset_index()
        bar_fig = px.bar(disease_analysis, x='DiseaseType', y='ERVisitProbability',
                        title='Average ER Visit Probability by Disease Type')
        bar_fig.update_layout(xaxis_title='Disease Type', yaxis_title='Average Probability')
    else:
        bar_fig = px.bar(x=['High Risk', 'Medium Risk', 'Low Risk'], 
                        y=[(df['ERVisitProbability'] > 0.7).sum(),
                           ((df['ERVisitProbability'] > 0.3) & (df['ERVisitProbability'] <= 0.7)).sum(),
                           (df['ERVisitProbability'] <= 0.3).sum()],
                        title='Risk Level Distribution')
    
    # Scatter plot
    if 'MedicationDispenseFrequency' in df.columns:
        scatter_fig = px.scatter(df, x='MedicationDispenseFrequency', y='ERVisitProbability',
                                title='Medication Frequency vs ER Visit Probability',
                                color='PredictedERVisit')
    else:
        scatter_fig = px.scatter(df, x=df.index, y='ERVisitProbability',
                                title='Patient Risk Scores',
                                color='PredictedERVisit')
    
    return pie_fig, hist_fig, bar_fig, scatter_fig

# Create visualizations
pie_fig, hist_fig, bar_fig, scatter_fig = create_visualizations(results_df)

# Top risk patients
top_risk_patients = results_df.nlargest(10, 'ERVisitProbability')[['PatientID', 'ERVisitProbability']].copy()
top_risk_patients['ERVisitProbability'] = top_risk_patients['ERVisitProbability'].round(3)

# Metrics for display
metrics_data = [
    {'Metric': 'Model Accuracy', 'Value': f'{accuracy:.3f}'},
    {'Metric': 'Precision', 'Value': f'{precision:.3f}'},
    {'Metric': 'Recall', 'Value': f'{recall:.3f}'},
    {'Metric': 'Total Patients Analyzed', 'Value': f'{len(results_df)}'},
    {'Metric': 'High Risk Patients (>70%)', 'Value': f'{(results_df["ERVisitProbability"] > 0.7).sum()}'},
    {'Metric': 'Medium Risk Patients (30-70%)', 'Value': f'{((results_df["ERVisitProbability"] > 0.3) & (results_df["ERVisitProbability"] <= 0.7)).sum()}'}
]

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # This is important for deployment

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Healthcare ML Dashboard", style={'color': '#0c1e3e', 'text-align': 'center', 'margin-bottom': '10px'}),
        html.H3("Emergency Room Visit Prediction", style={'color': '#4983C3', 'text-align': 'center', 'margin-bottom': '30px'})
    ], style={'background': '#f8f9fa', 'padding': '20px', 'margin-bottom': '30px'}),
    
    # Metrics section
    html.Div([
        html.H4("Model Performance Metrics", style={'color': '#0c1e3e', 'margin-bottom': '20px'}),
        dash_table.DataTable(
            data=metrics_data,
            columns=[{"name": i, "id": i} for i in ['Metric', 'Value']],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#4983C3', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'}
        )
    ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '8px'}),
    
    # Charts section
    html.Div([
        html.Div([
            dcc.Graph(figure=pie_fig, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            dcc.Graph(figure=hist_fig, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=bar_fig, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ]),
    
    # High risk patients table
    html.Div([
        html.H4("Top 10 Highest Risk Patients", style={'color': '#0c1e3e', 'margin-bottom': '20px'}),
        dash_table.DataTable(
            data=top_risk_patients.to_dict('records'),
            columns=[{"name": i, "id": i} for i in top_risk_patients.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#4983C3', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '8px'}),
    
    # Footer
    html.Div([
        html.P("Developed by BMG Capital | Healthcare Analytics & Machine Learning Solutions", 
               style={'text-align': 'center', 'color': '#666', 'margin-top': '40px'})
    ])
], style={'font-family': 'Arial, sans-serif', 'max-width': '1200px', 'margin': '0 auto', 'padding': '20px'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050))) 