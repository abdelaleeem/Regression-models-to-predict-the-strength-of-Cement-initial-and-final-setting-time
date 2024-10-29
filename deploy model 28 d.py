import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load the trained model
with open("Linear_Reg_28d_strength_model.joblib", "rb") as file:
    model = joblib.load(file)
    
with open("Linear_Reg_28d_strength_model_scaler.joblib", "rb") as file:
    scaler = joblib.load(file)
# Initialize Dash
app = dash.Dash(__name__)

# Dash Layout
app.layout = html.Div([
    html.H1("28 Days Strength Prediction Model"),
    
    # Input fields for each feature
    html.Label("CaO:"),
    dcc.Input(id='CaO', type='number', required=True),
    
    html.Label("LOI:"),
    dcc.Input(id='LOI', type='number', required=True),
    
    html.Label("SM:"),
    dcc.Input(id='SM', type='number', required=True),
    
    
    html.Label("2 d:"),
    dcc.Input(id='2d', type='number', required=True),
    
    html.Label("Blaine:"),
    dcc.Input(id='Blaine', type='number', required=True),
    
    html.Label("90 m R:"),
    dcc.Input(id='90mR', type='number', required=True),
    

    
    html.Button('Predict', id='submit-val', n_clicks=0),
    html.Div(id='output-container')
])

# Callback to update the prediction based on input values
@app.callback(
    Output('output-container', 'children'),
    [Input('CaO', 'value'),
     Input('LOI', 'value'),
     Input('SM', 'value'),
     
     Input('2d', 'value'),
     Input('Blaine', 'value'),
   
     Input('90mR', 'value'),
     
     Input('submit-val', 'n_clicks')]
)
def update_prediction(CaO, LOI, SM, d2, Blaine, R90, n_clicks):
    if n_clicks > 0:
        # Create a DataFrame for the input data
        input_data = scaler.transform(pd.DataFrame([[CaO, LOI, SM, d2, Blaine, R90]],
                                  columns=['CaO', 'LOI', 'SM', '2 d', 'Blaine', '90 m R']))

        # Predict using the model
        prediction = model.predict(input_data)[0]
        return f'Predicted 28 Days Strength: {prediction}'
    return ''

# Run the Dash server
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)



