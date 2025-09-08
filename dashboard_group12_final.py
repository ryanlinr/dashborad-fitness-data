#%% import modules
import pandas as pd
import numpy as np
import plotly.express as px
from dash import dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
#%% load the data 
df = pd.read_csv("fitness_dataset.csv")
#data cleanse 
print(df['smokes'].unique())
df['smokes'] = df['smokes'].replace(['0', 0], 'no').replace(['1', 1], 'yes')
print(df['smokes'].unique())
df = df.dropna() #sleep hours had 160 missing values
#############################################################3#############
#%% 
df.dtypes
#%%
# Prepare data
feature_cols = [col for col in df.columns if col not in ['smokes', 'gender', 'is_fit'  ]]
X = df[feature_cols]
y = df['is_fit']
#%%
# Find top 3 correlated features with is_fit
corrs = df[feature_cols + ['is_fit']].corr()['is_fit'].abs().sort_values(ascending=False)
top3 = corrs.index[1:4].tolist()  # skip 'is_fit' itself
print(top3)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#%%
# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Fitness Data Explorer"),
    html.Div([
        html.Label("X-axis:"),
        dcc.Dropdown(id='xcol', 
                     options=[{'label': c, 'value': c} for c in feature_cols], 
                     value=feature_cols[0]),
        html.Label("Y-axis:"),
        dcc.Dropdown(id='ycol', 
                     options=[{'label': c, 'value': c} for c in feature_cols], 
                     value=feature_cols[1]),
        html.Label("Data View:"),
        dcc.Dropdown(id='data_view',
                     options=[
                         {'label': 'All', 'value': 'all'},
                         {'label': 'Fit', 'value': 'Fit'}, 
                         {'label': 'Not Fit', 'value': 'Not Fit'}
                         ],
        value='all',
        clearable=False,),
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Graph(id='scatter'),
    #heatmap
    html.H4("Correlation heatmap"),
    dcc.Graph(id='heatmaps'),
    #dropdown of boxplot
    html.H4("Boxplot with Top 3 variable by Fitness Status"),
    dcc.Dropdown(
        id='box-dropdown',
        options=[
            {'label': 'Age', 'value': 'age'},
            {'label': 'Nutrition quality', 'value': 'nutrition_quality'},
            {'label': 'Activity Index', 'value': 'activity_index'}
        ],
        value='age'
    ),

    html.Div(id='box3-plots'),


    html.H4("Model Training"),
    html.Label("Select Model:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Linear Regression', 'value': 'lr'},
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'SVM', 'value': 'svm'}
        ],
        value='lr'
    ),
    html.Div(id='mae-output')
], style={'margin': '50px'})

# Create is_fit_label for data view dropdown
df['is_fit_label'] = df['is_fit'].map({0: 'Not Fit', 1: 'Fit'})

@app.callback(
    Output('scatter', 'figure'),
    Input('xcol', 'value'),
    Input('ycol', 'value'),
    Input('data_view', 'value')
)

def update_scatter(xcol, ycol, selected_fit_label):
    if selected_fit_label == 'all':
        fig = px.scatter(df, x=xcol, y=ycol, color='is_fit', title=f"{xcol} vs {ycol} colored by is_fit")
    else:
        # Have different colors for selected and non-selected fitness levels
        selected_df = df[df['is_fit_label'] == selected_fit_label]
        opposite_df = df[df['is_fit_label'] != selected_fit_label]
        
        fig = px.scatter(opposite_df, x=xcol, y=ycol, color_discrete_sequence=['#cccccc'], title=f"{xcol} vs {ycol} colored by is_fit")
        # Pronounced color for selected fitness level
        fig.add_trace(px.scatter(
            selected_df,
            x=xcol,
            y=ycol,
            color_discrete_sequence=['#FFA000'], 
        ).data[0])

    return fig

#callback fror heatmap:
@app.callback(
    Output('heatmaps', 'figure'),
    Input('xcol', 'value')
)
def update_heatmap(_):
    corr = df[['age', 'height_cm', 'weight_kg', 'heart_rate', 
               'blood_pressure', 'sleep_hours', 
               'nutrition_quality', 'activity_index', 'is_fit']].corr()
    fig1 = px.imshow(
        corr, text_auto=True, color_continuous_scale='RdBu', 
        zmin=-1, zmax=1,
        width=800, height=800
    )
    return fig1


    #def update_top3(_):
    graphs = []
    for col in top3:
        fig = px.box(df, x='is_fit', y=col, color='is_fit', title=f"{col} vs is_fit")
        graphs.append(dcc.Graph(figure=fig))
    return fig1, graphs
#callback for boxplot
@app.callback(
    Output('box3-plots', 'children'),
    Input('box-dropdown', 'value')  # dummy input to trigger
)
def update_top3(value_box):
    graphs = []
    fig = px.box(df, x='is_fit', y=value_box, color='is_fit', title=f"{value_box} vs is_fit")
    graphs.append(dcc.Graph(figure=fig))
    return graphs
#callback for model training
@app.callback(
    Output('mae-output', 'children'),
    Input('model-dropdown', 'value')
)
def train_and_report(model_name):
    if model_name == 'lr':
        model = LinearRegression()
    elif model_name == 'rf':
        model = RandomForestRegressor(random_state=42)
    else:
        model = SVR()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=np.float64)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    return f"Mean Absolute Error on Test Set: {mae:.3f} (Baseline: {baseline_mae:.3f})"


if __name__ == '__main__':
    app.run(debug=True, port = 8053)
    print("Dash app running at http://127.0.0.1:8053/")
# %%
