import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dash_daq as daq
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from dash import Dash, dcc, html, Input, Output, callback,  dash_table
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import plotly.graph_objects as go
import plotly.express as px

messy_data = pd.read_csv("messy_data.csv", skipinitialspace=True)
data = pd.read_csv("data.csv")
data_encoded = pd.read_csv("data_encoded.csv", skipinitialspace=True)
data_selected = pd.read_csv("data_selected.csv", skipinitialspace=True)

app = Dash(__name__)


app.layout = html.Div([
    html.Div(
        className='header-container',
        children = [
        html.H1("Konrad Dusza S21516 - Projekt PAD"),
        html.Img(src=app.get_asset_url('LOGO-PJWSTK.png')),]
    ),
    html.Hr(),
    html.H2("Etapy przygotowania danych"),
    html.Br(),
    html.Div(
        className='radio-buttons',
        children = [dcc.RadioItems(['Początkowe ', 'Uporządkowane ','Zakodowane ', 'Wybrane atrybuty '], 'Początkowe ', id='data-stages-radio', inline=True)]
    ),
    html.Div(id='dataset'),

    html.Br(),
    html.Br(),
    html.Hr(),
    html.H2("Analiza atrybutów numerycznych"),
    html.Div(
        className='input-container',
        children=[
        dcc.Dropdown(
            ['carat', 'x dimension', 'y dimension', 'z dimension'],
            'carat',
            id='x-property-dropdown'
        ),
        daq.ToggleSwitch(
            label='Grupuj wartości',
            value=False,
            id='group-price-scat-toggle',
            className='radio-buttons',
        )
    ]),
    html.Div([dcc.Graph(id='scatter-plot-property')]),

    html.Br(),
    html.Br(),
    html.Hr(),
    html.H2("Analiza atrybutów kategorycznych"), 
    html.Div(
        className='input-container',
        children=[
        dcc.Dropdown(
            ['clarity', 'color', 'cut'],
            'clarity',
            id='cat-property-dropdown'
        ),
        html.Div(
            className='radio-buttons',
            children = [dcc.RadioItems(['Liczebność', 'Średnia cena'], 'Liczebność', id='bar-type-radio', inline=True)]
        ),
    ]),
    html.Div([dcc.Graph(id='bar-plot-property')]),

    html.Br(),
    html.Br(),
    html.Hr(),
    html.H2("Eliminacja wsteczna atrybutów w modelu regresji liniowej"),
        
    html.Div(
        className='input-container',
        children=[
        html.Div(
            className='radio-buttons',
            children = [dcc.RadioItems(['MSE', 'R^2', 'Adjusted R^2'], 'MSE', id='evaluation-type-radio', inline=True)]
        ),
    ]),
    html.Div([dcc.Graph(id='scatter-plot-evaluation')]),

    html.Br(),
    html.Br(),
    html.Hr(),
    html.H2("Wizualizacja modelu regresji liniowej"),
        
    html.Div(
        className='input-container',
        children=[
        html.Div(
            className='radio-buttons',
            children = [dcc.RadioItems(['zbiór testowy', 'zbiór treningowy'], 'zbiór testowy', id='evaluation-dataset-radio', inline=True)]
        ),
        daq.ToggleSwitch(
            label='Skala osi X według wartości przewidywanej',
            value=False,
            id='residuals-x-scale-toggle',
            className='radio-buttons',
        )
    ]),
    html.Div([dcc.Graph(id='bar-plot-residuals')]),
    

])

#-------------------------------------------------------------------------------------------

@app.callback(
    Output('bar-plot-residuals', 'figure'),
    Input('evaluation-dataset-radio', 'value'),
    Input('residuals-x-scale-toggle', 'value'),
    )
def update_graph_residuals_bar(dataset_type, change_x_scale):
    
    if(dataset_type == 'zbiór testowy'):
        residuals_df = pd.concat([y_test_final, y_pred_final_test], axis=1, keys=['price_real', 'price_pred'])
        t="Wizualizacja błędów predykcji atrybutu 'price' przez model regresji liniowej na podstawie danych testowych"
    else:
        residuals_df = pd.concat([y_train_final, y_pred_final_train], axis=1, keys=['price_real', 'price_pred'])
        t="Wizualizacja błędów predykcji atrybutu 'price' przez model regresji liniowej na podstawie danych treningowych"

    residuals_df = residuals_df.sort_values(by='price_pred')

    xaxis=residuals_df['price_pred'] if change_x_scale else range(1, len(residuals_df['price_pred']) + 1)

    fig = px.bar(x=xaxis, y=residuals_df['price_real']-residuals_df['price_pred'], title=t, color = ['wartość przewidywana mniejsza od rzeczywistej' if pred > real else 'wartość przewidywana większa od rzeczywistej' for pred, real in zip(residuals_df['price_real'], residuals_df['price_pred'])])
    fig.update_traces(width= 40 if change_x_scale else 1)
    fig.update_xaxes(title= 'predicted price' if change_x_scale else '')
    fig.update_layout(style_plot_background)
    fig.update_yaxes(title='residuals')
    

    return fig

#-------------------------------------------------------------------------------------------

@app.callback(
    Output('scatter-plot-evaluation', 'figure'),
    Input('evaluation-type-radio', 'value'),
    )
def update_graph_eval_scatter(eval_param):
    

    fig = px.scatter(x=range(len(eval_params_dict[eval_param])), y=eval_params_dict[eval_param], title=f"Zmiana '{eval_param}' w kolejnych etapach eliminacji wstecznej dla p-value > 0.05", height=680)
    fig.update_layout(style_plot_background, xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4,5],
        ticktext = list_columns
    ))
    fig.update_xaxes(title='', tickangle= -90)
    fig.update_yaxes(title=eval_param)
    

    return fig

#-------------------------------------------------------------------------------------------

@app.callback(
    Output('bar-plot-property', 'figure'),
    Input('cat-property-dropdown', 'value'),
    Input('bar-type-radio', 'value'),
    )
def update_graph_bar(cat, bar_type):
    
    if(bar_type == 'Średnia cena'):
        y1 = data.groupby(cat)['price'].mean()
        x1 = y1.index
        t=f"Wykres średniej wartości atrybutu decyzyjnego 'price' od atrybutu kategorycznego '{cat}'"
    else:
        y1 = data[cat].value_counts()
        x1 = y1.index
        t=f"Wykres liczebności poszczególnych wartości atrybutu kategorycznego '{cat}'"


    fig = px.bar(x=x1, y=y1, title=t)
    fig.update_layout(style_plot_background)
    fig.update_xaxes(title=cat)
    fig.update_yaxes(title="mean price" if bar_type == 'Średnia cena' else 'price')

    return fig

#-------------------------------------------------------------------------------------------

@app.callback(
    Output('scatter-plot-property', 'figure'),
    Input('x-property-dropdown', 'value'),
    Input('group-price-scat-toggle', 'value'),
    )
def update_graph_scatter(xaxis, group):
    

    if(not group):
        y1 = data['price']
        x1 = data[xaxis]
    else:
        y1 = data.groupby(xaxis)['price'].mean()
        x1 = y1.index

    fig = px.scatter(x=x1, y=y1, title=f"Wykres {'średniej wartości' if group else ''} atrybutu decyzyjnego 'price' od atrybutu '{xaxis}'")
    fig.update_layout(style_plot_background)
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title="mean price" if group else 'price')
    

    return fig


#-------------------------------------------------------------------------------------------

@callback(
    Output('dataset', 'children'),
    Input('data-stages-radio', 'value')
)
def update_output(value):
    if value == 'Początkowe ':
        return dash_table.DataTable(messy_data.head(15).to_dict('records'), [{"name": i, "id": i} for i in messy_data.columns], style_table=style_tab, style_header=style_tab_head, style_cell=style_tab_cell, style_cell_conditional=style_tab_cell_cond)
    elif value == 'Uporządkowane ':
        return dash_table.DataTable(data.head(15).to_dict('records'), [{"name": i, "id": i} for i in data.columns], style_table=style_tab, style_header=style_tab_head, style_cell=style_tab_cell, style_cell_conditional=style_tab_cell_cond)
    elif value == 'Zakodowane ':
        return dash_table.DataTable(data_encoded.head(15).to_dict('records'), [{"name": i, "id": i} for i in data_encoded.columns], style_table=style_tab, style_header=style_tab_head, style_cell=style_tab_cell, style_cell_conditional=style_tab_cell_cond)
    elif value == 'Wybrane atrybuty ':
        return dash_table.DataTable(data_selected.head(15).to_dict('records'), [{"name": i, "id": i} for i in data_selected.columns], style_table=style_tab, style_header=style_tab_head, style_cell=style_tab_cell, style_cell_conditional=style_tab_cell_cond)

#-------------------------------------------------------------------------------------------

c1 = "rgb(234, 230, 255)"
c2 = 'rgb(213, 204, 255)'
style_plot_background = {"paper_bgcolor": c1, "plot_bgcolor": c2}
style_tab = {'borderRadius': '8px', 'overflow': 'hidden'}
style_tab_head = {'backgroundColor': 'slateblue', 'color': 'white', 'fontWeight': 'bold'}
style_tab_cell={'padding': '6px', 'textAlign': 'center', 'backgroundColor': c1}
#style_tab_cell={'padding': '5px', 'textAlign': 'left', 'backgroundColor': '#C0C0C0'}
style_tab_cell_cond =[{
            'if': {'column_id': 'price'},
            'textAlign': 'right',
            'backgroundColor': c2
        }]



def back_elimination():
    X_3 = data_encoded.drop(columns=['price'])
    X_3 = sm.add_constant(X_3)
    y_3 = data_encoded.price

    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.1, random_state=42)

    list_mse_3=[]
    list_R2_3=[]
    list_adj_R2=[]
    list_columns=[]

    num_features = X_train_3.shape[1]
    for i in range(num_features):
        model = sm.OLS(y_train_3, X_train_3).fit()

        y_pred_3 = model.predict(X_test_3[X_train_3.columns])
        list_mse_3.append(mean_squared_error(y_test_3, y_pred_3))
        r2 = r2_score(y_test_3, y_pred_3)
        list_R2_3.append(r2)
        n = len(y_test_3)
        k = X_test_3.shape[1]
        list_adj_R2.append(1 - ((1 - r2) * (n - 1) / (n - k - 1)))
        list_columns.append(' '.join(X_test_3.columns))
        
        max_pvalue = max(model.pvalues)
        if max_pvalue > 0.05:
            remove_index = np.argmax(model.pvalues[1:]) + 1
            X_train_3 = X_train_3.drop(X_train_3.columns[remove_index], axis=1)
            X_test_3 = X_test_3.drop(X_test_3.columns[remove_index], axis=1)
        else:
            break


    y_pred_3 = model.predict(X_test_3[X_train_3.columns])
    mse_3 = mean_squared_error(y_test_3, y_pred_3)
    r2_3 = r2_score(y_test_3, y_pred_3)


    return list_mse_3, list_R2_3, list_adj_R2, list_columns


list_mse, list_R2, list_adj_R2, list_columns = back_elimination()

eval_params_dict = {
    'MSE': list_mse,
    'R^2': list_R2,
    'Adjusted R^2': list_adj_R2
}

X_train_final = pd.read_csv("X_train_final.csv", skipinitialspace=True)
X_test_final = pd.read_csv("X_test_final.csv", skipinitialspace=True)
y_train_final = pd.read_csv("y_train_final.csv", skipinitialspace=True, index_col=None, header=None, skiprows=1,squeeze=True)
y_test_final = pd.read_csv("y_test_final.csv", skipinitialspace=True, index_col=None, header=None, skiprows=1, squeeze=True)

loaded_final_model = joblib.load('final_model.joblib')
y_pred_final_test = loaded_final_model.predict(X_test_final)
y_pred_final_train = loaded_final_model.predict(X_train_final)



if __name__ == '__main__':
    app.run_server(debug=True)