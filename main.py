import numpy as np
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_data(original):
    sales_df = original.drop(columns=[original.columns[0], original.columns[1]], axis=1)
    sales_df.dropna(axis=0, inplace=True)
    sales_df['Year'] = sales_df['Year'].astype('int')
    sales_df.replace(['PS2', 'PS3', 'PS4', 'PSV'], 'PS', inplace=True)
    sales_df.replace(['XB', 'XOne', 'X360'], 'Xbox', inplace=True)
    sales_df.replace(['Wii', 'N64', 'GC', 'SNES'], 'NES', inplace=True)
    sales_df.replace(['3DS', 'DS', 'GC', 'GBA'], 'GB', inplace=True)
    sales_df = sales_df.rename(
        columns={"NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP", "Other_Sales": "Other", "Global_Sales": "Total"},
        errors="raise")
    sales_df = sales_df.groupby(['Year', 'Platform', 'Publisher', 'Genre'], as_index=False).agg(
        {'NA': 'sum', 'EU': 'sum', 'JP': 'sum', 'Other': 'sum', 'Total': 'sum'}).reset_index()
    years_to_exclude = sales_df['Year'].value_counts().loc[lambda x: x < 200].index.astype('int64').values.tolist()
    platforms_to_exclude = sales_df['Platform'].value_counts().loc[lambda x: x < 200].index.values.tolist()
    publishers_to_exclude = sales_df['Publisher'].value_counts().loc[lambda x: x < 200].index.values.tolist()
    sales_df = sales_df[~sales_df['Year'].isin(years_to_exclude)]
    sales_df.replace(publishers_to_exclude, 'Other', inplace=True)
    sales_df.replace(platforms_to_exclude, 'Other', inplace=True)
    return sales_df

def get_trend_line_type_options(trend_type):
    if trend_type == 'ols':
        return dict(log_x=True)
    elif trend_type == 'lowess':
        return dict(frac=0.5)
    elif trend_type == 'ewm':
        return dict(halflife=2)
    elif trend_type == 'rolling':
        return dict(window=5)
    elif trend_type == 'expanding':
        return dict(function="mean")
    return None


app = dash.Dash(__name__)

original_df = pd.read_csv('vgsales.csv')
original_df.drop_duplicates(keep='first', inplace=True)
original_df.dropna(inplace=True)
original_df['Year'] = original_df['Year'].astype('int')
originalColumnsNoGSales = original_df.columns.values.tolist()
originalColumnsNoGSales.remove('Global_Sales')

df = prepare_data(original_df)

years = df['Year'].value_counts().index.tolist()
years.sort()

genres = df['Genre'].value_counts().index.tolist()
genres.sort()
platforms = df['Platform'].value_counts().index.tolist()
platforms.sort()
publishers = df['Publisher'].value_counts().index.tolist()
publishers.sort()

salesOptions = [{'label': 'Total sales', 'value': 'Total'},
                {'label': 'NA sales', 'value': 'NA'},
                {'label': 'EU sales', 'value': 'EU'},
                {'label': 'JP sales', 'value': 'JP'},
                {'label': 'Other sales', 'value': 'Other'}]

defaultGroup = 'Genre'
app.layout = html.Div([
    html.H1('Video game sales'),
    html.H2('Data to analyze:'),
    dash.dash_table.DataTable(original_df.to_dict('records'),
                              [{"name": i, "id": i} for i in original_df.columns],
                              id='originalData',
                              page_size=10),
    html.H2('Sales dependency analysis:'),
    html.Div([
        html.Label('Years range:'),
        dcc.RangeSlider(years[0], years[-1], 1, marks={i: '{}'.format(i) for i in years}, id='years-range', value=[years[0], years[-1]]),
        html.Br()
    ]),
    html.Div(
        className="row", children=[
            html.Div([
                html.Label('Sales:'),
                dcc.Dropdown(options=salesOptions, value='Total', id='sales-dd'),
                html.Div(id="sales-dd-div")], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Group by:'),
                dcc.Dropdown(options=[{'label': column, 'value': column} for column in [
                                defaultGroup, 'Publisher', 'Platform']], value=defaultGroup, id='group-dd'),
                html.Div(id="group-dd-div"),
                ], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Filter:'),
                dcc.Dropdown(id='filter-dd', value='All', multi=True),
                html.Div(id="filter-dd-div"),
            ], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Trend type:'),
                dcc.Dropdown(options=[{'label': column, 'value': column}
                                      for column in ['ols', 'rolling', 'ewm', 'lowess', 'expanding']],
                             value='ols', id='trend-dd'),
                html.Div(id="trend-dd-div"),
                ], style={"width": "20%"}),
        ], style=dict(display='flex')),
    html.Br(),
    dcc.Graph(id='dependency'),
    html.H2('Region based sales analysis:'),
    html.Div([
        html.Label('Years range:'),
        dcc.RangeSlider(years[0], years[-1], 1, marks={i: '{}'.format(i) for i in years}, id='years-pie-range',
                        value=[years[0], years[-1]]),
        html.Br()
    ]),
    html.Div(
        className="row", children=[
            html.Div([
                html.Label('Genre:'),
                dcc.Dropdown(options=[{'label': column, 'value': column} for column in genres], value='All', id='genre-dd', multi=True),
                html.Div(id="genre-dd-div")], style={"width": "25%"}),
            html.Div([], style={"width": "12.5%"}),
            html.Div([
                html.Label('Platform:'),
                dcc.Dropdown(options=[{'label': column, 'value': column} for column in platforms], value='All', id='platform-dd', multi=True),
                html.Div(id="platform-dd-div"),
                ], style={"width": "25%"}),
            html.Div([], style={"width": "12.5%"}),
            html.Div([
                html.Label('Publisher:'),
                dcc.Dropdown(options=[{'label': column, 'value': column} for column in publishers], value='All', id='publisher-dd', multi=True),
                html.Div(id="Publisher-dd-div"),
            ], style={"width": "25%"}),
        ], style=dict(display='flex')),
    dcc.Graph(id='region-pie'),
    html.H2('Regression model analysis targeting Global sales:'),
    html.Div(
        className="row", children=[
            html.Div([
                html.Label('Data set columns to build model from:'),
                dcc.Dropdown(id='columns-dd', options=[{'label': column, 'value': column} for column in originalColumnsNoGSales], value=originalColumnsNoGSales, multi=True),
                html.Div(id="columns-dd-div")], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Model score: '),
                html.Div(id='score'),
            ], style={"width": "30%"}),
        ], style=dict(display='flex')),
    html.H2('Prediction analysis targeting Global sales:'),
    html.Div([
        html.Label('Filter source years range:'),
        dcc.RangeSlider(years[0], years[-1], 1, marks={i: '{}'.format(i) for i in years}, id='years-prediction-range',
                        value=[years[-2], years[-1]]),
        html.Br(),
        html.Div( className="row", children=[
            html.Div([
                html.Label('Genre:'),
                dcc.Dropdown(id='genre-prediction-dd', options=[{'label': column, 'value': column} for column in genres], value='All', multi=True),
                html.Div(id="genre-prediction-dd-div")], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Publisher:'),
                dcc.Dropdown(id='publisher-prediction-dd', options=[{'label': column, 'value': column} for column in publishers], value='All', multi=True),
                html.Div(id="publisher-prediction-dd-div"),
            ], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Platform:'),
                dcc.Dropdown(id='platform-prediction-dd', options=[{'label': column, 'value': column} for column in platforms], value='All', multi=True),
                html.Div(id="platform-prediction-dd-div"),
            ], style={"width": "20%"}),
            html.Div([], style={"width": "6%"}),
            html.Div([
                html.Label('Trend type:'),
                dcc.Dropdown(options=[{'label': column, 'value': column}
                                      for column in ['ols', 'rolling', 'ewm', 'lowess', 'expanding']],
                             value='ols', id='trend-prediction-dd'),
                html.Div(id="trend-prediction-dd-div"),
            ], style={"width": "20%"}),
        ], style=dict(display='flex')),
    ]),
    dcc.Graph(id='prediction'),
])


@app.callback(
    dash.dependencies.Output('filter-dd', 'options'),
    [dash.dependencies.Input('group-dd', 'value')]
)
def set_filter_options(group_option):
    return [{'label': i, 'value': i} for i in df[group_option].unique().tolist()]

@app.callback(
    dash.dependencies.Output('dependency', 'figure'),
    [dash.dependencies.Input("years-range", "value"),
    dash.dependencies.Input("sales-dd", "value"),
    dash.dependencies.Input("group-dd", "value"),
    dash.dependencies.Input("filter-dd", "value"),
    dash.dependencies.Input("trend-dd", "value")]
)
def update_graph(years_range, sales_option, group_option, filter_values, trend_type):
    filter_list = parseFilter(filter_values)
    df_sales = df[(df['Year'] >= years_range[0]) & (df['Year'] <= years_range[-1])]
    if 'All' not in filter_list:
        df_sales = df[df[group_option].isin(filter_list)]
    df_sales = df_sales.groupby(['Year', group_option], as_index=False).agg(
        {'NA': 'sum', 'EU': 'sum', 'JP': 'sum', 'Other': 'sum', 'Total': 'sum'}).reset_index()
    figure = px.scatter(df_sales, 'Year', sales_option, title=f"Year vs {sales_option} sales",
                        labels={"Year": "Year", sales_option: f"{sales_option} sales in mln",
                            group_option: group_option},
                        trendline_options=get_trend_line_type_options(trend_type),
                        trendline=trend_type, color=group_option)
    return figure

def parseFilter(filter):
    filter_list = filter if type(filter) == list else [filter]
    return filter_list if filter_list else ['All']

@app.callback(
    dash.dependencies.Output('region-pie', 'figure'),
    [dash.dependencies.Input("years-pie-range", "value"),
    dash.dependencies.Input("genre-dd", "value"),
    dash.dependencies.Input("platform-dd", "value"),
    dash.dependencies.Input("publisher-dd", "value")]
)
def update_pie(years_range, genres_option, platforms_option, publishers_option):
    genre_list = parseFilter(genres_option)
    platform_list = parseFilter(platforms_option)
    publisher_list = parseFilter(publishers_option)
    df_sales = df[(df['Year'] >= years_range[0]) & (df['Year'] <= years_range[-1])]
    if 'All' not in genre_list:
        df_sales = df[df['Genre'].isin(genre_list)]
    if 'All' not in platform_list:
        df_sales = df[df['Platform'].isin(platform_list)]
    if 'All' not in publisher_list:
        df_sales = df[df['Publisher'].isin(publisher_list)]
    df_sales = df_sales.drop(columns=['Total'], axis=1).groupby(['Year'], as_index=False).\
        agg({'NA': 'sum', 'EU': 'sum', 'JP': 'sum', 'Other': 'sum'}).reset_index().sum()
    df_sales.drop(['Year', 'index'], inplace=True)
    names = df_sales.index.values.tolist()
    return px.pie(values=df_sales.values.tolist(), names=names, title='Sales per region')


def prepare_model(columns):
    model = {}
    column_list = parseFilter(columns)
    if 'All' in column_list:
        column_list = originalColumnsNoGSales
    rgr_df = original_df.copy()
    columns_to_drop = [c for c in rgr_df.columns if c not in column_list]
    columns_to_drop.remove('Global_Sales')
    print(columns_to_drop)
    rgr_df.drop(columns=columns_to_drop, inplace=True)

    for c in rgr_df.columns:
        if rgr_df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(rgr_df[c].values))
            rgr_df[c] = lbl.transform(list(rgr_df[c].values))
            model[rgr_df[c].name] = lbl.classes_

    y = rgr_df.pop('Global_Sales')

    x_train, x_test, y_train, y_test = train_test_split(rgr_df, y, test_size=0.20, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    model['model'] = lr
    model['df'] = rgr_df
    model['columns'] = rgr_df.columns.values.tolist()
    model['score'] = f"{lr.score(x_test, y_test) * 100}%"
    return model

@app.callback(
    dash.dependencies.Output('score', 'children'),
    dash.dependencies.Input("columns-dd", "value")
)
def update_score(columns):
    return prepare_model(columns)['score']

@app.callback(
    dash.dependencies.Output('prediction', 'figure'),
    [dash.dependencies.Input("columns-dd", "value"),
    dash.dependencies.Input("years-prediction-range", "value"),
    dash.dependencies.Input("genre-prediction-dd", "value"),
    dash.dependencies.Input("platform-prediction-dd", "value"),
    dash.dependencies.Input("publisher-prediction-dd", "value"),
    dash.dependencies.Input("trend-prediction-dd", "value")]
)
def update_prediction_graph(columns, years_range, genres_option, platforms_option, publishers_option, trend_type):
    model = prepare_model(columns)
    genre_list = parseFilter(genres_option)
    platform_list = parseFilter(platforms_option)
    publisher_list = parseFilter(publishers_option)
    print(f"Model: {columns}, years: {years_range}, genres: {genre_list}, platforms: {platform_list}, publishers: {publisher_list}")
    rgr_df = model['df']
    rgr_columns = model['columns']
    filtered_df = rgr_df[(rgr_df['Year'] > years_range[0]) & (rgr_df['Year'] <= years_range[-1])]
    if 'All' not in genre_list and 'Genre' in rgr_columns:
        genre_list = list(map(lambda v:genre_list.index(v), genre_list))
        filtered_df = filtered_df[filtered_df['Genre'].isin(genre_list)]
    if 'All' not in platform_list and 'Platform' in rgr_columns:
        platform_list = list(map(lambda v:platform_list.index(v), platform_list))
        filtered_df = filtered_df[filtered_df['Platform'].isin(platform_list)]
    if 'All' not in publisher_list and 'Publisher' in rgr_columns:
        publisher_list = list(map(lambda v:publisher_list.index(v), publisher_list))
        filtered_df = filtered_df[filtered_df['Publisher'].isin(publisher_list)]
    print(filtered_df.columns.values)
    new_df = filtered_df.copy()
    yearDiff = years_range[-1] - years_range[0]
    new_df['Year'] = new_df['Year'] + yearDiff
    print (new_df)

    y_pred = model['model'].predict(filtered_df)
    new_df['Global_Sales'] = y_pred

    figure = px.scatter(new_df, 'Year', 'Global_Sales', title=f"Year vs {'Global_Sales'} sales",
                        labels={"Year": "Year", 'Global_Sales': f"{'Global_Sales'} sales in mln"},
                        trendline_options=get_trend_line_type_options(trend_type),
                        trendline=trend_type)
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
