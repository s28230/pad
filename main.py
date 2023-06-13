import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression


def group_data(original):
    df = original.drop(columns=[original.columns[0], original.columns[1]], axis=1)
    df = df.groupby(['Year', 'Platform', 'Publisher', 'Genre'], as_index=False).agg(
        {'NA': 'sum', 'EU': 'sum', 'JP': 'sum', 'Other': 'sum', 'Total': 'sum'}).reset_index()
    return df


def clear_data(original):
    df = original_df.drop_duplicates(keep='first')
    df.dropna(inplace=True)
    df['Year'] = df['Year'].astype('int')
    df.replace(['PS2', 'PS3', 'PS4', 'PSV'], 'PS', inplace=True)
    df.replace(['XB', 'XOne', 'X360'], 'Xbox', inplace=True)
    df.replace(['Wii', 'N64', 'GC', 'SNES'], 'NES', inplace=True)
    df.replace(['3DS', 'DS', 'GC', 'GBA'], 'GB', inplace=True)
    df = df.rename(
        columns={"NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP", "Other_Sales": "Other", "Global_Sales": "Total"},
        errors="raise")
    years_to_exclude = df['Year'].value_counts().loc[lambda x: x < 200].index.astype('int64').values.tolist()
    platforms_to_exclude = df['Platform'].value_counts().loc[lambda x: x < 200].index.values.tolist()
    publishers_to_exclude = df['Publisher'].value_counts().loc[lambda x: x < 200].index.values.tolist()
    df = df[~df['Year'].isin(years_to_exclude)]
    df.replace(publishers_to_exclude, 'Other', inplace=True)
    df.replace(platforms_to_exclude, 'Other', inplace=True)
    return df


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
cleared_df = clear_data(original_df)

originalColumnsNoGSales = cleared_df.columns.values.tolist()
originalColumnsNoGSales.remove('Total')

grouped_df = group_data(cleared_df)

years = grouped_df['Year'].value_counts().index.tolist()
years.sort()
genres = grouped_df['Genre'].value_counts().index.tolist()
genres.sort()
platforms = grouped_df['Platform'].value_counts().index.tolist()
platforms.sort()
publishers = grouped_df['Publisher'].value_counts().index.tolist()
publishers.sort()

salesOptions = [{'label': 'Total sales', 'value': 'Total'},
                {'label': 'NA sales', 'value': 'NA'},
                {'label': 'EU sales', 'value': 'EU'},
                {'label': 'JP sales', 'value': 'JP'},
                {'label': 'Other sales', 'value': 'Other'}]

defaultGroup = 'Genre'



app.layout = html.Div([
    html.H1('Video game sales analysis'),
    dcc.Tabs(id="tabs", value='data', children=[
        dcc.Tab(label='Data to analyze', value='data'),
        dcc.Tab(label='Sales dependency analysis', value='sales'),
        dcc.Tab(label='Region based sales analysis', value='region'),
        dcc.Tab(label='Regression model analysis targeting Global sales', value='regression'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(dash.dependencies.Output('tabs-content', 'children'),
              dash.dependencies.Input('tabs', 'value'))
def render_content(tab):
    if tab == 'data':
        return dash.dash_table.DataTable(original_df.to_dict('records'),
                              [{"name": i, "id": i} for i in original_df.columns],
                              id='originalData',
                              page_size=10)
    elif tab == 'sales':
        return html.Div([
            html.Div([html.Label('Years range:'),
                        dcc.RangeSlider(years[0], years[-1], 1, marks={i: '{}'.format(i) for i in years}, id='years-range', value=[years[0], years[-1]]),
                    ]),
            html.Div(className="row", children=[
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
            dcc.Graph(id='dependency')
        ])
    elif tab == 'region':
        return html.Div([
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
                        dcc.Dropdown(options=[{'label': column, 'value': column} for column in genres], value='All',
                                     id='genre-dd', multi=True),
                        html.Div(id="genre-dd-div")], style={"width": "25%"}),
                    html.Div([], style={"width": "12.5%"}),
                    html.Div([
                        html.Label('Platform:'),
                        dcc.Dropdown(options=[{'label': column, 'value': column} for column in platforms], value='All',
                                     id='platform-dd', multi=True),
                        html.Div(id="platform-dd-div"),
                    ], style={"width": "25%"}),
                    html.Div([], style={"width": "12.5%"}),
                    html.Div([
                        html.Label('Publisher:'),
                        dcc.Dropdown(options=[{'label': column, 'value': column} for column in publishers], value='All',
                                     id='publisher-dd', multi=True),
                        html.Div(id="Publisher-dd-div"),
                    ], style={"width": "25%"}),
                ], style=dict(display='flex')),
            dcc.Graph(id='region-pie')
        ])
    elif tab == 'regression':
        return html.Div([
            html.Div(
                className="row", children=[
                    html.Div([
                        html.Label(
                            'Data set columns to build model from (Year ,Genre, Platform, Publisher are mandatory but do not affect model score much):'),
                        dcc.Dropdown(id='columns-dd',
                                     options=[{'label': column, 'value': column} for column in originalColumnsNoGSales],
                                     value=originalColumnsNoGSales, multi=True),
                        html.Div(id="columns-dd-div")], style={"width": "50%"}),
                    html.Div([], style={"width": "5%"}),
                    html.Div([
                        html.Label('Model type: '),
                        dcc.Dropdown(id='model-dd',
                                     options=[{'label': column, 'value': column} for column in ['Linear', 'Lasso', 'Ridge', 'ElasticNet', 'PLS', 'PCA']],
                                     value='Linear'),
                        html.Div(id='score'),
                    ], style={"width": "20%"}),
                    html.Div([], style={"width": "5%"}),
                    html.Div([
                        html.Label('Model score: '),
                        html.Div(id='score'),
                    ], style={"width": "20%"}),
                ], style=dict(display='flex')),
            html.H2('Prediction analysis targeting Global sales:'),
            html.Div([
                html.Label('Filter source years range:'),
                dcc.RangeSlider(years[0], years[-1], 1, marks={i: '{}'.format(i) for i in years},
                                id='years-prediction-range',
                                value=[years[-2], years[-1]]),
                html.H4('Filter'),
                html.Div(className="row", children=[
                    html.Div([
                        html.Label('Genre:'),
                        dcc.Dropdown(id='genre-prediction-dd',
                                     options=[{'label': column, 'value': column} for column in genres], value='All',
                                     multi=True),
                        html.Div(id="genre-prediction-dd-div")], style={"width": "20%"}),
                    html.Div([], style={"width": "6%"}),
                    html.Div([
                        html.Label('Publisher:'),
                        dcc.Dropdown(id='publisher-prediction-dd',
                                     options=[{'label': column, 'value': column} for column in publishers], value='All',
                                     multi=True),
                        html.Div(id="publisher-prediction-dd-div"),
                    ], style={"width": "20%"}),
                    html.Div([], style={"width": "6%"}),
                    html.Div([
                        html.Label('Platform:'),
                        dcc.Dropdown(id='platform-prediction-dd',
                                     options=[{'label': column, 'value': column} for column in platforms], value='All',
                                     multi=True),
                        html.Div(id="platform-prediction-dd-div"),
                    ], style={"width": "20%"}),
                    html.Div([], style={"width": "6%"}),
                ], style=dict(display='flex')),
                html.Br(),
                html.Div(className="row", children=[
                    html.Div([
                        html.Label('Group by:'),
                        dcc.Dropdown(options=[{'label': column, 'value': column} for column in [
                            defaultGroup, 'Publisher', 'Platform']], value=defaultGroup, id='group-prediction-dd'),
                        html.Div(id="group-prediction-dd-div"),
                    ], style={"width": "20%"}),
                    html.Div([], style={"width": "6%"}),
                    html.Div([
                        html.Label('Trend type:'),
                        dcc.Dropdown(options=[{'label': column, 'value': column}
                                              for column in ['ols', 'rolling', 'ewm', 'lowess', 'expanding']],
                                     value='ols', id='trend-prediction-dd'),
                        html.Div(id="trend-prediction-dd-div"),
                    ], style={"width": "20%"})
                ], style=dict(display='flex')),
                html.Br(),
            ]),
            dcc.Graph(id='prediction'),
        ])

@app.callback(
    dash.dependencies.Output('filter-dd', 'options'),
    [dash.dependencies.Input('group-dd', 'value')]
)
def set_filter_options(group_option):
    optionList = genres
    if group_option == 'Genre':
        optionList = genres
    if group_option == 'Platform':
        optionList = platforms
    if group_option == 'Publisher':
        optionList = publishers
    return [ {'label': i, 'value': i} for i in optionList]

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
    df_sales = grouped_df[(grouped_df['Year'] >= years_range[0]) & (grouped_df['Year'] <= years_range[-1])]

    if 'All' not in filter_list:
        df_sales = df_sales[df_sales[group_option].isin(filter_list)]
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
    pie_df = grouped_df[(grouped_df['Year'] >= years_range[0]) & (grouped_df['Year'] <= years_range[-1])]
    if 'All' not in genre_list:
        pie_df = pie_df[pie_df['Genre'].isin(genre_list)]
    if 'All' not in platform_list:
        pie_df = pie_df[pie_df['Platform'].isin(platform_list)]
    if 'All' not in publisher_list:
        pie_df = pie_df[pie_df['Publisher'].isin(publisher_list)]
    pie_df = pie_df.drop(columns=['Total'], axis=1).groupby(['Year'], as_index=False).\
        agg({'NA': 'sum', 'EU': 'sum', 'JP': 'sum', 'Other': 'sum'}).reset_index().sum()
    pie_df.drop(['Year', 'index'], inplace=True)
    names = pie_df.index.values.tolist()
    return px.pie(values=pie_df.values.tolist(), names=names, title='Sales per region')


def prepare_model(columns, regression_model):
    model = {}
    rgr_df = cleared_df.copy()
    columns_to_drop = [c for c in rgr_df.columns if c not in columns]
    columns_to_drop.remove('Total')
    rgr_df.drop(columns=columns_to_drop, inplace=True)

    for c in rgr_df.columns:
        if rgr_df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(rgr_df[c].values))
            rgr_df[c] = lbl.transform(list(rgr_df[c].values))
            model[rgr_df[c].name] = lbl.classes_.tolist()

    y = rgr_df.pop('Total')

    x_train, x_test, y_train, y_test = train_test_split(rgr_df, y, test_size=0.20, random_state=42)
    rm = LinearRegression()
    if regression_model == 'Linear':
        rm = LinearRegression()
    elif regression_model == 'Lasso':
        rm = Lasso(alpha=0.1)
    elif regression_model == 'Ridge':
        rm = Ridge(alpha=0.1)
    elif regression_model == 'ElasticNet':
        rm = ElasticNet(alpha=0.1)
    elif regression_model == 'PLS':
        rm = PLSRegression()
    rm.fit(x_train, y_train)
    model['model'] = rm
    model['df'] = rgr_df
    model['columns'] = rgr_df.columns.values.tolist()
    model['score'] = f"{rm.score(x_test, y_test) * 100}%"
    model['Total'] = y
    return model

@app.callback(
    dash.dependencies.Output('score', 'children'),
    [dash.dependencies.Input("columns-dd", "value"),
        dash.dependencies.Input("model-dd", "value")]
)
def update_score(columns, model):
    column_list = parseFilter(columns)
    if 'All' in column_list:
        column_list = originalColumnsNoGSales
    return prepare_model(column_list, model)['score']

def filter_df(in_df, column, map_list, source_list):
    in_df['Grouped'] =  in_df.apply (lambda row: map_list[int(row[column])], axis=1)
    in_df.drop(columns=[column], axis=1, inplace=True)
    in_df.rename(columns={"Grouped": column}, inplace=True)
    if 'All' not in source_list:
        in_df = in_df[in_df[column].isin(source_list)]
    return in_df

@app.callback(
    dash.dependencies.Output('prediction', 'figure'),
    [dash.dependencies.Input("columns-dd", "value"),
        dash.dependencies.Input("model-dd", "value"),
        dash.dependencies.Input("years-prediction-range", "value"),
        dash.dependencies.Input("genre-prediction-dd", "value"),
        dash.dependencies.Input("platform-prediction-dd", "value"),
        dash.dependencies.Input("publisher-prediction-dd", "value"),
        dash.dependencies.Input("trend-prediction-dd", "value"),
        dash.dependencies.Input("group-prediction-dd", "value"),
     ]
)
def update_prediction_graph(columns, model, years_range, genres_option, platforms_option, publishers_option, trend_type, group_option):
    column_list = parseFilter(columns)
    if 'All' in column_list:
        column_list = originalColumnsNoGSales
    for c in ['Year', 'Genre', 'Platform', 'Publisher']:
        if c not in column_list:
            column_list.append(c)
    model = prepare_model(columns, model)
    genre_list = parseFilter(genres_option)
    platform_list = parseFilter(platforms_option)
    publisher_list = parseFilter(publishers_option)

    print(f"Model: {columns}, years: {years_range}, genres: {genre_list}, platforms: {platform_list}, publishers: {publisher_list}")
    rgr_df = model['df']

    filtered_df = rgr_df[(rgr_df['Year'] > years_range[0]) & (rgr_df['Year'] <= years_range[-1])]
    new_df = filtered_df.copy()
    yearDiff = years_range[-1] - years_range[0]
    new_df['Year'] = new_df['Year'] + yearDiff

    y_pred = model['model'].predict(new_df)
    new_df['Total'] = y_pred
    rgr_df['Total'] = model['Total']

    combined_df = pd.concat([rgr_df, new_df], axis=0)

    combined_df = filter_df(combined_df, 'Genre', model['Genre'], genre_list)
    combined_df = filter_df(combined_df, 'Platform', model['Platform'], platform_list)
    combined_df = filter_df(combined_df, 'Publisher', model['Publisher'], publisher_list)

    combined_df = combined_df.groupby(['Year', group_option], as_index=False).agg({'Total': 'sum'}).reset_index()

    figure = px.scatter(combined_df, 'Year', 'Total', title=f"Year vs Total sales",
                        labels={"Year": "Year", 'Total': 'Total sales in mln', group_option: group_option},
                        trendline_options=get_trend_line_type_options(trend_type),
                        trendline=trend_type, color=group_option)

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
