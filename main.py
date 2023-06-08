import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf


def prepare_data():
    sales_df = pd.read_csv('vgsales.csv')
    sales_df.drop(columns=[sales_df.columns[0], sales_df.columns[1]], axis=1, inplace=True)
    sales_df.dropna(axis=0, inplace=True)
    sales_df.replace(['PS2', 'PS3', 'PS4', 'PSV'], 'PS', inplace=True)
    sales_df.replace(['XB', 'XOne', 'X360'], 'Xbox', inplace=True)
    sales_df.replace(['Wii', 'N64', 'GC', 'SNES'], 'NES', inplace=True)
    sales_df.replace(['3DS', 'DS', 'GC', 'GBA'], 'GB', inplace=True)
#   df.replace(['SNES', 'N64', 'PS4', 'PSV'], 'PS', inplace=True)
    sales_df = sales_df.rename(
        columns={"NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP", "Other_Sales": "Other", "Global_Sales": "Total"},
        errors="raise")
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
df = prepare_data()
years = df['Year'].value_counts().index.astype('int64').tolist()
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
    dash.dash_table.DataTable(df.to_dict('records'),
                              [{"name": i, "id": i} for i in df.columns],
                              id='wineData',
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
    html.H2('Sales least significant factor analysis:'),
    html.Div(
        className="row", children=[
            html.Div([
                html.Label('Sales:'),
                dcc.Dropdown(options=salesOptions, value='Total', id='sales-independent-dd'),
                html.Div(id="sales-independent-dd-div")], style={"width": "20%"}),
            html.Div([], style={"width": "12.5%"}),
            html.Div([
                html.Label('Eliminate:'),
                dcc.Dropdown(options=[{'label': column, 'value': column} for column in ['Year', 'Genre', 'Platform', 'Publisher']], value='All', id='eliminate-dd', multi=True),
                html.Div(id="eliminate-dd-div"),
                ], style={"width": "25%"}),
            html.Div([], style={"width": "12.5%"}),
        ], style=dict(display='flex')),
    html.Br(),
    dcc.Markdown(dangerously_allow_html=True, id='model-fit')
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

@app.callback(
    dash.dependencies.Output('model-fit', 'children'),
    [dash.dependencies.Input("sales-independent-dd", "value"),
    dash.dependencies.Input("eliminate-dd", "value")]
)
def set_model_fit(independentColumn, eliminateColumns):
    dependentColumns = parseFilter(eliminateColumns)
    if 'All' in dependentColumns:
        dependentColumns = ['Year', 'Genre', 'Platform', 'Publisher']
    model = smf.ols(createFormula(independentColumn, dependentColumns), data=df).fit()
    #return model.summary().as_html()
    return model.summary().as_html()

def createFormula(independentColumn, dependentColumns):
    return f"{independentColumn} ~ {' + '.join(dependentColumns)}"

if __name__ == '__main__':
    app.run_server(debug=True)
