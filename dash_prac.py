import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import datetime
from dash.dependencies import Input, Output
import psycopg2
import numpy as np


conn = psycopg2.connect("conect info")
df=pd.read_sql('select * from snow',conn)
hire = pd.read_sql('select count(distinct employee_nbr) as hires, extract(month from doh) as month from snow where extract(year from doh )=2017 group by extract(month from doh) order by extract(month from doh)',conn)
term = pd.read_sql('select count(distinct employee_nbr) as left, extract(month from dot) as month from snow where extract(year from dot )=2017 group by extract(month from dot) order by extract(month from dot)',conn)
ten = pd.read_sql('select count(distinct employee_nbr) as tas, extract(year from doh) as year, extract(year from now()) as curr from snow group by extract(year from doh) order by extract(year from doh)',conn)

df = df.drop(df[df['month'] == "Term"].index)
df['month2'] = df['month']
d = {'Jan ':1,'Feb ':2,'Mar ':3,'Apr ':4,'May ':5,'Jun ':6,'Jul ':7,'Aug ':8,'Sep ':9,'Oct ':10,'Nov ':11,'Dec ':12}
df['month'] = df['month'].map(d)

username_pass = [
    ['Username','password'], ['Username','password']
]


app = dash.Dash(__name__)

# Boostrap CSS.
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501
auth = dash_auth.BasicAuth(app,username_pass)
server = app.server
#can use html.hr for horizontal rule- horizontal line between parts
#app layout
app.layout = html.Div(
    [
        html.Div([

            html.Div(
                [
                        html.Img(
                            src="https://s3.us-east-2.amazonaws.com/hr-intellalytics/logo_hr.jpg",
                            className='two columns',
                            style={
                                'height': '100',
                                'width': '100%',
                                'float':'left',
                                'position':'relative',
                            },
                        ),
                        html.Div(
                            [
                                html.Button('Home',id='home',style={'float':'right','width':'100%'}),
                                html.Button('HR Metrics',id='metrics',style={'float':'right','width':'100%'}),
                                html.Button('HR Dashboard',id='dashboard',style={'float':'right','width':'100%'}),
                                html.Button('Contact Info',id='contact',style={'float':'right','width':'100%'})
                            ],
                            className='two columns',style={'width':'100%','float':'right','height':'800px'}
                        ),
                    ],
                    className='ten rows'
                ),
            ],
            className='two columns',style={'backgroundColor':'#63748e'}
        ),
        html.Div(
            [
        html.Div(
            [
                html.H1(
                    'HR Dashboard',
                    className='six columns',
                )
            ],
            className='row'
        ),
        html.Div(
            [
                html.P('Filter by Month:'), #noqa: E501
                dcc.RangeSlider(
                    id='month_slider',
                    marks={str(month): str(month) for month in df['month'].unique()},
                    min=df['month'].min(),
                    max=df['month'].max(),
                    value=[df['month'].unique().min(), df['month'].unique().max()]
                    #make so it is a range not first and last value in slider
                ),
            ],
            style={'margin-top':'0px', ' width':' 98%'}
        ),
        #first row w/ total compensation, average compensation etc
        html.Div(
            [
                html.Div(
                    [
                        html.Div(children='Total Compensation(YTD)',
                        style={'textAlign': 'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_total_comp',
                                style={'textAlign':'center'}
                                ),
                        ]),
                        html.Div(children='Average Age',
                        style={'textAlign': 'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_aver_age',
                                style={'textAlign':'center'}
                                ),
                        ]),
                    ], className='three columns',style={'width':'24%'}
                ),
                html.Div(
                    [
                        html.Div(children='Average Compensation',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_aver_comp',
                                style={'textAlign':'center'}
                                ),
                        ]),
                        html.Div(children='Absenteeism Rate',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_abs_rate',
                                style={'textAlign':'center'}
                                ),
                        ]),
                    ], className='three columns',style={'width':'24%'}
                ),

                html.Div(
                    [
                        html.Div(children='Turnover Rate',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_turn_rate',
                                style={'textAlign':'center'}
                                ),
                        ]),
                        html.Div(children='Hired',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_hired',
                                style={'textAlign':'center'}
                                ),
                        ]),
                    ], className='three columns',style={'width':'24%'}
                ),
                html.Div(
                    [
                        html.Div(children='1st Year Turnover Rate',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_1yr_turn',
                                style={'textAlign':'center'}
                                ),
                            ]),
                        html.Div(children='Left',
                        style={'textAlign':'center','fontSize':10}),
                        html.Div([
                            html.Div(id='display_left',
                                style={'textAlign':'center'}
                                ),
                            ]),

                    ], className='three columns',style={'width':'24%'}
                ),

            ],
            className='row',style={'marginTop': '30px'}
        ),
        #middle section w/ total hours & total compensation graphs
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='tot_hours',config={'displayModeBar':False})
                    ],
                    className='six columns',
                    style={'margin-top':'10','height':300}
                ),
                html.Div(
                    [
                        dcc.Graph(id='tot_comp',config={'displayModeBar':False})
                    ],
                    className='six columns',
                    style={'margin-top':'10','height':300}
                )
            ],
            className='row'
        ),
        #bottom section w/ tenure, status, and hired vs. left graphs
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='Emp_tenure',config={'displayModeBar':False})


                    ],
                    className='four columns',
                    style={'margin-top':'10','height':250}
                ),
                html.Div(
                    [
                        dcc.Graph(id='Emp_status',config={'displayModeBar':False})


                    ],
                    className='four columns',
                    style={'margin-top':'10','height':250}
                ),
                html.Div(
                    [
                        dcc.Graph(id='hire_vs_fire',config={'displayModeBar':False})
                    ],
                    className='four columns',
                    style={'margin-top':'10','height':250}
                )
            ],
            className='row'
        ),

    ],
    className='ten columns offset by one',style={'backgroundColor':'#F1F1F1'}),
    ],
    className='twelve columns'
)


#Helper function for month slider
def filter_df(df,month_slider):
    dff = df[df['month'].isin(month_slider)]
    return dff
'''
mydates = [datetime.date(year, month, 1) for year, month in product(range(1970, 2017), range(1, 13))]
terrorism['date'].between(mydates[date_range[0]], mydates[date_range[1]])
def filter_df(df,month_slider):
    dff = df[df['month'] > datetime.datetime.strftime(month_slider[0],'%b')
        & (df['month'] < datetime.datetime(month_slider[1],'%b'))]
    return dff
'''

#helper function for tenure dataset
def group_up(dl):
    if dl > 20: return '20+'
    elif 10 < dl <= 20: return '10-20'
    elif 5 < dl <= 10: return '5-10'
    elif 0 < dl <= 5: return '0-5'
    else: return 'None'

#Total compensation YTD
@app.callback(Output('display_total_comp','children'),
            [Input('month_slider','value')])
def make_totyr(month_slider):
    dff = filter_df(df,month_slider)
    val = round(dff['amount'].sum())
    return '${:,.2f}'.format(val)
#Average Compensation
@app.callback(Output('display_aver_comp','children'),
            [Input('month_slider','value')])
def make_avgcomp(month_slider):
    dff = filter_df(df,month_slider)
    av_comp = round(dff['amount'].mean())
    return '${:,.2f}'.format(av_comp)
#Turnover Rate
@app.callback(Output('display_turn_rate','children'),
            [Input('month_slider','value')])
def turnovr_rte(month_slider):
    dff = filter_df(df,month_slider)
    rat = round((len(dff['dot'].unique())/len(dff['employee_nbr'].unique()))*100,1)
    return "{}%".format(rat)
#1st Year Turnover Rate
@app.callback(Output('display_1yr_turn','children'),
            [Input('month_slider','value')])
def frstyr_turn_rte(month_slider):
    dff = filter_df(df,month_slider)
    first = round((len(dff['dot'].unique())/len(dff['employee_nbr'].unique()))*100,1)
    return "{}%".format(first)
#average age
@app.callback(Output('display_aver_age','children'),
            [Input('month_slider','value')])
def av_age(month_slider):
    dff=filter_df(df,month_slider)
    ag = round(dff['age'].mean())
    return ag

#Absenteeism Rate
@app.callback(Output('display_abs_rate','children'),
            [Input('month_slider','value')])
def abst_rate(month_slider):
    dff=filter_df(df,month_slider)
    val = len(dff.loc[dff['ed_code'].isin(['E02','E04','E03','E01','E05'])])
    va = len(dff.loc[dff['ed_code'].isin(['E20','E12','E11','E14','E13'])])
    vall = round(((val-va)/val)*100,1)
    return "{}%".format(vall)

#Hired
@app.callback(Output('display_hired','children'),
            [Input('month_slider','value')])
def hire_rate(month_slider):
    dff=filter_df(hire,month_slider)
    hir = dff.hires.sum()
    return hir
#Left
@app.callback(Output('display_left','children'),
            [Input('month_slider','value')])
def left_rate(month_slider):
    dff=filter_df(term,month_slider)
    lef = dff.left.sum()
    return lef

#slider -> headcount
'''
@app.callback(Output('Headcount','figure'),
            [Input('month_slider','value')])
def head_count(month_slider):
    dff = filter_df(df,month_slider)
    data = [
        go.Pie(
            values=[len(df['sex'][df['sex']=='M']),len(df['sex'][df['sex']=='F'])],
            labels=df['sex'].unique(),
            hole=.6,
            showlegend=False,
            name='Headcount',
            domain=dict(x=[0,1],y=[0,1]),
            hoverinfo='values'
        )
    ]
    layout = go.Layout(
        title='Headcount',
        height=150,
        font=dict(size=10)
    )
    figg = go.Figure(data=data,layout=layout)
    return figg
'''
#slider -> total hours chart
@app.callback(Output('tot_hours','figure'),
            [Input('tot_hours','clickData')])
def tot_hors(month_slider):
    data=[
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E02']),'hours'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Worked',
            text=round(df.loc[df['ed_code'].isin(['E02']),'hours'].groupby(df['month']).sum()),
            marker=dict(color='#1f77b4'),
            showlegend=False,
            hoverinfo='text+name'
        ),
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E04','E05']),'hours'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Time Over',
            text=round(df.loc[df['ed_code'].isin(['E04','E05']),'hours'].groupby(df['month']).sum()),
            marker=dict(color='#FF7F0E'),
            showlegend=False,
            hoverinfo='text+name'
        ),
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E20','E13']),'hours'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Bereavement & Paid Time off',
            text=round(df.loc[df['ed_code'].isin(['E20','E13']),'hours'].groupby(df['month']).sum()),
            marker=dict(color='#2CA02C'),
            showlegend=False,
            hoverinfo='text+name'
        ),
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E11','E14']),'hours'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Vacation & Holiday',
            text=round(df.loc[df['ed_code'].isin(['E11','E14']),'hours'].groupby(df['month']).sum()),
            marker=dict(color='#d62728'),
            showlegend=False,
            hoverinfo='text+name'
        ),
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E12']),'hours'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Sick',
            text=round(df.loc[df['ed_code'].isin(['E12']),'hours'].groupby(df['month']).sum()),
            marker=dict(color='#9467bd'),
            showlegend=False,
            hoverinfo='text+name'
        )

    ]
    layout= go.Layout(
        title='Total Hours',
        legend=dict(x=-.1,y=1.3,orientation="h"),
        hovermode='closest',
        barmode='stack',
        height=300,
        font=dict(size=10)
    )
    figure = {'data':data,'layout':layout}
    return figure
'''#click -> drill down
@app.callback(Output('tot_hours','figure'),
            [Input('tot_hours','clickData')])
def tot_hors(month_slider):
    data=[
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E02','E04','E03','E01','E05']),'hours'].groupby(df['month']).sum(),
            x=df[df['month2']=='Jan'],
            name='Worked',
            text=df.loc[df['ed_code'].isin(['E02','E04','E03','E01','E05']),'hours'].groupby(df['month']).sum(),
            marker=dict(color='#1f77b4'),
            showlegend=True
        ),
        go.Bar(
            y=df.loc[df['ed_code'].isin(['E20','E12','E11','E14','E13']),'hours'].groupby(df['month']).sum(),
            x=df[df['month2']=='Jan'],
            name='Sick',
            text=df.loc[df['ed_code'].isin(['E20','E12','E11','E14','E13']),'hours'].groupby(df['month']).sum(),
            marker=dict(color='#FF7F0E'),
            showlegend=True
        )

    ]
    layout= go.Layout(
        title='Total Hours',
        legend=dict(x=-.1,y=1.3,orientation="h"),
        hovermode='closest',
        barmode='stack',
        height=300,
        font=dict(size=10)
    )
    figure = {'data':data,'layout':layout}
    return figure
'''
#slider -> total compensation chart
@app.callback(Output('tot_comp','figure'),
            [Input('month_slider','value')])
def tot_hors(month_slider):
    data=[
        go.Bar(
            y=df['amount'].groupby(df['month']).sum(),
            x=df['month2'].unique(),
            name='Total Compensation',
            text=round(df['amount'].groupby(df['month']).sum()),
            marker=dict(color='#1f77b4'),
            showlegend=True,
            hoverinfo='text'
        ),
        go.Bar(
            y=df['amount'].groupby(df['month']).mean(),
            x=df['month2'].unique(),
            name='Average Compensation',
            text=round(df['amount'].groupby(df['month']).mean()),
            marker=dict(color='#FF7F0E'),
            showlegend=True,
            hoverinfo='text'
        )

    ]
    layout= go.Layout(
        title='Total Compensation',
        legend=dict(x=-.1,y=1.3,orientation="h"),
        font=dict(size=10),
        hovermode='closest',
        barmode='group',
        height=300
    )
    figure ={'data':data,'layout':layout}
    return figure

#slider -> employment status pie chart
@app.callback(Output('Emp_status','figure'),
            [Input('month_slider','value')])
def update_empstat(month_slider):
    dff= filter_df(df,month_slider)
    data = [
        go.Pie(
            values= [len(df[df['status']=='FT']),len(df[df['status']=='PT'])],
            labels= df['status'].unique(),
            hole=.6,
            showlegend=False,
            name='Employment Status',
            domain=dict(x=[0,1],y=[0,1]),
            hoverinfo='percent'

        )
    ]
    layout = go.Layout(
        title='Employment Status',
        hovermode='closest',
        height=250,
        font=dict(size=10)
    )
    fig= {'data':data,'layout':layout}
    return fig

#slider -> tenure pie chart
@app.callback(Output('Emp_tenure','figure'),
            [Input('month_slider','value')])
def update_empten(month_slider):
    dff= filter_df(df,month_slider)

    varr = ten.assign(years = ten['curr']-ten['year'])
    varr['bin'] = varr['years'].map(group_up)
    data = [
        go.Pie(
            values=[varr[varr['bin']=='20+']['tas'].sum(),varr[varr['bin']=='10-20']['tas'].sum(),varr[varr['bin']=='5-10']['tas'].sum(),
            varr[varr['bin']=='0-5']['tas'].sum(),varr[varr['bin']=='None']['tas'].sum()],
            labels=varr['bin'].unique(),
            hole=.6,
            showlegend=False,
            name='Employment Tenure',
            domain=dict(x=[0,1],y=[0,1]),
            hoverinfo='percent'
        )
    ]
    layout = go.Layout(
        title='Employment Tenure',
        hovermode='closest',
        height=250,
        font=dict(size=10)
    )
    figure= {'data':data,'layout':layout}
    return figure
# hired vs. left scatter plot
@app.callback(Output('hire_vs_fire','figure'),
            [Input('month_slider','value')])
            #fix this below function
def update_hileft(month_slider):
    data =[
        go.Bar(
            y=hire['hires'],
            x=df['month2'].unique(),
            name='Number Hired',
            text=hire['hires'],
            marker=dict(color='#1f77b4'),
            showlegend=True,
            hoverinfo='y'
        ),
        go.Bar(
            y=term['left'],
            x=df['month2'].unique(),
            name='Number Left',
            base=-term['left'],
            text=term['left'],
            marker=dict(color='#FF7F0E'),
            showlegend=True,
            hoverinfo='y'
        )
    ]
    layout= go.Layout(
        title='Left vs Hired',
        legend=dict(x=0,y=1.4,orientation="h"),
        hovermode='closest',
        barmode='stack',
        height=250,
        font=dict(size=10)
    )
    figure={'data':data,'layout':layout}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True,threaded=True)
#install psycopg2 -> pip install psycopg2-binary
