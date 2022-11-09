import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import ColorBar
from bokeh.plotting import figure, show, curdoc
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6
from bokeh.layouts import row
from bokeh_bits import *
import datetime

def main():

    fname = '/Users/Josh/Desktop/Longs_FairVal/raw_yield_data.csv'
    raw_df = pd.read_csv(fname)
    raw_df.set_index('Unnamed: 0', inplace=True)
    raw_df.index.rename('Date', inplace=True)
    raw_df.index = pd.to_datetime(raw_df.index)
    df = pd.DataFrame([], columns = ['10y_G', '30y_G', '1y1y', '2y1y', '3y2y', '5y5y'])
    df[df.columns[:6]] = raw_df.iloc[:, 0:6]
    y = df['30y_G'] - df['10y_G']
    x = df[['1y1y', '2y1y', '3y2y', '5y5y']]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    df = model.predict(x) - y
    new_df = pd.DataFrame([], columns = ['Pred'])
    new_df['Pred'] = df
    new_df['Date'] = df.index
    new_df.set_index('Date', inplace=True)
    new_df['Date'] = new_df.index

    plt_formatters = ['']
    plt_tools = ['']
    tooltips = ['']

    # Create residual chart with boll bands
    hist_chart = BokehApp(figure(), new_df,
                          plt_formatters, plt_tools,
                          x_col='Date', y_col='Pred', width=400, height=400)

    hist_chart_plot, hist_chart_ds, hist_patch_ds = hist_chart.create_plot(hist_chart.src, "line", fit_line=False,
                                                                           patch=True, fill_color=None)
    # Create scatter plot with colormap for date of datapoint
    scatt_df = new_df
    scatt_df['Actual'] = y
    scatt_df['Today'] = pd.to_datetime([datetime.date.today()] * len(scatt_df))
    scatt_df['Date'] = pd.to_datetime(scatt_df['Date'])
    scatt_df['fill_color'] = scatt_df['Today'] - scatt_df['Date']
    scatt_df['fill_color'] = scatt_df['fill_color'].astype(np.int64)/int(86*1e12)
    mapper = linear_cmap(field_name='fill_color', palette=Spectral6,
                         low=min(scatt_df['fill_color']), high=max(scatt_df['fill_color']))

    main_scatter = BokehApp(figure(), scatt_df, plt_formatters, plt_tools, tooltips,
                            x_col="Actual", y_col='Pred', width=400, height=400)
    scatter_plot, scatter_ds, scatter_patch_ds = main_scatter.create_plot(main_scatter.src, "scatter", fit_line=True,
                                                                          patch=False, fill_color=mapper)
    color_bar = ColorBar(color_mapper=mapper['transform'], width = 10)
    scatter_plot.add_layout(color_bar, 'right')

    # Plot in bokeh layout
    rw = row(scatter_plot, hist_chart_plot)
    curdoc().title = "Longs Fair Value Tool"
    curdoc().add_root(rw)
    show(rw)

main()
