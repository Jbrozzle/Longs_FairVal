from bokeh.models import DataTable, ColumnDataSource, TableColumn, \
    HTMLTemplateFormatter, Patch, CustomJS, HoverTool
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd

def get_sma(prices, rate):
    return prices.rolling(rate).mean()

def get_bollinger_bands(prices, rate):
    sma = get_sma(prices, rate) # <-- Get SMA for 20 days
    std = prices.rolling(rate).std()
    upper = sma + std * 1.5 # Calculate top band
    lower = sma - std * 1.5 # Calculate bottom band
    return upper, lower

class BokehApp:

    def __init__(self, type, data, formatters=[], tools=[], tooltips = [], x_col='', y_col='', height=200, width=200):
        self.app_type = type
        self.src = data
        self.formatters = formatters
        self.tools = tools
        self.tooltips = tooltips
        self.x_col = x_col if x_col is not None else 0
        self.y_col = y_col if y_col is not None else 0
        self.width = width
        self.height = height

    def create_plot(self, data, subtype, fit_line, patch, fill_color):
        if fill_color is None:
            source = ColumnDataSource(data={'x_values': data[self.x_col],
                                        'y_values': data[self.y_col]})
        else:
            source = ColumnDataSource(data={'x_values': data[self.x_col],
                                        'y_values': data[self.y_col],
                                      'fill_color': data['fill_color']})
        plot = figure(width=self.width, height=self.height)
        if subtype == 'line':
            xvals = np.array(pd.to_datetime(data[self.x_col], format="%m/%d/%Y"))
            data.index = xvals
            data.sort_index(inplace=True)
            xvals = data.index
            source = ColumnDataSource(data={'x_values': xvals,
                                            'y_values': data[self.y_col]})
            plot = figure(width=self.width, height=self.height, x_axis_type='datetime')
            if patch:
                # do patch work here for standard deviations, make dynamic with slide widgets
                lookback = 25
                upper, lower = get_bollinger_bands(data[self.y_col], lookback)
                df = pd.DataFrame({'upper': upper, 'lower': lower, 'Date': xvals}).dropna()
                patch_y_vals = np.hstack((df['lower'], df['upper'][::-1]))
                patch_x_vals = np.hstack((df['Date'], df['Date'][::-1]))
                patch_source = ColumnDataSource(data={'x_values': patch_x_vals,
                                                      'y_values': patch_y_vals})
                glyph = Patch(x="x_values", y="y_values", fill_color="green", fill_alpha=0.5, line_width=0)
                plot.add_glyph(patch_source, glyph)
            plot.line(x='x_values', y='y_values', source=source, line_width=2)
            if fit_line:
                poly_coefs = np.polyfit(data[self.x_col], data[self.y_col], 3)
                poly_vals = np.polyval(poly_coefs, np.array(data[self.x_col]))
                plot.line(x=data[self.x_col], y=poly_vals)
        if subtype == "scatter":
            if fill_color is not None:
                plot.scatter(x='x_values', y='y_values', source=source, fill_color=fill_color)
                plot.xaxis.axis_label, plot.yaxis.axis_label = self.x_col, self.y_col
            else:
                plot.scatter(x='x_values', y='y_values', source=source)
                plot.xaxis.axis_label, plot.yaxis.axis_label = self.x_col, self.y_col
            if fit_line:
                poly_coefs = np.polyfit(data[self.x_col], data[self.y_col], 1)
                poly_vals = np.polyval(poly_coefs, np.array(data[self.x_col]))
                plot.line(x=data[self.x_col], y=poly_vals)
                patch_source = None
        if subtype == "vbar":
            plot.vbar(x='x_values', y='y_values', source=source)
            patch_source = None

        return plot, source, patch_source

    def create_widget(self, data, columns):
        data['Indexer'] = range(0, len(data))
        source = ColumnDataSource(data)
        if self.app_type == DataTable:
            data_table = DataTable(source=source, columns=columns, width=self.width, height=self.height,
                                   autosize_mode='none')

            return data_table, source

    def create_callback(self, data, widget):
        pass
