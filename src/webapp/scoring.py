''' Present an interactive explorer for metrics as scores.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve scoring.py

at your command prompt. Then navigate to the URL

    http://localhost:5678/scoring

in your browser.

'''
import os
import ptvsd
from sys import path
from src.data.metrics import MetricID


path.append(os.getcwd())

if os.environ['BOKEH_VS_DEBUG'] == 'true':
    # 5678 is the default attach port in the VS Code debug configurations
    print('Waiting for debugger attach')
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()





import pandas as pd
import numpy as np

from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Dropdown
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]



# Data and UI control for the systems' domains:
systems_domains_df = pd.read_csv('./files/systems-domains.csv')
systems_domains = dict(zip(systems_domains_df.System, systems_domains_df.Domain))
systems_qc_names = dict(zip(systems_domains_df.System, systems_domains_df.System_QC_name))
temp = systems_domains_df.groupby('Domain').count()
domains = dict(zip(temp.index.to_list(), temp['System'].to_list()))
dd_domain_items = [(f'[All system types] ({temp.System.sum()})', '__ALL__')] + list([(cap(f'{item[0]} ({item[1]})'), item[0]) for item in domains.items()])
selected_domain = dd_domain_items[0]
dd_domain = Dropdown(label=selected_domain[0], menu=dd_domain_items)



# Data and UI for the metric/score types
dd_scores_items = list([(f'[{item.name}] {item.value}', item.name) for item in list(MetricID)])
selected_score = dd_scores_items[0]
dd_scores = Dropdown(label=selected_score[0], menu=dd_scores_items)




# Set up data
source = ColumnDataSource(data=dict(x=[0], y=[0]))
# Set up plot
plot = figure(height=400, width=400, title="my sine wave",
              tools="crosshair", x_range=[0, 1], y_range=[0, 1])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


from src.distribution.distribution import Distribution, ECDF, KDECDF_approx
d: Distribution = None


def update_plot():
    global d, domains, selected_score, selected_domain, systems_domains, systems_qc_names
    if d is None:
        d = Distribution(df=pd.read_csv('csv/metrics.csv'))
    
    metric_id = MetricID[selected_score[1]]
    systems = None
    if selected_domain[1] != '__ALL__':
        # Gather all systems with the selected domain.
        temp = filter(lambda di: di[1] == selected_domain[1], systems_domains.items())
        # Map the names to the Qualitas compiled corpus:
        systems = list(map(lambda di: systems_qc_names[di[0]], temp))

    data = d.get_cdf_data(metric_id=metric_id, unique_vals=False, systems=systems)
    cdf = KDECDF_approx(data=data)
    #cdf = ECDF(data=data)
    pr = cdf.practical_range

    # Generate the new curve
    x = np.linspace(pr[0], pr[1], int(1e3))
    y = 1. - cdf(x)

    ext = pr[1] - pr[0]
    plot.x_range.start = pr[0] - ext * 1e-2
    plot.x_range.end = pr[1] + ext * 1e-2

    #plot.x_range = Range1d(start=pr[0], end=pr[1]) # Range() #[pr[0], pr[1]]
    source.data = dict(x=x, y=y)



def dd_system_click(evt: MenuItemClick):
    global selected_domain
    temp = { key: val for (val, key) in dd_domain_items }
    dd_domain.label = temp[evt.item]
    selected_domain = (temp[evt.item], evt.item)
    update_plot()


def dd_scores_click(evt: MenuItemClick):
    global selected_score
    temp = { key: val for (val, key) in dd_scores_items }
    dd_scores.label = temp[evt.item]
    selected_score = (temp[evt.item], evt.item)
    update_plot()


dd_domain.on_click(dd_system_click)
dd_scores.on_click(dd_scores_click)




# Set up layouts and add to document
inputs = column(dd_domain, dd_scores)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Metrics as Scores"
