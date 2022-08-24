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

path.append(os.getcwd())

if os.environ['BOKEH_VS_DEBUG'] == 'true':
    # 5678 is the default attach port in the VS Code debug configurations
    print('Waiting for debugger attach')
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()





import pandas as pd
import numpy as np
from typing import Union
from src.data.metrics import MetricID
from collections import OrderedDict
from pickle import load
from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Dropdown, CheckboxGroup, Button
from bokeh.palettes import Category20_12
from bokeh.plotting import figure
from bokeh.models.widgets.markups import Div
from src.distribution.distribution import ECDF, KDECDF_approx


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]



# Data and UI control for the systems' domains:
systems_domains_df = pd.read_csv('./files/systems-domains.csv')
systems_domains = dict(zip(systems_domains_df.System, systems_domains_df.Domain))
systems_qc_names = dict(zip(systems_domains_df.System, systems_domains_df.System_QC_name))
temp = systems_domains_df.groupby('Domain').count()
domains = OrderedDict(zip(temp.index.to_list(), temp['System'].to_list()))
domains_labels = OrderedDict({ key: f'{cap(key)} ({val})' for (key, val) in domains.items() })
domains['__ALL__'] = int(temp.System.sum())
domains_labels['__ALL__'] = '[All domain types combined]'



# Data and UI for the metric/score types
dd_scores_items = list([(f'[{item.name}] {item.value}', item.name) for item in list(MetricID)])
selected_score = dd_scores_items[13] # LCOM
dd_scores = Dropdown(label=selected_score[0], menu=dd_scores_items)


# Data and UI for cutting off distributions
cbg_cutoff_items = ['Cut off smoothed distributions beyond actual values']
selected_cutoff = False
cbg_cutoff = CheckboxGroup(labels=cbg_cutoff_items, active=[])



# Data and UI for selected CDF type
dd_denstype_items = [('PDF from KDE', 'PDF'), ('ECDF', 'ECDF'), ('Smoothed approx. ECDF from KDE', 'KDE_CDF_approx'), ('[Score] ECCDF', 'ECCDF'), ('[Score] Smoothed approx. ECCDF from KDE', 'KDE_CCDF_approx')]
selected_denstype = dd_denstype_items[4]
dd_denstype = Dropdown(label=selected_denstype[0], menu=dd_denstype_items)


# Contain button
btn_contain = Button(label='Contain plot')




# Set up data
#source = ColumnDataSource(data=dict(x=[0], y=[0]))
source = ColumnDataSource(data=pd.DataFrame(columns=list([f'x_{domain}' for domain in domains.keys()]) + list(domains.keys())))
# Set up plot
plot = figure(sizing_mode = 'stretch_width', height=850, title="Metrics as Scores",
              x_axis_label='Metric Value', y_axis_label='Corresponding Score',
              tools="crosshair,hover,pan,wheel_zoom,xwheel_zoom,ywheel_zoom,reset", x_range=[0, 1], y_range=[0 - .02, 1.02], active_scroll='wheel_zoom')
#plot.toolbar.active_scroll = plot.select_one('wheel_zoom') #'wheel_zoom'


for idx, domain in enumerate(domains.keys()):
    plot.line(f'x_{domain}', domain, source=source, line_width=2, line_alpha=1., color=Category20_12[idx], legend_label=domains_labels[domain])

plot.legend.location = 'top_right'
plot.legend.click_policy = 'hide'





cdfs_ECDF: dict[str, ECDF] = None
cdfs_KDECDF_approx: dict[str, KDECDF_approx] = None

try:
    with open('./results/cdfs_ECDF.pickle', 'rb') as f:
        cdfs_ECDF = load(f)
    with open('./results/cdfs_KDECDF_approx.pickle', 'rb') as f:
        cdfs_KDECDF_approx = load(f)
except Exception as e:
    raise Exception('This webapp relies on precomputed results. Please generate them using the file src/data/pregenerate.py before running this webapp.') from e



def update_plot(contain_plot: bool=False):
    sd = selected_denstype[1]
    is_ecdf = 'ECDF' in sd or 'ECCDF' in sd
    is_ccdf = 'CCDF' in sd
    is_pdf = 'PDF' in sd

    # The E(C)CDF is always cut off.
    cbg_cutoff.disabled = is_ecdf

    densities = {}
    df_cols = { }
    lb, ub = 0, 0
    
    use_densities = cdfs_ECDF if is_ecdf else cdfs_KDECDF_approx

    for domain in domains.keys():
        density: Union[ECDF, KDECDF_approx] = use_densities[f'{domain}_{selected_score[1]}']
        densities[domain] = density

        pr = density.practical_range

        pr = densities[domain].practical_range
        if pr[0] < lb:
            lb = pr[0]
        if pr[1] > ub:
            ub = pr[1]

    if not is_ecdf and selected_cutoff:
        # This will cut off values falsely indicated by the smoothness.
        lb = min(map(lambda _dens: _dens._range_data[0], densities.values()))
        #if is_pdf:
        #    lb = min(map(lambda _dens: _dens._range_data[0], densities.values()))
        #else:
        #    lb = min(map(lambda _dens: _dens.range[0], densities.values()))

    if contain_plot:
        ext = ub - lb
        plot.x_range.start = lb - ext * 1e-2
        plot.x_range.end = ub + ext * 1e-2

        if is_pdf:
            plot.y_range.end = max(map(lambda _dens: _dens.practical_range_pdf[1], densities.values()))
            plot.y_range.start = 0. - .02 * plot.y_range.end
        else:
            plot.y_range.start = 0. - .02
            plot.y_range.end = 1. + .02
        return


    for domain in domains.keys():
        density = densities[domain]
        #ext_domain = density.practical_range[1] - density.practical_range[0]
        #lb_domain = density.practical_range[0]# - ext_domain * 0.03
        lb_domain = density._range_data[0] if not is_ecdf and selected_cutoff else density.practical_range[0]
        #ub_domain = density.practical_range[1]# + ext_domain * 0.03
        ub_domain = density._range_data[1] if not is_ecdf and selected_cutoff else density.practical_range[1]
        domain_x = np.linspace(lb_domain, ub_domain, 250 if is_pdf else 500) # PDF is slower
        df_cols[f'x_{domain}'] = domain_x
        if is_pdf:
            df_cols[domain] = densities[domain].pdf(domain_x)
        else:
            df_cols[domain] = densities[domain].cdf(domain_x)
            if is_ccdf:
                df_cols[domain] = 1. - df_cols[domain]

    source.data = pd.DataFrame(df_cols)




def get_score_type(val: str) -> str:
    if 'PDF' in val:
        return 'PDF'
    if 'CCDF' in val:
        return 'CCDF'
    return 'CDF'


def cbg_cutoff_click(active: list[int]):
    global selected_cutoff
    selected_cutoff = len(active) > 0
    update_plot()


def dd_denstype_click(evt: MenuItemClick):
    global selected_denstype
    type_before = get_score_type(selected_denstype[1])
    type_after = get_score_type(evt.item)
    temp = { key: val for (val, key) in dd_denstype_items }
    dd_denstype.label = temp[evt.item]
    selected_denstype = (temp[evt.item], evt.item)
    update_plot()
    if type_before != type_after:
        update_plot(contain_plot=True)


def dd_scores_click(evt: MenuItemClick):
    global selected_score
    score_before = selected_score[1]
    score_after = evt.item
    temp = { key: val for (val, key) in dd_scores_items }
    dd_scores.label = temp[evt.item]
    selected_score = (temp[evt.item], evt.item)
    update_plot()
    if score_before != score_after:
        update_plot(contain_plot=True)


def btn_contain_click(*args):
    update_plot(contain_plot=True)


def read_text(file: str) -> str:
    with open(file=file, mode='r') as f:
        return f.read().strip()

cbg_cutoff.on_click(cbg_cutoff_click)
dd_scores.on_click(dd_scores_click)
dd_denstype.on_click(dd_denstype_click)
btn_contain.on_click(btn_contain_click)


header = Div(text=read_text('./src/webapp/header.html'))
footer = Div(text=read_text('./src/webapp/footer.html'))


input_row = row([dd_scores, dd_denstype])
input_row2 = row([btn_contain, cbg_cutoff])
plot_row = row([plot], sizing_mode='stretch_both')
root_col = column(header, input_row, input_row2, plot_row, footer, sizing_mode='stretch_both')

curdoc().add_root(root_col)
curdoc().title = "Metrics as Scores"

update_plot()
update_plot(contain_plot=True)
