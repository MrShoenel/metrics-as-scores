''' Present an interactive explorer for metrics as scores.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve scoring.py

at your command prompt. Then navigate to the URL

    http://localhost:5678/scoring

in your browser.

'''
import os
from sys import path


path.append(os.getcwd())

if 'BOKEH_VS_DEBUG' in os.environ and os.environ['BOKEH_VS_DEBUG'] == 'true':
    import ptvsd
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
from bokeh.models import ColumnDataSource, Dropdown, CheckboxGroup, Button, DataTable, TableColumn
from bokeh.models.widgets.inputs import Spinner
from bokeh.palettes import Category20_12
from bokeh.plotting import figure
from bokeh.models.widgets.markups import Div
from src.distribution.distribution import ECDF, DistTransform, KDECDF_approx
from src.tools.lazy import SelfResetLazy
from numbers import Integral
from gc import collect


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


dd_transf_items = list([(item.value, item.name) for item in list(DistTransform)])
dd_transf_items[0] = ('<No Transformation>', DistTransform.NONE.name)
selected_transf = dd_transf_items[0]
dd_transf = Dropdown(label=selected_transf[0], menu=dd_transf_items)


# Data and UI for cutting off distributions
cbg_cutoff_items = ['Cut off smoothed distributions beyond observed values']
selected_cutoff = False
cbg_cutoff = CheckboxGroup(labels=cbg_cutoff_items, active=[])


# Input for own metric's value
input_own = Spinner(mode='float', placeholder='Check Own Metric Value')

# Checkbox for automatic transform
cbg_autotrans_items = ['Apply transform using ideal value']
selected_autotransf = True
cbg_autotrans = CheckboxGroup(labels=cbg_autotrans_items, active=[0])



# Data and UI for selected CDF type
dd_denstype_items = [('PDF from KDE', 'PDF'), ('ECDF', 'ECDF'), ('Smoothed approx. ECDF from KDE', 'KDE_CDF_approx'), ('[Score] ECCDF', 'ECCDF'), ('[Score] Smoothed approx. ECCDF from KDE', 'KDE_CCDF_approx')]
selected_denstype = dd_denstype_items[4]
dd_denstype = Dropdown(label=selected_denstype[0], menu=dd_denstype_items)


# Contain button
btn_contain = Button(label='Contain Plot')



# Set up data
source = ColumnDataSource(data=pd.DataFrame(columns=list([f'x_{domain}' for domain in domains.keys()]) + list(domains.keys())))
# Set up plot
plot = figure(sizing_mode='stretch_width', height=640,# title="Metrics as Scores",
              x_axis_label='Metric Value', y_axis_label='Corresponding Score',
              tools="crosshair,hover,pan,wheel_zoom,xwheel_zoom,ywheel_zoom,reset", x_range=[0, 1], y_range=[0 - .02, 1.02], active_scroll='wheel_zoom')
#plot.toolbar.active_scroll = plot.select_one('wheel_zoom') #'wheel_zoom'


for idx, domain in enumerate(domains.keys()):
    plot.line(f'x_{domain}', domain, source=source, line_width=2, line_alpha=1., color=Category20_12[idx], legend_label=domains_labels[domain])

# Also add a vertical line for own metric
line_own_source = ColumnDataSource(data=pd.DataFrame(columns=['x', 'y']))
line_own = plot.line('x', 'y', source=line_own_source, line_alpha=1., color='black', line_width=1.5)
def update_own_line():
    val = input_own.value
    if (selected_transf[1] == 'NONE' or (selected_transf[1] != 'NONE' and not selected_autotransf)) and (isinstance(val, int) or isinstance(val, float) or isinstance(val, Integral)):
        line_own_source.data = dict(x=[val, val], y=[plot.y_range.start, plot.y_range.end])
    else:
        line_own_source.data = dict(x=[], y=[])



plot.legend.title = 'Domain'
plot.legend.location = 'top_right'
plot.legend.click_policy = 'hide'


# Table for transformation values
tbl_transf_src = ColumnDataSource(pd.DataFrame(columns=['domain', 'transf_value', 'metric_value', 'own_value']))
tbl_transf_cols = [
    TableColumn(field='domain', title='Domain'),
    TableColumn(field='transf_value', title='Used Transformation Value'),
    TableColumn(field='metric_value', title='Metric Value (not transformed)'),
    TableColumn(field='own_value', title='Corresponding Score')]
tbl_transf = DataTable(source=tbl_transf_src, columns=tbl_transf_cols, index_position=None, sizing_mode='stretch_both')


def update_transf():
    if selected_autotransf:
        if selected_transf[1] == 'NONE':
            tbl_transf_cols[2].title = 'Metric Value (no transformation chosen)'
        else:
            tbl_transf_cols[2].title = 'Metric Distance (transformed using ideal)'
    else:
        tbl_transf_cols[2].title = 'Metric Value'





cdfs_ECDF: dict[str, ECDF] = None
cdfs_KDECDF_approx: dict[str, KDECDF_approx] = None


cdfs: dict[str, SelfResetLazy[dict[str, Union[ECDF, KDECDF_approx]]]] = {}
clazzes = [ECDF, KDECDF_approx]
transfs = list(DistTransform)

def unpickle(file: str):
    try:
        with open(file=file, mode='rb') as f:
            return load(f)
    except Exception as e:
        raise Exception('This webapp relies on precomputed results. Please generate them using the file src/data/pregenerate.py before running this webapp.') from e

for clazz in clazzes:
    for transf in transfs:
        cdfs[f'{clazz.__name__}_{transf.name}'] = SelfResetLazy(reset_after=3600.0,
            fn_create_val=lambda clazz=clazz, transf=transf: unpickle(f'./results/cdfs_{clazz.__name__}_{transf.name}.pickle'))


def update_plot(contain_plot: bool=False):
    global selected_cutoff
    sd = selected_denstype[1]
    is_ecdf = 'ECDF' in sd or 'ECCDF' in sd
    is_ccdf = 'CCDF' in sd
    is_pdf = 'PDF' in sd

    # The E(C)CDF is always cut off.
    cbg_cutoff.disabled = is_ecdf
    if cbg_cutoff.disabled:
        # We also should uncheck it, since it's not possible.
        cbg_cutoff.active = []
        selected_cutoff = False

    densities: dict[str, Union[ECDF, KDECDF_approx]] = {}
    df_cols = {}
    lb, ub = 0, 0
    
    clazz = ECDF if is_ecdf else KDECDF_approx
    use_densities = cdfs[f'{clazz.__name__}_{selected_transf[1]}'].value

    for domain in domains.keys():
        density: Union[ECDF, KDECDF_approx] = use_densities[f'{domain}_{selected_score[1]}']
        densities[domain] = density

        prd = density.practical_domain

        prd = densities[domain].practical_domain
        if prd[0] < lb:
            lb = prd[0]
        if prd[1] > ub:
            ub = prd[1]

    if not is_ecdf and selected_cutoff:
        # This will cut off values falsely indicated by the smoothness.
        lb = max(lb, min(map(lambda _dens: _dens._range_data[0], densities.values())))
        ub = min(ub, max(map(lambda _dens: _dens._range_data[1], densities.values())))

    if contain_plot:
        ext = ub - lb
        plot.x_range.start = lb - ext * 1e-2
        plot.x_range.end = ub + ext * 1e-2

        if is_pdf:
            plot.y_range.end = max(map(lambda _dens: _dens.practical_range_pdf[1], densities.values()))
            plot.y_range.start = 0. - .02 * plot.y_range.end
            plot.y_range.end += .02 * plot.y_range.end
        else:
            plot.y_range.start = 0. - .02
            plot.y_range.end = 1. + .02
        return


    v = input_own.value
    has_own = isinstance(v, int) or isinstance(v, float) or isinstance(v, Integral)
    own_values = []
    input_own.low = lb
    input_own.high = ub
    input_own.step = (ub - lb) / 25.

    for domain in domains.keys():
        density = densities[domain]
        use_v = v
        if has_own and selected_autotransf and density.transform_value is not None:
            use_v = np.abs(density.transform_value - v)
        lb_domain = max(density._range_data[0], density.practical_domain[0]) if not is_ecdf and selected_cutoff else density.practical_domain[0]
        ub_domain = min(density._range_data[1], density.practical_domain[1]) if not is_ecdf and selected_cutoff else density.practical_domain[1]
        domain_x = np.linspace(lb_domain, ub_domain, 300 if is_pdf else 600) # PDF is slower
        df_cols[f'x_{domain}'] = domain_x
        if is_pdf:
            df_cols[domain] = density.pdf(domain_x)
            if has_own:
                own_values.append(density.pdf(use_v))
        else:
            df_cols[domain] = density.cdf(domain_x)
            if is_ccdf:
                df_cols[domain] = 1. - df_cols[domain]
            if has_own:
                if is_ccdf:
                    own_values.append(1. - density.cdf(use_v))
                else:
                    own_values.append(density.cdf(use_v))

    source.data = pd.DataFrame(df_cols)


    def tbl_format(v: float=None):
        if v is None:
            return '<none>'
        
        if v >= 0. and v <= 1e-300:
            return 0.
        exp = np.floor(np.log10(np.abs(v)))
        if exp < 0.:
            if exp >= -4.0:
                return np.round(v, 4)
            return np.round(v, int(np.ceil(2. + np.abs(exp))))
        if exp > 4.0:
            return np.round(v)
        return np.round(v, 4)
        
    
    tbl_transf_src.data = {
        'domain': list(map(lambda domain: domains_labels[domain], domains.keys())),
        'transf_value': list(map(lambda domain: tbl_format(densities[domain].transform_value), domains.keys())),
        'metric_value': list(map(lambda domain: tbl_format(np.abs(densities[domain].transform_value - v) if has_own and selected_autotransf and densities[domain].transform_value is not None else v), domains.keys())),
        'own_value': list([tbl_format(v=v) for v in own_values]) if has_own else list([tbl_format(v=None) for _ in range(len(domains.keys()))])
    }

    update_own_line()
    update_transf()




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


def cbg_autotrans_click(active: list[int]):
    global selected_autotransf
    selected_autotransf = len(active) > 0
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
    
    if 'PDF' in selected_denstype[1]:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Relative Likelihood'
    elif 'CCDF' in selected_denstype[1]:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Corresponding Score'
    else:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Cumulative Probability'


def dd_transf_click(evt: MenuItemClick):
    global selected_transf
    transf_before = selected_transf[1]
    transf_after = evt.item
    temp = { key: val for (val, key) in dd_transf_items }
    dd_transf.label = temp[evt.item]
    selected_transf = (temp[evt.item], evt.item)
    update_plot()
    if transf_before != transf_after:
        update_plot(contain_plot=True)
    if selected_transf[1] == DistTransform.NONE.name:
        plot.xaxis.axis_label = 'Metric Value'
    else:
        plot.xaxis.axis_label = 'Metric Distance from Ideal'


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


def input_own_change(attr, old, new):
    update_plot()


def read_text(file: str) -> str:
    with open(file=file, mode='r') as f:
        return f.read().strip()

cbg_cutoff.on_click(cbg_cutoff_click)
cbg_autotrans.on_click(cbg_autotrans_click)
dd_scores.on_click(dd_scores_click)
dd_transf.on_click(dd_transf_click)
dd_denstype.on_click(dd_denstype_click)
btn_contain.on_click(btn_contain_click)
input_own.on_change('value', input_own_change)


header = Div(text=read_text('./src/webapp/header.html'))
footer = Div(text=read_text('./src/webapp/footer.html'))



input_row1 = row([
    dd_scores, dd_denstype
    #column(Div(text='Metric:'), dd_scores),
    #column(Div(text='Transformation:'), dd_transf)
])
input_row2 = row([
    dd_transf, input_own, cbg_autotrans
    #column(Div(text='Distribution Type:'), dd_denstype),
    #column(Div(text='Check Own Value (press Enter):'), input_own, cbg_autotrans)
])
input_row3 = row([
    btn_contain, cbg_cutoff
    #column(Div(text='Plot Controls:'), btn_contain),
    #column(Div(text=''), cbg_cutoff)
])

plot_row = row([column(tbl_transf, plot)], sizing_mode='stretch_both')
root_col = column(header, input_row1, input_row2, input_row3, plot_row, footer, sizing_mode='stretch_both')

curdoc().add_root(root_col)
curdoc().title = "Metrics As Scores"

update_plot()
update_plot(contain_plot=True)
