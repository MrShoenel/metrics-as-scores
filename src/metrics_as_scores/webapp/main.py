"""
This is the Web Application of Metrics As Scores. It should be run by
using the command line interface and selecting a dataset. It is an
interactive explorer for the selected dataset.

During development, this may also be run using the following commands:

    ``bokeh serve src/metrics_as_scores/webapp/ --port 5678 --allow-websocket-origin * --args dataset=qcc``

This command should be run from the project's root. It will start a
web server on port `5678` using the dataset `qcc` (use the ID of the
dataset, it is shown by listing all locally available datasets in the CLI).
"""
from os import environ

if 'BOKEH_VS_DEBUG' in environ and environ['BOKEH_VS_DEBUG'] == 'true':
    import ptvsd
    # 5678 is the default attach port in the VS Code debug configurations
    print('Waiting for debugger attach')
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from metrics_as_scores.webapp.exception import PlotException
from metrics_as_scores.tools.funcs import nonlinspace, natsort
from metrics_as_scores.distribution.distribution import Empirical, DistTransform, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete, Dataset, LocalDataset
from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Dropdown, CheckboxGroup, Button, DataTable, TableColumn
from bokeh.models.widgets.inputs import Spinner
from bokeh.palettes import Category20_20
from bokeh.plotting import figure
from bokeh.models.widgets.markups import Div
from numbers import Integral


# Import the data; Note this is process-static!
from metrics_as_scores.webapp import data
ds: Dataset = data.ds
if ds is None:
    dummy_manifest: LocalDataset = dict(
        name = '', desc = '', author = '',
        qtypes = dict(QTY1 = 'continuous'), contexts = [],
        desc_qtypes = dict(QTY1 = ''))
    ds = Dataset(ds=dummy_manifest, df=pd.DataFrame())
cdfs = data.cdfs
contexts = list(ds.contexts(include_all_contexts=False))
contexts.sort(key=natsort)
contexts = contexts + ['__ALL__']

this_dir = Path(__file__).resolve().parent
dataset_dir: Path = data.dataset_dir
if dataset_dir is None:
    dataset_dir = Path(__file__)
web_dir = dataset_dir.joinpath('./web')


def cap(s: str) -> str:
    """ :meta private: """
    return s[:1].upper() + s[1:]



# Data and UI for the qtype/score types
dd_scores_items: list[tuple[str, str]] = []
if len(ds.quantity_types_discrete) > 0:
    qtypes_discrete = ds.quantity_types_discrete
    qtypes_discrete.sort(key=natsort)
    dd_scores_items.append(('------------ Discrete (integral) Quantities', '-'))
    dd_scores_items += list([(f'[{qtype}] {ds.qytpe_desc(qtype=qtype)}', qtype) for qtype in qtypes_discrete])
if len(ds.quantity_types_continuous) > 0:
    if len(ds.quantity_types_discrete) > 0:
        dd_scores_items.append((' ', '-'))
    qtypes_continuous = ds.quantity_types_continuous
    qtypes_continuous.sort(key=natsort)
    dd_scores_items.append(('------------ Continuous (real) Quantities', '-'))
    dd_scores_items += list([(f'[{qtype}] {ds.qytpe_desc(qtype=qtype)}', qtype) for qtype in qtypes_continuous])

selected_score = dd_scores_items[1]
"""
:meta private:
"""
dd_scores = Dropdown(label=selected_score[0], menu=dd_scores_items)


dd_transf_items = list([(item.value, item.name) for item in list(DistTransform)])
dd_transf_items[0] = ('<No Transformation>', DistTransform.NONE.name)
selected_transf = dd_transf_items[0]
dd_transf = Dropdown(label=selected_transf[0], menu=dd_transf_items)


# Data and UI for cutting off distributions
cbg_cutoff_items = ['Cut off smoothed distributions beyond observed values']
selected_cutoff = False
cbg_cutoff = CheckboxGroup(labels=cbg_cutoff_items, active=[])


# Input for own qtype's value
input_own = Spinner(mode='float', placeholder='Check Own Value', min_width=350)

# Checkbox for automatic transform
cbg_autotrans_items = ['Apply transform using ideal value']
selected_autotransf = True
cbg_autotrans = CheckboxGroup(labels=cbg_autotrans_items, active=[0])


# Contain button
btn_contain = Button(label='Contain Plot')
# Toggle Legend button
btn_toggle_legend = Button(label='Toggle Legend')



# Set up data
source = ColumnDataSource(data=pd.DataFrame(columns=list([f'x_{ctx}' for ctx in contexts]) + contexts))
# Set up plot
plot = figure(sizing_mode='stretch_width', height=640,
              x_axis_label='Quantity Value', y_axis_label='Corresponding Score',
              tools="box_zoom,crosshair,hover,pan,wheel_zoom,xwheel_zoom,ywheel_zoom,reset", x_range=[0, 1], y_range=[0 - .02, 1.02], active_scroll='wheel_zoom')
#plot.toolbar.active_scroll = plot.select_one('wheel_zoom') #'wheel_zoom'



for idx, ctx in enumerate(contexts):
    plot.line(f'x_{ctx}', ctx, source=source, line_width=2, line_alpha=1., color=Category20_20[idx], legend_label='[All groups combined]' if ctx == '__ALL__' else ctx)

# Also add a vertical line for own quantity
line_own_source = ColumnDataSource(data=pd.DataFrame(columns=['x', 'y']))
line_own = plot.line('x', 'y', source=line_own_source, line_alpha=1., color='black', line_width=1.5)
def update_own_line():
    """ :meta private: """
    val = input_own.value
    if (selected_transf[1] == 'NONE' or (selected_transf[1] != 'NONE' and not selected_autotransf)) and (isinstance(val, int) or isinstance(val, float) or isinstance(val, Integral)):
        line_own_source.data = dict(x=[val, val], y=[plot.y_range.start, plot.y_range.end])
    else:
        line_own_source.data = dict(x=[], y=[])



plot.legend.title = f'Group [{ds.ds["colname_context"]}]'
plot.legend.location = 'top_right'
plot.legend.click_policy = 'hide'


# Table for transformation values
tbl_transf_cols = [
    TableColumn(field='context', title=f'Group [{ds.ds["colname_context"]}]'),
    TableColumn(field='transf_value', title='Used Transformation Value'),
    TableColumn(field='qtype_value', title='Quantity\'s Value (not transformed)'),
    TableColumn(field='own_value', title='Corresponding Score'),
    TableColumn(field='pdist', title='Parametric Distrib.'),
    TableColumn(field='pdist_dval', title='Statistic')]
tbl_transf_src = ColumnDataSource(pd.DataFrame(columns=list([tc.field for tc in tbl_transf_cols])))
tbl_transf = DataTable(source=tbl_transf_src, columns=tbl_transf_cols, index_position=None, sizing_mode='stretch_width', min_height=30*len(contexts), height_policy='min')



def update_discrete_cont_mismatch():
    """ :meta private: """
    discrete = ds.is_qtype_discrete(qtype=selected_score[1])
    sd = selected_denstype[1]
    if not discrete and 'discrete' in sd:
        raise PlotException(msg='There are no discrete fits for continuous quantities.')


def update_labels():
    """ :meta private: """
    global selected_denstype, selected_transf, selected_autotransf
    sd = selected_denstype[1]
    st = selected_transf[1]
    is_ppf = 'PPF' in sd


    if selected_autotransf:
        if st == 'NONE':
            tbl_transf_cols[2].title = 'Value (no transformation chosen)'
        else:
            tbl_transf_cols[2].title = 'Distance (transformed using ideal)'
    else:
        tbl_transf_cols[2].title = 'Probability' if is_ppf else 'Value'


    if 'PDF' in sd:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Relative Likelihood'
    elif 'PMF' in sd:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Probability Mass'
    elif 'CCDF' in sd:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Corresponding Score'
    elif is_ppf:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Value of Random Variable'
    else:
        plot.yaxis.axis_label = tbl_transf_cols[3].title = 'Cumulative Probability'


    if is_ppf:
        input_own.placeholder = 'Check Probability To Sample A Value'
        plot.xaxis.axis_label = 'Probability'
    else:
        input_own.placeholder = 'Check Own Value'

        if st == DistTransform.NONE.name:
            plot.xaxis.axis_label = 'Value'
        else:
            plot.xaxis.axis_label = 'Distance from Ideal'



# Data and UI for selected CDF type
dd_denstype_items = [
    ('[Rel. Lik.] PDF from Parametric', 'Param_PDF'),
    ('[Rel. Lik.] PDF from KDE (EPDF)', 'KDE_PDF'),
    ('[Prob. Mass] PMF from Parametric (discrete)', 'Param_PMF_discrete'),
    ('[Prob. Mass] PMF from Empirical (EPMF, discrete)', 'EPMF_discrete'),
    ('------------', '-'),
    ('[Cum. Prob.] CDF from Parametric', 'Param_CDF'),
    ('[Cum. Prob.] CDF from Parametric (discrete)', 'Param_CDF_discrete'),
    ('[Cum. Prob.] CDF from Empirical (ECDF)', 'ECDF'),
    ('[Cum. Prob.] Smoothed approx. ECDF from KDE', 'KDE_CDF'),
    ('------------', '-'),
    ('[Score] CCDF from Parametric', 'Param_CCDF'),
    ('[Score] CCDF from Parametric (discrete)', 'Param_CCDF_discrete'),
    ('[Score] CCDF from Empirical (ECCDF)', 'ECCDF'),
    ('[Score] Smoothed approx. ECCDF from KDE', 'KDE_CCDF'),
    ('------------', '-'),
    ('[Quantiles] PPF from Parametric', 'Param_PPF'),
    ('[Quantiles] PPF from Parametric (discrete)', 'Param_PPF_discrete'),
    ('[Quantiles] PPF from Empirical (EPPF, discrete)', 'EPPF_discrete'),
    ('[Quantiles] Smoothed approx. inverse ECCDF from KDE', 'KDE_PPF')
]
selected_denstype = dd_denstype_items[1]
dd_denstype = Dropdown(label=selected_denstype[0], menu=dd_denstype_items, min_width=350, sizing_mode='stretch_width')


throbber = Div(text='<span class="flower-12l-24x24"></span>')
throbber.visible = False
status = Div(text='Ready.', sizing_mode='stretch_width')

def update_plot(contain_plot: bool=False):
    """ :meta private: """
    throbber.visible = True
    status.text = 'Loading ...'
    status.css_classes = []
    def temp():
        try:
            update_plot_internal(contain_plot=contain_plot)
            status.text = 'Ready.'
        except PlotException as pex:
            status.css_classes = ['error']
            status.text = pex.msg
        except Exception as e:
            status.text = getattr(e, 'message', repr(e))
            status.css_classes = ['error']
        finally:
            throbber.visible = False

    curdoc().add_next_tick_callback(temp)


def update_plot_internal(contain_plot: bool=False):
    """ :meta private: """
    global selected_cutoff, selected_autotransf
    sd = selected_denstype[1]
    is_empirical = sd.startswith('E')
    is_ecdf = 'ECDF' in sd or 'ECCDF' in sd
    is_ccdf = 'CCDF' in sd
    is_pdf = 'PDF' in sd or 'PMF' in sd # In either case, we'll call .pdf(..)
    is_ppf = 'PPF' in sd
    is_parametric = 'Param' in sd
    is_discrete = 'discrete' in sd
    is_kde = 'KDE' in sd

    # The E(C)CDF and EPMF are already cut off.
    cbg_cutoff.disabled = is_empirical
    if cbg_cutoff.disabled:
        # We also should uncheck it, since it's not possible.
        cbg_cutoff.active = []
        selected_cutoff = False
    
    if is_ppf:
        cbg_autotrans.disabled = True
        cbg_autotrans.active = []
        selected_autotransf = False
    else:
        cbg_autotrans.disabled = False

    densities: dict[str, Union[Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete]] = {}
    df_cols = {}

    clazz = None
    if is_empirical:
        if is_ecdf:
            clazz = Empirical
        elif is_discrete:
            clazz = Empirical_discrete
    elif is_parametric:
        if is_discrete:
            clazz = Parametric_discrete
        else:
            clazz = Parametric
    else:
        clazz = KDE_approx
    use_densities = cdfs[f'{clazz.__name__}_{selected_transf[1]}'].value

    # lb is corrected downwards, ub upwards
    lb, ub = 1e308, -1e308
    for ctx in contexts:
        density: Union[Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete] = use_densities[f'{ctx}_{selected_score[1]}']
        densities[ctx] = density
        if (is_parametric or (is_empirical and is_discrete)) and not density.is_fit:
            continue

        prd = density.practical_domain
        if prd[0] < lb:
            lb = prd[0]
        if prd[1] > ub:
            ub = prd[1]
    # Non-existing fits (e.g., discrete for continuous quantity) will not be
    # able to correctly move lb, ub.
    lb, ub = (max(-1e154, lb if lb < ub else ub), min(1e154, ub if ub > lb else lb))

    def range_data(d: Union[KDE_approx, Parametric]) -> tuple[float, float]:
        if is_parametric:
            return d.range
        return d._range_data

    if not is_empirical and selected_cutoff:
        # This will cut off values falsely indicated by the smoothness.
        lb = max(lb, min(map(lambda _dens: range_data(_dens)[0], densities.values())))
        ub = min(ub, max(map(lambda _dens: range_data(_dens)[1], densities.values())))

    if contain_plot:
        ext = ub - lb
        plot.x_range.start = lb - ext * 1e-2
        plot.x_range.end = ub + ext * 1e-2

        if is_pdf:
            plot.y_range.end = max([1e-10] + list(map(lambda _dens: _dens.practical_range_pdf[1], densities.values())))
            plot.y_range.start = 0. - .02 * plot.y_range.end
            plot.y_range.end += .02 * plot.y_range.end
        elif is_ppf:
            plot.x_range.start = 0. - 1e-2
            plot.x_range.end = 1. + 1e-2
            plot.y_range.start = lb - ext * 1e-2
            plot.y_range.end = ub + ext * 1e-2
        else:
            plot.y_range.start = 0. - .02
            plot.y_range.end = 1. + .02

        update_discrete_cont_mismatch()
        return


    v = input_own.value
    has_own = isinstance(v, int) or isinstance(v, float) or isinstance(v, Integral)
    own_values = []

    if is_ppf:
        input_own.step = 1. / 25.
    else:
        input_own.step = abs(ub - lb) / 25.

    for ctx in contexts:
        density = densities[ctx]
        use_v = v
        if has_own and selected_autotransf and density.transform_value is not None:
            use_v = np.abs(density.transform_value - v)
        lb_domain: float = None
        if is_ecdf:
            lb_domain = density.range[0]
        else:
            lb_domain = max(range_data(density)[0], density.practical_domain[0]) if selected_cutoff else density.practical_domain[0]
        ub_domain = min(range_data(density)[1], density.practical_domain[1]) if not is_ecdf and selected_cutoff else density.practical_domain[1]
        if lb_domain == ub_domain:
            ub_domain = lb + 1e-10
        npoints = 350


        if is_ppf:
            domain_x = np.linspace(start=1e-6, stop=1-1e-6, num=npoints)
        else:
            if is_ecdf or not is_pdf:
                npoints *= 2
            if is_empirical and is_discrete:
                npoints *= 3
            domain_x = nonlinspace(lb_domain, ub_domain, npoints)

        df_cols[f'x_{ctx}'] = domain_x
        if is_parametric and not density.is_fit:
            df_cols[ctx] = np.zeros((domain_x.size,))
            if has_own:
                own_values.append(0.)
            continue

        if is_pdf:
            df_cols[ctx] = density.pdf(domain_x)
            if has_own:
                own_values.append(density.pdf(use_v))
        elif is_ppf:
            df_cols[ctx] = density.ppf(df_cols[f'x_{ctx}'])
            if has_own:
                own_values.append(density.ppf(use_v))
        else:
            df_cols[ctx] = density.cdf(domain_x)
            if is_ccdf:
                df_cols[ctx] = 1. - df_cols[ctx]
            if has_own:
                if is_ccdf:
                    own_values.append(1. - density.cdf(use_v))
                else:
                    own_values.append(density.cdf(use_v))

    source.data = pd.DataFrame(df_cols)


    def tbl_format(v: float=None):
        if v is None:
            return '<none>'
        elif np.isnan(v):
            return '<nan>'

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

    def pdist_format(d: Parametric):
        if d.is_fit:
            return d.dist_name
        return '<not possible>'

    pdist: list[str] = None
    pdist_dval: list[str] = None
    if is_parametric:
        pdist = list([pdist_format(densities[context]) for context in contexts])
        pdist_dval = list([tbl_format(None if not densities[context].is_fit else densities[context].stat) for context in contexts])
    else:
        pdist = ['<not parametric>'] * len(contexts)
        if is_kde:
            pdist_dval = list([tbl_format(None if np.isnan(densities[context].stat) else densities[context].stat) for context in contexts])
        else:
            pdist_dval = list([tbl_format(None) for _ in range(len(contexts))])

    use_v = v
    if has_own and is_discrete:
        use_v = np.rint(v)
    tbl_transf_src.data = {
        'context': list(map(lambda ctx: '[All groups combined]' if ctx == '__ALL__' else ctx, contexts)),
        'transf_value': list(map(lambda context: tbl_format(None if not densities[context].transform_value is None and np.isnan(densities[context].transform_value) else densities[context].transform_value), contexts)),
        'qtype_value': list(map(lambda context: tbl_format(np.abs(densities[context].transform_value - use_v) if has_own and selected_autotransf and densities[context].transform_value is not None else use_v), contexts)),
        'own_value': list([tbl_format(v=v) for v in own_values]) if has_own else list([tbl_format(v=None) for _ in range(len(contexts))]),
        'pdist': pdist,
        'pdist_dval': pdist_dval
    }

    update_own_line()
    update_discrete_cont_mismatch()
    update_labels()


def cbg_cutoff_click(active: list[int]):
    """ :meta private: """
    global selected_cutoff
    selected_cutoff = len(active) > 0
    update_plot()


def cbg_autotrans_click(active: list[int]):
    """ :meta private: """
    global selected_autotransf
    selected_autotransf = len(active) > 0
    update_plot()


def dd_denstype_click(evt: MenuItemClick):
    """ :meta private: """
    global selected_denstype
    if evt.item == '-':
        return # Ignore, this is just a separator
    temp = { key: val for (val, key) in dd_denstype_items }
    dd_denstype.label = temp[evt.item]
    selected_denstype = (temp[evt.item], evt.item)

    update_plot()
    # For now, always call contain plot when the type changes, as even from one
    # CDF to the next (or PDF), the results can be vastly different and warrant
    # for a call to contain().
    # if type_before != type_after:
    update_plot(contain_plot=True)


def dd_transf_click(evt: MenuItemClick):
    """ :meta private: """
    global selected_transf
    transf_before = selected_transf[1]
    transf_after = evt.item
    temp = { key: val for (val, key) in dd_transf_items }
    dd_transf.label = temp[evt.item]
    selected_transf = (temp[evt.item], evt.item)
    update_plot()
    if transf_before != transf_after:
        update_plot(contain_plot=True)


def dd_scores_click(evt: MenuItemClick):
    """ :meta private: """
    global selected_score
    if evt.item == '-':
        return # Ignore, this is just a separator
    score_before = selected_score[1]
    score_after = evt.item
    temp = { key: val for (val, key) in dd_scores_items }
    dd_scores.label = temp[evt.item]
    selected_score = (temp[evt.item], evt.item)
    update_plot()
    if score_before != score_after:
        update_plot(contain_plot=True)


def btn_contain_click(*args):
    """ :meta private: """
    update_plot(contain_plot=True)


def btn_toggle_legend_click(*args):
    """ :meta private: """
    plot.legend.visible = not plot.legend.visible


def input_own_change(attr, old, new):
    """ :meta private: """
    update_plot()


def read_text(file: str) -> str:
    """ :meta private: """
    with open(file=file, mode='r', encoding='utf-8') as f:
        return f.read().strip()


cbg_cutoff.on_click(cbg_cutoff_click)
cbg_autotrans.on_click(cbg_autotrans_click)
dd_scores.on_click(dd_scores_click)
dd_transf.on_click(dd_transf_click)
dd_denstype.on_click(dd_denstype_click)
btn_contain.on_click(btn_contain_click)
input_own.on_change('value', input_own_change)
btn_toggle_legend.on_click(btn_toggle_legend_click)


html_about = web_dir.joinpath('./about.html')
html_refs = web_dir.joinpath('./references.html')
header = Div(text=f'''
    {read_text(this_dir.joinpath('header.html'))}
    <h2>Loaded Dataset: <b>{ds.ds["name"]}</b></h2>
    <div id="about">
        <p><b>Author(s)</b>: {', '.join(ds.ds["author"])}</p>
        <div>
            <p><b>Description</b>: {ds.ds["desc"]}</p>
            {read_text(html_about) if html_about.exists() else ""}
        </div>
    </div>
    <hr/>''')

footer = Div(text=f'''
    {read_text(this_dir.joinpath('footer.html'))}
    {read_text(html_refs) if html_refs.exists() else ""}
</ol>''')






input_row1 = row([
    dd_scores, dd_denstype, throbber, status
])
input_row2 = row([
    dd_transf, input_own, cbg_autotrans
])
input_row3 = row([
    btn_contain, btn_toggle_legend, cbg_cutoff
])

plot_row = row([column(tbl_transf, plot)], sizing_mode='stretch_both')
root_col = column(header, input_row1, input_row2, input_row3, plot_row, footer, sizing_mode='stretch_both')

curdoc().add_root(root_col)
curdoc().title = f'Metrics As Scores - {ds.ds["name"]}'

update_plot()
update_plot(contain_plot=True)
