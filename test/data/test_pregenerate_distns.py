from test_Dataset import get_elisa
#from metrics_as_scores.data.pregenerate_fit import fit, get_data_tuple
from metrics_as_scores.data.pregenerate_distns import generate_parametric_fits
from metrics_as_scores.distribution.fitting import FitterPymoo
from metrics_as_scores.distribution.distribution import DistTransform
from metrics_as_scores.cli.FitParametric import FitParametricWorkflow
from scipy.stats._continuous_distns import norm_gen





def test_generate_parametric_fits():
    ds = get_elisa()
    ds.ds['qtypes'] = { 'Lot1': 'continuous' }
    ds.ds['contexts'] = ['Run1']
    fpw = FitParametricWorkflow()
    fpw.num_cpus = 1
    fpw.ds = ds

    transform_values_dict, data_dict = fpw._get_data_tuples(dist_transform=DistTransform.NONE, continuous=True)
    # Discrete:
    transform_values_discrete_dict, data_discrete_dict = fpw._get_data_tuples(dist_transform=DistTransform.NONE, continuous=False)

    res = generate_parametric_fits(
        ds=ds,
        num_jobs=2,
        fitter_type=FitterPymoo,
        dist_transform=DistTransform.NONE,
        data_dict=data_dict,
        transform_values_dict=transform_values_dict,
        data_discrete_dict=data_discrete_dict,
        transform_values_discrete_dict=transform_values_discrete_dict,
        selected_rvs_c=[norm_gen],
        selected_rvs_d=[])
    
    for fr in res:
        assert isinstance(fr['params'], dict)
        
    
