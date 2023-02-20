from re import escape
import numpy as np
from tempfile import tempdir
from pathlib import Path
from test_Dataset import get_elisa
from pickle import dump
from pytest import raises
#from metrics_as_scores.data.pregenerate_fit import fit, get_data_tuple
from metrics_as_scores.data.pregenerate_distns import generate_parametric_fits
from metrics_as_scores.distribution.fitting import FitterPymoo
from metrics_as_scores.distribution.distribution import DistTransform, Parametric
from metrics_as_scores.cli.FitParametric import FitParametricWorkflow
from metrics_as_scores.data.pregenerate import fits_to_MAS_densities, generate_parametric
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

    # Should throw for empty grid:
    with raises(Exception, match='Not enough quantity types and random variables were selected for fitting. Aborting.'):
        generate_parametric_fits(
            ds=ds,
            num_jobs=1,
            fitter_type=FitterPymoo,
            dist_transform=DistTransform.NONE,
            data_dict=data_dict,
            transform_values_dict=transform_values_dict,
            data_discrete_dict=data_discrete_dict,
            transform_values_discrete_dict=transform_values_discrete_dict,
            selected_rvs_c=[],
            selected_rvs_d=[])

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
    

    # Write res so we can read it in generate_parametric
    fits_dir = Path(tempdir)
    with open(file=str(fits_dir.joinpath(f'./pregen_distns_{DistTransform.NONE.name}.pickle')), mode='wb') as fp:
        dump(obj=res, file=fp)
    
    generate_parametric(dataset=ds, densities_dir=fits_dir, fits_dir=fits_dir, clazz=Parametric, transform=DistTransform.NONE)


    # Let's try to make MAS densities directly, because
    # computing the above is expensive for a unit test.
    distns_dict = { item['grid_idx']: item for item in res }
    dens_dict = fits_to_MAS_densities(dataset=ds, distns_dict=distns_dict, dist_transform=DistTransform.NONE, use_continuous=True)

    assert isinstance(dens_dict, dict)

    r1 = dens_dict['Run1_Lot1']
    temp = r1.cdf(np.linspace(start=0.9, stop=1.8, num=100))
    assert np.all((temp >= 0.0) & (temp <= 1.0))


    # Also test what happens if fits-file does not exist:
    import warnings
    warnings.filterwarnings("error")
    fits_file = str(fits_dir.joinpath(f'./pregen_distns_{DistTransform.EXPECTATION.name}.pickle').resolve()) 
    with raises(Exception, match=escape(f'Cannot generate parametric distribution for {Parametric.__name__} and transformation {DistTransform.EXPECTATION.name}, because the file {fits_file} does not exist. Did you forget to create the fits using the script pregenerate_distns.py?')):
        generate_parametric(dataset=ds, densities_dir=fits_dir, fits_dir=fits_dir, clazz=Parametric, transform=DistTransform.EXPECTATION)
    warnings.resetwarnings()
