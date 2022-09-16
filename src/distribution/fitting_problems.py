import numpy as np
from nptyping import Float, NDArray, Shape
from scipy.stats._discrete_distns import rv_discrete, bernoulli_gen, betabinom_gen, binom_gen, boltzmann_gen, dlaplace_gen, geom_gen, hypergeom_gen, logser_gen, nbinom_gen, nchypergeom_fisher_gen, nchypergeom_wallenius_gen, nhypergeom_gen, planck_gen, poisson_gen, randint_gen, skellam_gen, yulesimon_gen, zipf_gen, zipfian_gen
from pymoo.core.variable import Variable
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer



class MixedVariableDistributionFittingProblem(ElementwiseProblem):
    def __init__(self, dist: rv_discrete, data: NDArray[Shape['*'], Float], vars: dict[str, Variable], n_ieq_constr: int=0, **kwargs):
        self.ext = int(np.max(data) - np.min(data))
        self.dist = dist
        # All of them have 'loc':
        vars['loc'] = Integer(bounds=(
            int(np.floor(np.min(data))) - (int(5 * self.ext)),
            int(np.ceil(np.max(data))) + int(5 * self.ext)))
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=n_ieq_constr, **kwargs)
        self.data = data
    
    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        theta = ()
        for vn in self.vars.keys():
            theta += (X[vn],)
        out['F'] = self.dist.nnlf(theta=theta, x=self.data)
        return out



class Fit_bernoulli_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=bernoulli_gen(), data=data, vars={
            'p': Real(bounds=[0., 1.])
        }, **kwargs)


class Fit_betabinom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=betabinom_gen(), data=data, vars={
            'n': Integer(bounds=(0, 10_000)),
            'a': Real(bounds=(5e-324, 1e3)),
            'b': Real(bounds=(5e-324, 1e3))
        }, **kwargs)


class Fit_binom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=binom_gen(), data=data, vars={
            'n': Integer(bounds=(1, 25_000)),
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_boltzmann_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=boltzmann_gen(), data=data, vars={
            'lambda': Real(bounds=(0, 1e5)),
            'N': Integer(bounds=(1, 25_000))
        }, **kwargs)


class Fit_dlaplace_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=dlaplace_gen(), data=data, vars={
            'a': Real(bounds=(5e-324, 1e4))
        }, **kwargs)


class Fit_geom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=geom_gen(), data=data, vars={
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_hypergeom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=hypergeom_gen(), data=data, vars={
            #M = [1, 25_000],         n = [0, 25_000],        N = [0, 25_000]
            'M': Integer(bounds=(1, 25_000)),
            'n': Integer(bounds=(0, 25_000)),
            'N': Integer(bounds=(0, 25_000))
        }, n_ieq_constr=4, **kwargs)
    
    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        out = super()._evaluate(X, out, *args, **kwargs)
        M, n, N = X['M'], X['n'], X['N']
        out['G'] = np.asarray([
            -n, # n >= 0  ->  -n <= 0
            -N, # N >= 0  ->  -N <= 0
            n - M, # n <= M  ->  n - M <= 0
            N - M] # N <= M  ->  N - M <= 0
        )
        return out


class Fit_logser_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=logser_gen(), data=data, vars={
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_nbinom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=nbinom_gen(), data=data, vars={
            'n': Integer(bounds=(0, 25_000)),
            'p': Real(bounds=[0., 1.])
        }, **kwargs)


class Fit_nchypergeom_fisher_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        vars = {
            'M': Integer(bounds=(1, 5 * data.size)),
            'n': Integer(bounds=(1, 5 * data.size)),
            'N': Integer(bounds=(1, 5 * data.size)),
            'odds': Real(bounds=(5e-324, 1e4))
        }
        super().__init__(dist=nchypergeom_fisher_gen(), data=data, vars=vars, n_ieq_constr=4, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        out = super()._evaluate(X=X, out=out, *args, **kwargs)
        M, n, N, b = X['M'], X['n'], X['N'], np.max(self.data)

        # Let's add the following inequality constraints to the output:
        out['G'] = np.asarray([
            N - M, # N <= M  ->  N - M <= 0
            n - M, # n <= M  ->  n - M <= 0
            b - N, # max(data) <= N  ->  max(data) - N <= 0
            b - n] # max(data) <= n  ->  max(data) - n <= 0
        )
        return out


class Fit_nchypergeom_wallenius_gen(Fit_nchypergeom_fisher_gen):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        """
        This distribution has the same parameters and constraints,
        everything else is the same as for nchypergeom_fisher_gen.
        """
        super().__init__(data, **kwargs)
        self.dist = nchypergeom_wallenius_gen()


class Fit_nhypergeom_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=nhypergeom_gen(), data=data, vars={
            'M': Integer(bounds=(0, 25_000)),
            'n': Integer(bounds=(0, 25_000)),
            'r': Integer(bounds=(0, 25_000))
        }, **kwargs)


class Fit_planck_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=planck_gen(), data=data, vars={
            # planck takes  as shape parameter. The Planck distribution can be written as a geometric distribution (geom) with p = 1 - exp(-lambda) shifted by loc = -1.
            # # exp(-lambda) get small very quickly, so we choose 100 (~3.7e-44)
            'p': Real(bounds=(5e-324, 1e2))
        }, **kwargs)


class Fit_poisson_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=poisson_gen(), data=data, vars={
            'mu': Real(bounds=(0., 1e6))
        }, **kwargs)


class Fit_randint_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=randint_gen(), data=data, vars={
            'low': Integer(bounds=(-25_000, 25_000)),
            'high': Integer(bounds=(-25_000, 25_000))
        }, **kwargs)


class Fit_skellam_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=skellam_gen(), data=data, vars={
            'mu1': Real(bounds=(5e-324, 5e3)),
            'mu2': Real(bounds=(5e-324, 5e3))
        }, **kwargs)


class Fit_yulesimon_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=yulesimon_gen(), data=data, vars={
            'alpha': Real(bounds=(5e-324, 2e4)) # Guessed
        }, **kwargs)


class Fit_zipf_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=zipf_gen(), data=data, vars={
            'a': Real(bounds=(1. + 1e-12, 2e4)) # Guessed
        }, **kwargs)


class Fit_zipfian_gen(MixedVariableDistributionFittingProblem):
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=zipfian_gen(), data=data, vars={
            'a': Real(bounds=(0., 2e4)),
            'n': Integer(bounds=(0, 25_000))
        }, **kwargs)
