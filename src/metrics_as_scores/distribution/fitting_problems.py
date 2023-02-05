"""
This module contains :code:`pymoo` fitting problems that allow fitting
distributions to almost arbitrary discrete data. The discrete random
variables in :code:`scipy` do not have a :code:`fit()`-method, as their fitting
often requires a global search. Also, many distributions require
discrete parameters, or a mixture of real and integer parameters.
The problems in this module provide a generalized way for `pymoo` to
find parameters for all of :code:`scipy`'s discrete random variables.

"""

import numpy as np
from nptyping import Float, NDArray, Shape
from scipy.stats._discrete_distns import rv_discrete, bernoulli_gen, betabinom_gen, binom_gen, boltzmann_gen, dlaplace_gen, geom_gen, hypergeom_gen, logser_gen, nbinom_gen, nchypergeom_fisher_gen, nchypergeom_wallenius_gen, nhypergeom_gen, planck_gen, poisson_gen, randint_gen, skellam_gen, yulesimon_gen, zipf_gen, zipfian_gen
from pymoo.core.variable import Variable
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer



class MixedVariableDistributionFittingProblem(ElementwiseProblem):
    """
    This is the base class for fitting all of ``scipy``'s discrete random variables.
    Therefore, it accepts a dictionary of parameters for each distribution to find
    optimal values for.
    """
    def __init__(self, dist: rv_discrete, data: NDArray[Shape['*'], Float], vars: dict[str, Variable], n_ieq_constr: int=0, **kwargs):
        """
        Constructor for a fitting any discrete random variable with one or more
        parameters that can be of any type as supported by ``pymoo.core.variable.Variable`` (e.g., ``Integer``, ``Real``, etc.).

        Parameters
        ----------
        dist: ``rv_discrete``
            An instance of the concrete discrete random variable that should be fit to the data.
        
        vars: ``dict[str, Variable]``
            An ordered dictionary of named variables to optimize. These must correspond
            one to one with the variable names of those defined for the random variable.

        data: ``NDArray[Shape['*'], Float]``
            The data the distribution should be fit to.
        
        n_ieq_constr: ``int``
            Number of inequality constraints. If there are any, then the problem also
            overrides ``_evaluate()`` and sets values for each constraint.
        """
        self.ext = int(np.max(data) - np.min(data))
        self.dist = dist
        # All of them have 'loc':
        vars['loc'] = Integer(bounds=(
            int(np.floor(np.min(data))) - (int(5 * self.ext)),
            int(np.ceil(np.max(data))) + int(5 * self.ext)))
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=n_ieq_constr, **kwargs)
        self.data = data
    
    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        r"""
        This is an internal method that evaluates the discrete random variable's
        negative log likelihood, given the currently set values for all of its
        variables (stored in ``X``). This method is called by ``pymoo``, so be
        sure to check out their references, too.
        Note that the ``X``-dictionary is used to build :math:`\theta`, the vector
        of parameters for the random variable. The order of the parameters is that
        vector depends on the order of ``self.vars``.
        This method usually does not need to be overridden, except for when, e.g.,
        it is required to evaluate (in-)equality constraints (that is, whenever
        something else than 'F' in the ``out``-dictionary must be accessed).

        X: ``dict[str, Any]``
            The (ordered) dictionary with the variables' names and values.
        out: ``dict[str, Any]``
            A dictionary used by ``pymoo`` to store results in; e.g., in 'F' it
            stores the result of the evaluation, and in 'G' it stores the inequality
            constraints' values.
        
        :return: Returns the ``out``-dictionary. However, the dictionary is accessed
            by reference, so this method does not have to return anything.

        :rtype: ``dict[Any,Any]``
        """
        theta = ()
        for vn in self.vars.keys():
            theta += (X[vn],)
        out['F'] = self.dist.nnlf(theta=theta, x=self.data)
        return out



class Fit_bernoulli_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Bernoulli distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``bernoulli_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``p``: (``int``) :math:`\left[0,1\right]]`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=bernoulli_gen(), data=data, vars={
            'p': Real(bounds=[0., 1.])
        }, **kwargs)


class Fit_betabinom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Beta-Binomial distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``betabinom_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``n``: (``int``) :math:`\left(0,1e^{4}\right)`
    - ``a``: (``float``) :math:`\left(5e^{-324},1e^3\right)`
    - ``b``: (``float``) :math:`\left(5e^{-324},1e^3\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=betabinom_gen(), data=data, vars={
            'n': Integer(bounds=(0, 10_000)),
            'a': Real(bounds=(5e-324, 1e3)),
            'b': Real(bounds=(5e-324, 1e3))
        }, **kwargs)


class Fit_binom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Binomial distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``binom_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``n``: (``int``) :math:`\left(1,25e^{3}\right)`
    - ``p``: (``float``) :math:`\left(0,1\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=binom_gen(), data=data, vars={
            'n': Integer(bounds=(1, 25_000)),
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_boltzmann_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Boltzman distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``boltzmann_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``lambda`` [:math:`\lambda`]: (``float``) :math:`\left(0,1e^{5}\right)`
    - ``N``: (``int``) :math:`\left(1,25e^3\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=boltzmann_gen(), data=data, vars={
            'lambda': Real(bounds=(0, 1e5)),
            'N': Integer(bounds=(1, 25_000))
        }, **kwargs)


class Fit_dlaplace_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Laplacian distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``dlaplace_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``a``: (``float``) :math:`\left(5e^{-324},1e^4\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=dlaplace_gen(), data=data, vars={
            'a': Real(bounds=(5e-324, 1e4))
        }, **kwargs)


class Fit_geom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Geometric distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``geom_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``p``: (``float``) :math:`\left(0,1\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=geom_gen(), data=data, vars={
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_hypergeom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Hypergeometric distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``hypergeom_gen`` as base distribution.

    Notes
    -----
    This problem **does** override `_evaluate()` and has four inequality constraints.
    These are:

    - :math:`n\geq0` (or :math:`-n\leq0`)
    - :math:`N\geq0` (or :math:`-N\leq0`)
    - :math:`n\leq M` (or :math:`n-M\leq0`)
    - :math:`N\leq M` (or :math:`N-M\leq0`)

    Calls the super constructor with these variables (in this order):

    - ``M``: (``int``) :math:`\left(1,25e^{3}\right)`
    - ``n``: (``int``) :math:`\left(0,25e^{3}\right)`
    - ``N``: (``int``) :math:`\left(0,25e^{3}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=hypergeom_gen(), data=data, vars={
            #M = [1, 25_000],         n = [0, 25_000],        N = [0, 25_000]
            'M': Integer(bounds=(1, 25_000)),
            'n': Integer(bounds=(0, 25_000)),
            'N': Integer(bounds=(0, 25_000))
        }, n_ieq_constr=4, **kwargs)
    
    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        """
        Overridden to evaluate the inequality constraints, too.
        For all other documentaion, check out ``MixedVariableDistributionFittingProblem._evaluate()``.
        """
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
    r"""
    This class allows to fit the generalized Logarithmic Series distribution using a
    ``pymoo`` problem. It uses ``scipy``'s ``logser_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``p``: (``float``) :math:`\left(0,1\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=logser_gen(), data=data, vars={
            'p': Real(bounds=(0., 1.))
        }, **kwargs)


class Fit_nbinom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Negative Binomial Series distribution using
    a ``pymoo`` problem. It uses ``scipy``'s ``nbinom_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``n``: (``int``) :math:`\left(0,25e^{3}\right)`
    - ``p``: (``float``) :math:`\left[0,1\right]`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=nbinom_gen(), data=data, vars={
            'n': Integer(bounds=(0, 25_000)),
            'p': Real(bounds=[0., 1.])
        }, **kwargs)


class Fit_nchypergeom_fisher_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Fisher's Non-central Hypergeometric
    distribution using a ``pymoo`` problem. It uses ``scipy``'s
    ``nchypergeom_fisher_gen`` as base distribution.

    Notes
    -----
    This problem **does** override `_evaluate()` and has four inequality constraints.
    These are:

    - :math:`N\leq M` (or :math:`N-M\leq0`)
    - :math:`n\leq M` (or :math:`n-M\leq0`)
    - :math:`\max{(\text{data})}\leq N` (or :math:`\max{(\text{data})}-N\leq0`)
    - :math:`\max{(\text{data})}\leq n` (or :math:`\max{(\text{data})}-n\leq0`)

    Calls the super constructor with these variables (in this order; note that :math:`k=` ``data.size``):

    - ``M``: (``int``) :math:`\left(1,5\times k\right)`
    - ``n``: (``int``) :math:`\left(1,5\times k\right)`
    - ``N``: (``int``) :math:`\left(1,5\times k\right)`
    - ``odds``: (``float``) :math:`\left(5e^{-324},1e^{4}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        vars = {
            'M': Integer(bounds=(1, 5 * data.size)),
            'n': Integer(bounds=(1, 5 * data.size)),
            'N': Integer(bounds=(1, 5 * data.size)),
            'odds': Real(bounds=(5e-324, 1e4))
        }
        super().__init__(dist=nchypergeom_fisher_gen(), data=data, vars=vars, n_ieq_constr=4, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs) -> dict:
        """
        Overridden to evaluate the inequality constraints, too.
        For all other documentaion, check out ``MixedVariableDistributionFittingProblem._evaluate()``.
        """
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
    r"""
    This class allows to fit the generalized Wallenius Non-central Hypergeometric
    distribution using a ``pymoo`` problem. It uses ``scipy``'s ``nchypergeom_wallenius_gen`` as base distribution.

    Notes
    -----
    This distribution has the same parameters and constraints as the
    nchypergeom_fisher_gen, which is implemented by the problem
    ``Fit_nchypergeom_fisher_gen`` (which this problem inherits from directly).
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(data, **kwargs)
        self.dist = nchypergeom_wallenius_gen()


class Fit_nhypergeom_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Negative Hypergeometric distribution
    using a ``pymoo`` problem. It uses ``scipy``'s ``nhypergeom_gen`` as base
    distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``M``: (``int``) :math:`\left(0,25e^3\right)`
    - ``n``: (``int``) :math:`\left(0,25e^3\right)`
    - ``r``: (``int``) :math:`\left(0,25e^3\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=nhypergeom_gen(), data=data, vars={
            'M': Integer(bounds=(0, 25_000)),
            'n': Integer(bounds=(0, 25_000)),
            'r': Integer(bounds=(0, 25_000))
        }, **kwargs)


class Fit_planck_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Planck distribution
    using a ``pymoo`` problem. It uses ``scipy``'s ``planck_gen`` as base
    distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``p``: (``float``) :math:`\left(5e^{-324},1e^2\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=planck_gen(), data=data, vars={
            # planck takes  as shape parameter. The Planck distribution can be written as a geometric distribution (geom) with p = 1 - exp(-lambda) shifted by loc = -1.
            # # exp(-lambda) get small very quickly, so we choose 100 (~3.7e-44)
            'p': Real(bounds=(5e-324, 1e2))
        }, **kwargs)


class Fit_poisson_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Poisson distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``poisson_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``mu`` [:math:`\mu`]: (``float``) :math:`\left(0,1e^{6}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=poisson_gen(), data=data, vars={
            'mu': Real(bounds=(0., 1e6))
        }, **kwargs)


class Fit_randint_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Uniform distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``randint_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``low``: (``int``) :math:`\left(-25e^{3},25e^{3}\right)`
    - ``high``: (``int``) :math:`\left(-25e^{3},25e^{3}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=randint_gen(), data=data, vars={
            'low': Integer(bounds=(-25_000, 25_000)),
            'high': Integer(bounds=(-25_000, 25_000))
        }, **kwargs)


class Fit_skellam_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Skellam distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``skellam_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``mu1`` [:math:`\mu_1`]: (``float``) :math:`\left(5e^{-324},5e^{3}\right)`
    - ``mu2`` [:math:`\mu_2`]: (``float``) :math:`\left(5e^{-324},5e^{3}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=skellam_gen(), data=data, vars={
            'mu1': Real(bounds=(5e-324, 5e3)),
            'mu2': Real(bounds=(5e-324, 5e3))
        }, **kwargs)


class Fit_yulesimon_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Yule--Simon distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``yulesimon_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``alpha`` [:math:`\alpha`]: (``float``) :math:`\left(5e^{-324},2e^{4}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=yulesimon_gen(), data=data, vars={
            'alpha': Real(bounds=(5e-324, 2e4)) # Guessed
        }, **kwargs)


class Fit_zipf_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Zipf (Zeta) distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``zipf_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``a``: (``float``) :math:`\left(1+1e^{-12},2e^{4}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=zipf_gen(), data=data, vars={
            'a': Real(bounds=(1. + 1e-12, 2e4)) # Guessed
        }, **kwargs)


class Fit_zipfian_gen(MixedVariableDistributionFittingProblem):
    r"""
    This class allows to fit the generalized Zipfian distribution using a ``pymoo``
    problem. It uses ``scipy``'s ``zipfian_gen`` as base distribution.

    Notes
    -----
    Does not override `_evaluate()` and does not have any (in-)equality constraints.
    Calls the super constructor with these variables (in this order):

    - ``a``: (``float``) :math:`\left(0,2e^{4}\right)`
    - ``n``: (``int``) :math:`\left(0,25e^{3}\right)`
    """
    def __init__(self, data: NDArray[Shape['*'], Float], **kwargs):
        super().__init__(dist=zipfian_gen(), data=data, vars={
            'a': Real(bounds=(0., 2e4)),
            'n': Integer(bounds=(0, 25_000))
        }, **kwargs)
