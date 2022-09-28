Metrics as Scores
=================

Contains the data and scripts needed for the application __`Metrics as Scores`__, see <https://metrics-as-scores.ml/>.


# Use Your Own Data

Although this application was built primarily for analyzing the software metrics from the "Qualitas.class" corpus [[1]](#cite-1)[[2]](#cite-2), it can work with any kind of data! Everything required for importing and operationalizing own data is implemented in a use case-agnostic way. Also, the web application can be adapted quickly by swapping out the header and footer (see below).


In order to use your own data, two steps are required:

1. Provide your data in form of a `CSV`-file.
2. Run the scripts for pre-generating the distributions for high performance in the web application.
3. [Optional] You may run additional scripts that perform additional statistical tests. Currently, we support these tests:
	* ANOVA [[3]](#cite-3): analysis of variance of your data across the available contexts,
	* Tukey's Honest Significance Test (TukeyHSD) [[4]](#cite-4): This test is used to gain insights into the results of an ANOVA test. While the former only allows obtaining the amount of corroboration for the null hypothesis, TukeyHSD performs all pairwise comparisons (for all possible combinations of any two contexts),
	* Two-sample T-test: Compares the means of two samples to give an indication whether or not they appear to come from the same distribution. Again, this is useful for comparing contexts.


Note that in step \#2, if you decide to also pre-generate fitted parametric distributions, that each distribution's goodness-of-fit is evaluated using various additional one- and two-sample tests:

* Cramér-von Mises- [[5]](#cite-5) and Kolmogorov&ndash;Smirnov one-sample [[6]](#cite-6) tests: After fitting a distribution, the sample is tested against the fitted parametric distribution. Since the fitted distribution cannot usually accommodate all of the sample's subtleties, the test will indicate whether the fit is acceptable or not.
* Cramér-von Mises- [[7]](#cite-7), Kolmogorov&ndash;Smirnov-, and Epps&ndash;Singleton [[8]](#cite-8) two-sample tests: After fitting, we create a second sample by uniformly sampling from the `PPF`. Then, both samples can be used in these tests. The Epps&ndash;Singleton test is also applicable for discrete distributions.

Note that the tests are automatically carried out for either continuous or discrete data (not each test is valid for discrete data, such as the KS two-sample test).


## Data Preparation

You will have to provide the following `CSV`-files:

* [__`files/MetricID.csv`__](./files/MetricID.csv): A simple file with two columns (additional columns are ignored, but you may want to store extra meta information there): `Metric` and `Value`. The first column should contain the metric's short name or abbreviation (letters only, e.g., "KPI") and the value can be any string. It is shown in the web application like "`[Metric] Value`".
* [__`files/metrics-discrete.csv`__](./files/metrics-discrete.csv): Another simple two-column `CSV` with columns `Metric` and `Discrete`. This file is used to indicate whether a metric is discrete or continuous. Use the metric's short name in the first column, and either `True` or `False` in the other column.
* [__`files/metrics-ideal.csv`__](./files/metrics-ideal.csv): A third simple file. Similar to the previous one, in this file you can indicate a numeric ideal value for each metric (if any). Again, use the metric's short name in the first column, and either keep the second column empty (no ideal value) or fill in an ideal value. Note that this file indicates the global ideal values, not user-preferred ideal values. The value from this file is used when distributions are pre-generated.
* [__`csv/metrics.csv`__](./csv/metrics.csv): This is the <u>***main data file***</u>. It has three columns: `Metric`, `Domain`, and `Value`. Here you save the values that you have recorded for each metric, in each context/domain.


## Computing Fits For Parametric Distributions [Optional]

This step can be skipped if you **do not** want make use of parametric distributions. You will still have access to empirical distributions and Kernel density estimates.
Please note that this step is, computationally, **extremely expensive**. This is because for each metric, in each context, up to 120 distributions are fitted. About 20 of these (the discrete distributions) are fit using __`Pymoo`__ [[9]](#cite-9) and a mixed-variable global optimization. Some other distributions are currently deliberately disabled, because computing a single fit can take up to one day (see the variable `ignored_dists` in [`src/data/pregenerate_distns.py`](./src/data/pregenerate_distns.py)). Enable those at your own risk.


If you read this far, you probably want to compute parametric fits :)
In order to do that, run the below script from the root of this repository:

```shell
# Activate venv (Linux)
source venv/bin/activate
# Call the script with Python >= 3.10 (no further arguments):
python3.10 src/data/pregenerate_distns.py
```

Note that this script exploits all available CPU cores and thus is heavily parallelized.


## Pre-generating Distributions

This step is obligatory. If you have not previously created the fits for parametric distributions (previous step), the script called here will warn (can be ignored if you had no intention).
The purpose of this step is to trade space for computing time. The pre-generated distributions require disk space and RAM (a few hundred megabytes per transform and -distribution type [Empirical, Empirical_discrete, and KDE_approx]).
However, it allows for a smooth workflow in the web application later.

This step requires some compute power and will also leverage all available CPU cores. It is, however, by far not as heavy as the previous step (calculate about a minute per transform and -distribution).
You will need to run this script:

```shell
# Activate venv (Linux)
source venv/bin/activate
# Call the script with Python >= 3.10 (no further arguments):
python3.10 src/data/pregenerate.py
```


## Personalizing the Web Application

The web application "Metrics As Scores" is located in the directory [`src/webapp/`](./src/webapp/).
The app itself has three vertical blocks: a header, the interactive part, and a footer.
Header and footer can be easily edited by modifing the files [`src/webapp/header.html`](./src/webapp/header.html) and [`src/webapp/footer.html`](./src/webapp/footer.html).

If you want to change the title of the application, you will have to modify the file [`src/webapp/main.py`](./src/webapp/main.py) at the very end:

```python
# Change this line to your desired title.
curdoc().title = "Metrics As Scores"
```

**Important**: If you modify the web application, you must always maintain two links: one to <https://metrics-as-scores.ml/> and one to this repository, that is, <https://github.com/MrShoenel/metrics-as-scores>.


# Setup (development)

This project was developed using and requires Python `3.10`.
To use a virtual environment, follow these steps (Windows specific; activation of the environment might differ).

```shell
virtualenv --python=C:/Python310/python.exe venv # Use specific Python version for virtual environment
venv/Scripts/activate
pip install -r requirements.txt
```

Here is a Linux example that assumes you have Python `3.10` installed (this may also require installing `python3.10-venv` and/or `python3.10-dev`):

```shell
python3.10 -m venv venv
source venv/bin/activate # Linux
pip install -r requirements.txt
```


# References

<a id="cite-1"></a>[1] Terra, R., Miranda, L.F., Valente, M.T. and Bigonha, R.S., 2013. Qualitas.class Corpus: A compiled version of the Qualitas Corpus. ACM SIGSOFT Software Engineering Notes, 38(5), pp. 1-4. <https://dx.doi.org/10.1145/2507288.2507314>.

<a id="cite-2"></a>[2] Tempero, E., Anslow, C., Dietrich, J., Han, T., Li, J., Lumpe, M., Melton, H. and Noble, J., 2010, December. The qualitas corpus: A curated collection of java code for empirical studies. In 2010 Asia pacific software engineering conference (pp. 336-345). IEEE. <https://doi.org/10.1109/APSEC.2010.46>.

<a id="cite-3"></a>[3] Chambers, John M., και Trevor J. Hastie. "Statistical Models". Statistical Models in S. Routledge, 2017. 13–44. <https://doi.org/10.1201/9780203738535>.

<a id="cite-4"></a>[4] Tukey, John W. "Comparing Individual Means in the Analysis of Variance." Biometrics, vol. 5, no. 2, 1949, pp. 99–114. JSTOR, <https://doi.org/10.2307/3001913>.

<a id="cite-5"></a>[5] Cramér, H., 1928. On the composition of elementary errors: Statistical applications. Almqvist and Wiksell. <https://doi.org/10.1080/03461238.1928.10416862>.

<a id="cite-6"></a>[6] Stephens, M.A., 1974. EDF statistics for goodness of fit and some comparisons. Journal of the American statistical Association, 69(347), pp. 730-737. <https://doi.org/10.1080/01621459.1974.10480196>.

<a id="cite-7"></a>[7] Anderson, T.W., 1962. On the distribution of the two-sample Cramer-von Mises criterion. The Annals of Mathematical Statistics, pp. 1148-1159. <https://doi.org/10.1214/aoms/1177704477>.

<a id="cite-8"></a>[8] Epps, T.W. and Singleton, K.J., 1986. An omnibus test for the two-sample problem using the empirical characteristic function. Journal of Statistical Computation and Simulation, 26(3-4), pp. 177-203. <https://doi.org/10.1080/00949658608810963>.

<a id="cite-9"></a>[9] Blank, J. and Deb, K. (2020) "Pymoo: Multi-Objective Optimization in Python", IEEE Access. Institute of Electrical and Electronics Engineers (IEEE), 8, pp. 89497–89509. <https://doi.org/10.1109/access.2020.2990567>.