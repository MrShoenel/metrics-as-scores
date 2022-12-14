Metrics as Scores [![DOI](https://zenodo.org/badge/524333119.svg)](https://zenodo.org/badge/latestdoi/524333119) [![status](https://joss.theoj.org/papers/eb549efe6c0111490395496c68717579/status.svg)](https://joss.theoj.org/papers/eb549efe6c0111490395496c68717579)
=================

Contains the data and scripts needed for the application __`Metrics as Scores`__, check out <https://metrics-as-scores.ml/>.

This package accompanies the paper entitled "_Contextual Operationalization of Metrics As Scores: Is My Metric Value Good?_", which is currently under single-blind review.
It seeks to answer the question whether or not the context a metric was captured in, matters.
It enables the user to compare contexts and to understand their differences.
In order to answer the question of whether a metric value is actually good, we need to transform it into a **score**.
Scores are normalized **and rectified** distances, that can be compared in an apples-to-apples manner, across contexts.
The same metric value might be good in one context, while it is not in another.
To borrow an example from the context of software: It is much more acceptable (or common) to have large applications (in terms of lines of code) in the contexts/domains of games and databases than it is for the domains of IDEs and SDKs.
Given an *ideal* value for a metric (which may also be user-defined), we can transform observed metrics values to distances from that value and then use the cumulative distribution function to map distances to scores.


Jump to:


- [1. Installation](#1-installation)
- [2. Stand-alone Usage / Development Setup](#2-stand-alone-usage--development-setup)
	- [2.1. Setting Up a Virtual Environment](#21-setting-up-a-virtual-environment)
	- [2.2. Installing Packages](#22-installing-packages)
- [3. Use Your Own Data](#3-use-your-own-data)
	- [3.1. Data Preparation](#31-data-preparation)
	- [3.2. [Optional] Computing Fits For Parametric Distributions](#32-optional-computing-fits-for-parametric-distributions)
	- [3.3. Pre-generating Distributions](#33-pre-generating-distributions)
- [4. Personalizing the Web Application](#4-personalizing-the-web-application)
- [References](#references)


# 1. Installation

For using the package in your own project, install it from PyPI:

```shell
pip install metrics-as-scores
```

However, if you are interested in importing your own data (see below), it is perhaps best to just clone this repo and run Metrics As Scores as a standalone application.

# 2. Stand-alone Usage / Development Setup

This project was developed using and requires Python `3.10`.
For using Metrics As Scores as standalone application (you want to do this when importing your own data, customizing the web application, or supporting development),

1. Clone the Repository,
2. Set up  a virtual environment,
3. Install packages.

## 2.1. Setting Up a Virtual Environment

It is recommended to use a virtual environment.
To use a virtual environment, follow these steps (Windows specific; activation of the environment might differ).

```shell
virtualenv --python=C:/Python310/python.exe venv # Use specific Python version for virtual environment
venv/Scripts/activate
```

Here is a Linux example that assumes you have Python `3.10` installed (this may also require installing `python3.10-venv` and/or `python3.10-dev`):

```shell
python3.10 -m venv venv
source venv/bin/activate # Linux
```

## 2.2. Installing Packages

The project is managed with `Poetry`.
To install the required packages, simply run the following.

```shell
venv/Scripts/activate
# Assuming you are in the activate virtual environment (Windows)
(venv) C:\repos\lnu_metrics-as-scores> poetry install
```

The same in Linux:

```shell
source venv/bin/activate # Linux
(venv) ubuntu@vm:/tmp/metrics-as-scores$ poetry install
```


# 3. Use Your Own Data

Although this application was built primarily for analyzing the software metrics from the "Qualitas.class" corpus [[1]](#cite-1)[[2]](#cite-2), it can work with any kind of data! Everything required for importing and operationalizing own data is implemented in a use case-agnostic way. Also, the web application can be adapted quickly by swapping out the header and footer (see below).


In order to use your own data, two steps are required:

1. Provide your data in form of a `CSV`-file and adapt the enum `MetricID`.
2. Run the scripts for pre-generating the distributions for high performance in the web application.
3. [Optional] You may run additional scripts that perform additional statistical tests. Currently, we support these tests:
	* ANOVA [[3]](#cite-3): analysis of variance of your data across the available contexts,
	* Tukey's Honest Significance Test (TukeyHSD) [[4]](#cite-4): This test is used to gain insights into the results of an ANOVA test. While the former only allows obtaining the amount of corroboration for the null hypothesis, TukeyHSD performs all pairwise comparisons (for all possible combinations of any two contexts),
	* Two-sample T-test: Compares the means of two samples to give an indication whether or not they appear to come from the same distribution. Again, this is useful for comparing contexts.


Note that in step \#2, if you decide to also pre-generate fitted parametric distributions, that each distribution's goodness-of-fit is evaluated using various additional one- and two-sample tests:

* Cram??r-von Mises- [[5]](#cite-5) and Kolmogorov&ndash;Smirnov one-sample [[6]](#cite-6) tests: After fitting a distribution, the sample is tested against the fitted parametric distribution. Since the fitted distribution cannot usually accommodate all of the sample's subtleties, the test will indicate whether the fit is acceptable or not.
* Cram??r-von Mises- [[7]](#cite-7), Kolmogorov&ndash;Smirnov-, and Epps&ndash;Singleton [[8]](#cite-8) two-sample tests: After fitting, we create a second sample by uniformly sampling from the `PPF`. Then, both samples can be used in these tests. The Epps&ndash;Singleton test is also applicable for discrete distributions.

Note that the tests are automatically carried out for either continuous or discrete data (not each test is valid for discrete data, such as the KS two-sample test).


## 3.1. Data Preparation

You will have to adapt the enum [__`src/metrics_as_scores/data/metrics.py/MetricID`__](./src/metrics_as_scores/data/metrics.py).
This is a simple Key-Value enumeration, where the key is the metric's short name or abbreviation (letters only, e.g., "KPI"). The value can be any string. It is shown in the web application like "`[Metric] Value`".

Next, you will have to provide the following `CSV`-files:

* [__`files/metrics-discrete.csv`__](./files/metrics-discrete.csv): Another simple two-column `CSV` with columns `Metric` and `Discrete`. This file is used to indicate whether a metric is discrete or continuous. Use the metric's short name in the first column, and either `True` or `False` in the other column.
* [__`files/metrics-ideal.csv`__](./files/metrics-ideal.csv): A third simple file. Similar to the previous one, in this file you can indicate a numeric ideal value for each metric (if any). Again, use the metric's short name in the first column, and either keep the second column empty (no ideal value) or fill in an ideal value. Note that this file indicates the global ideal values, not user-preferred ideal values. The value from this file is used when distributions are pre-generated.
* [__`csv/metrics.csv`__](./csv/metrics.csv.7z): This is the <u>***main data file***</u>. It has three columns: `Metric`, `Domain`, and `Value`. Here you save the values that you have recorded for each metric, in each context/domain.


## 3.2. [Optional] Computing Fits For Parametric Distributions

This step can be skipped if you **do not** want make use of parametric distributions. You will still have access to empirical distributions and Kernel density estimates.
Please note that this step is, computationally, **extremely expensive**. This is because for each metric, in each context, up to 120 distributions are fitted. About 20 of these (the discrete distributions) are fit using __`Pymoo`__ [[9]](#cite-9) and a mixed-variable global optimization. Some other distributions are currently deliberately disabled, because computing a single fit can take up to one day and longer (see the variable `ignored_dists` in [`src/metrics_as_scores/data/pregenerate_distns.py`](./src/metrics_as_scores/data/pregenerate_distns.py)). Enable those at your own risk.


If you read this far, you probably want to compute parametric fits :)
In order to do that, run the below script from the root of this repository:

```shell
# Activate venv (Linux)
source venv/bin/activate
# Call the script with Python >= 3.10 (no further arguments):
python3.10 src/metrics_as_scores/data/pregenerate_distns.py
```

Note that this script exploits all available CPU cores and thus is heavily parallelized.


## 3.3. Pre-generating Distributions

This step is obligatory. If you have not previously created the fits for parametric distributions, the script called here will warn (can be ignored if you had no intention).
The purpose of this step is to trade space for computing time. The pre-generated distributions require disk space and RAM (a few hundred megabytes per transform and -distribution type [Empirical, Empirical_discrete, and KDE_approx]).
The size of the pre-generated _parametric_ distributions can be ignored (few megabytes).
However, it allows for a smooth workflow in the web application later.


This step requires some compute power and will also leverage all available CPU cores. It is, however, by far not as heavy as the previous step (calculate about a minute per transform and -distribution).
You will need to run this script:

```shell
# Activate venv (Linux)
source venv/bin/activate
# Call the script with Python >= 3.10 (no further arguments):
python3.10 src/metrics_as_scores/data/pregenerate.py
```


# 4. Personalizing the Web Application

The web application _"[Metrics As Scores](https://metrics-as-scores.ml/)"_ is located in the directory [`src/metrics_as_scores/webapp/`](./src/metrics_as_scores/webapp/).
The app itself has three vertical blocks: a header, the interactive part, and a footer.
Header and footer can be easily edited by modifing the files [`src/metrics_as_scores/webapp/header.html`](./src/metrics_as_scores/webapp/header.html) and [`src/metrics_as_scores/webapp/footer.html`](./src/metrics_as_scores/webapp/footer.html).

If you want to change the title of the application, you will have to modify the file [`src/metrics_as_scores/webapp/main.py`](./src/metrics_as_scores/webapp/main.py) at the very end:

```python
# Change this line to your desired title.
curdoc().title = "Metrics As Scores"
```

**Important**: If you modify the web application, you must always maintain two links: one to <https://metrics-as-scores.ml/> and one to this repository, that is, <https://github.com/MrShoenel/metrics-as-scores>.




# References

<a id="cite-1"></a>[1] Terra, R., Miranda, L.F., Valente, M.T. and Bigonha, R.S., 2013. "Qualitas.class Corpus: A compiled version of the Qualitas Corpus". ACM SIGSOFT Software Engineering Notes, 38(5), pp. 1-4. <https://dx.doi.org/10.1145/2507288.2507314>.

<a id="cite-2"></a>[2] Tempero, E., Anslow, C., Dietrich, J., Han, T., Li, J., Lumpe, M., Melton, H. and Noble, J., 2010, December. "The qualitas corpus: A curated collection of java code for empirical studies". In 2010 Asia pacific software engineering conference (pp. 336-345). IEEE. <https://doi.org/10.1109/APSEC.2010.46>.

<a id="cite-3"></a>[3] Chambers, John M., and Trevor J. Hastie. "Statistical Models". Statistical Models in S. Routledge, 2017. 13???44. <https://doi.org/10.1201/9780203738535>.

<a id="cite-4"></a>[4] Tukey, John W. "Comparing Individual Means in the Analysis of Variance." Biometrics, vol. 5, no. 2, 1949, pp. 99???114. JSTOR, <https://doi.org/10.2307/3001913>.

<a id="cite-5"></a>[5] Cram??r, H., 1928. "On the composition of elementary errors: Statistical applications". Almqvist and Wiksell. <https://doi.org/10.1080/03461238.1928.10416862>.

<a id="cite-6"></a>[6] Stephens, M.A., 1974. "EDF statistics for goodness of fit and some comparisons". Journal of the American statistical Association, 69(347), pp. 730-737. <https://doi.org/10.1080/01621459.1974.10480196>.

<a id="cite-7"></a>[7] Anderson, T.W., 1962. "On the distribution of the two-sample Cramer-von Mises criterion". The Annals of Mathematical Statistics, pp. 1148-1159. <https://doi.org/10.1214/aoms/1177704477>.

<a id="cite-8"></a>[8] Epps, T.W. and Singleton, K.J., 1986. "An omnibus test for the two-sample problem using the empirical characteristic function". Journal of Statistical Computation and Simulation, 26(3-4), pp. 177-203. <https://doi.org/10.1080/00949658608810963>.

<a id="cite-9"></a>[9] Blank, J. and Deb, K. (2020) "Pymoo: Multi-Objective Optimization in Python", IEEE Access. Institute of Electrical and Electronics Engineers (IEEE), 8, pp. 89497???89509. <https://doi.org/10.1109/access.2020.2990567>.