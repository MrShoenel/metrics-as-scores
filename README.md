Metrics As Scores
[![DOI](https://zenodo.org/badge/524333119.svg)](https://zenodo.org/badge/latestdoi/524333119)
[![status](https://joss.theoj.org/papers/eb549efe6c0111490395496c68717579/status.svg)](https://joss.theoj.org/papers/eb549efe6c0111490395496c68717579)
[![codecov](https://codecov.io/github/MrShoenel/metrics-as-scores/branch/master/graph/badge.svg?token=HO1GYXVEUQ)](https://codecov.io/github/MrShoenel/metrics-as-scores)
================

- <a href="#usage" id="toc-usage"><span
  class="toc-section-number">1</span> Usage</a>
  - <a href="#text-based-user-interface-tui"
    id="toc-text-based-user-interface-tui"><span
    class="toc-section-number">1.1</span> Text-based User Interface
    (TUI)</a>
  - <a href="#web-application" id="toc-web-application"><span
    class="toc-section-number">1.2</span> Web Application</a>
  - <a href="#development-setup" id="toc-development-setup"><span
    class="toc-section-number">1.3</span> Development Setup</a>
    - <a href="#setting-up-a-virtual-environment"
      id="toc-setting-up-a-virtual-environment"><span
      class="toc-section-number">1.3.1</span> Setting Up a Virtual
      Environment</a>
    - <a href="#installing-packages" id="toc-installing-packages"><span
      class="toc-section-number">1.3.2</span> Installing Packages</a>
    - <a href="#running-tests" id="toc-running-tests"><span
      class="toc-section-number">1.3.3</span> Running Tests</a>
- <a href="#example-usage" id="toc-example-usage"><span
  class="toc-section-number">2</span> Example Usage</a>
  - <a href="#software-metrics-example"
    id="toc-software-metrics-example"><span
    class="toc-section-number">2.1</span> Software Metrics Example</a>
  - <a href="#diamonds-example" id="toc-diamonds-example"><span
    class="toc-section-number">2.2</span> Diamonds Example</a>
- <a href="#datasets" id="toc-datasets"><span
  class="toc-section-number">3</span> Datasets</a>
  - <a href="#use-your-own" id="toc-use-your-own"><span
    class="toc-section-number">3.1</span> Use Your Own</a>
  - <a href="#known-datasets" id="toc-known-datasets"><span
    class="toc-section-number">3.2</span> Known Datasets</a>
- <a href="#personalizing-the-web-application"
  id="toc-personalizing-the-web-application"><span
  class="toc-section-number">4</span> Personalizing the Web
  Application</a>
- <a href="#references" id="toc-references">References</a>

------------------------------------------------------------------------

**Please Note**: ***Metrics As Scores*** (`MAS`) changed considerably
between versions
[**`v1.0.8`**](https://github.com/MrShoenel/metrics-as-scores/tree/v1.0.8)
and **`v2.x.x`**.

The current version is `v2.2.0`.

From version **`v2.x.x`** it has the following new features:

- [Textual User Interface (TUI)](#text-based-user-interface-tui)
- Proper documentation and testing
- New version on PyPI. Install the package and run the command line
  interface by typing **`mas`**!

<video controls autoplay loop muted>
<source src="demo.webm" type="video/webm">
</source>
<source src="demo.mp4" type="video/mp4">
</source>
</video>

------------------------------------------------------------------------

Contains the data and scripts needed for the application
**`Metrics as Scores`**, check out <https://metrics-as-scores.ml/>.

This package accompanies the paper entitled “*Contextual
Operationalization of Metrics As Scores: Is My Metric Value Good?*”
(Hönel et al. 2022). It seeks to answer the question whether or not the
context a software metric was captured in, matters. It enables the user
to compare contexts and to understand their differences. In order to
answer the question of whether a metric value is actually good, we need
to transform it into a **score**. Scores are normalized **and
rectified** distances, that can be compared in an apples-to-apples
manner, across contexts. The same metric value might be good in one
context, while it is not in another. To borrow an example from the
context of software: It is much more acceptable (or common) to have
large applications (in terms of lines of code) in the contexts/domains
of games and databases than it is for the domains of IDEs and SDKs.
Given an *ideal* value for a metric (which may also be user-defined), we
can transform observed metric values to distances from that value and
then use the cumulative distribution function to map distances to
scores.

------------------------------------------------------------------------

# Usage

You may install Metrics As Scores directly from PyPI. For users that
wish to
[**contribute**](https://github.com/MrShoenel/metrics-as-scores/blob/master/CONTRIBUTING.md)
to Metrics As Scores, a [development setup](#development-setup) is
recommended. In either case, after the installation, [**you have access
to the text-based user interface**](#text-based-user-interface-tui).

``` shell
# Installation from PyPI:
pip install metrics-as-scores
```

You can **bring up the TUI** simply by typing the following after
installing or cloning the repo (see next section for more details):

``` shell
mas
```

## Text-based User Interface (TUI)

Metrics As Scores features a text-based command line user interface
(TUI). It offers a couple of workflows/wizards, that help you to work
and interact with the application. There is no need to modify any source
code, if you want to do one of the following:

- Show Installed Datasets
- Show List of Known Datasets Available Online That Can Be Downloaded
- Download and install a known or existing dataset
- Create Own Dataset to be used with Metrics-As-Scores
- Fit Parametric Distributions for Own Dataset
- Pre-generate distributions for usage in the
  [**Web-Application**](#web-application)
- Bundle Own dataset so it can be published
- Run local, interactive Web-Application using a selected dataset

![Metrics As Scores Text-based User Interface
(TUI).](./TUI.png "Metrics As Scores Text-based User Interface (TUI).")

## Web Application

Metrics As Scores’ main feature is perhaps the Web Application. It can
be run directly and locally from the TUI using a selected dataset (you
may download a known dataset or use your own). The Web Application
allows to visually inspect each *quantity type* across all the defined
contexts. It feates the PDF/PMF, CDF and CCDF, as well as the PPF for
each quantity in each context. It offers five different principal types
of densities: Parametric, Parametric (discrete), Empirical, Empirical
(discrete), and (approximate) Kernel Density Estimation. The Web
Application includes a detailed [Help
section](https://metrics-as-scores.ml/#help) that should answer most of
your questions.

![Metrics As Scores Interactive Web
.](./WebApp.png "Metrics As Scores Interactive Web Appliction.")

## Development Setup

This project was developed using and requires Python `>=3.10`. Steps:

1.  Clone the Repository,
2.  Set up a virtual environment,
3.  Install packages.

### Setting Up a Virtual Environment

It is recommended to use a virtual environment. To use a virtual
environment, follow these steps (Windows specific; activation of the
environment might differ).

``` shell
virtualenv --python=C:/Python310/python.exe venv # Use specific Python version for virtual environment
venv/Scripts/activate
```

Here is a Linux example that assumes you have Python `3.10` installed
(this may also require installing `python3.10-venv` and/or
`python3.10-dev`):

``` shell
python3.10 -m venv venv
source venv/bin/activate # Linux
```

### Installing Packages

The project is managed with `Poetry`. To install the required packages,
simply run the following.

``` shell
venv/Scripts/activate
# First install Poetry using pip:
(venv) C:\metrics-as-scores>pip install poetry
# Install the projects and its dependencies
(venv) C:\metrics-as-scores> poetry install
```

The same in Linux:

``` shell
source venv/bin/activate # Linux
(venv) ubuntu@vm:/tmp/metrics-as-scores$ pip install poetry
(venv) ubuntu@vm:/tmp/metrics-as-scores$ poetry install
```

### Running Tests

Tests are run using `poethepoet`:

``` shell
# Runs the tests and prints coverage
(venv) C:\metrics-as-scores>poe test
```

You can also generate coverage reports:

``` shell
# Writes reports to the local directory htmlcov
(venv) C:\metrics-as-scores>poe cov
```

------------------------------------------------------------------------

# Example Usage

*Metrics As Scores* can be thought of an *interactive*, *multiple-ANOVA*
analysis and explorer. The analysis of variance (ANOVA; John M. Chambers
(2017)) is usually used to analyze the differences among *hypothesized*
group means for a single *quantity*. An ANOVA might be used to estimate
the goodness-of-fit of a statistical model. Beyond ANOVA, `MAS` seeks to
answer the question of whether a sample of a certain quantity is more or
less common across groups. For each group, we can determine what might
constitute a common/ideal value, and how distant the sample is from that
value. This is expressed in terms of a percentile (a standardized scale
of `[0,1]`), which we call **score**.

## Software Metrics Example

Software metrics, when captured out of context, are meaningless (Gil and
Lalouche 2016). For example, typical values for complexity metrics are
vastly different, depending on the type of application. We find that,
for example, applications of type SDK have a much lower *expected*
complexity compared to Games (`1.9` vs. `3.1`) (Hönel et al. 2022).

Software metrics are often used in software quality models. However,
without knowledge of the application’s context (here: domain), the
deduced quality of these models is at least misleading, if not
completely off. This becomes apparent if we examine how an application’s
complexity scores across certain domains.

Since there are many software metrics that are captured simultaneously,
we can also compare domains in their entirety: How many metrics are
statistically significantly different from each other? Is there a set of
domains that are not distinguishable from each other? Are there metrics
that are always different across contexts and must be used with care?

This example is available as a downloadable dataset (Hönel 2023b). It is
based on software metrics and application domains of the “Qualitas.class
corpus” (Terra et al. 2013; Tempero et al. 2010).

## Diamonds Example

The diamonds dataset (Wickham 2016) holds prices of over 50,000 round
cut diamonds. It contains a number attributes for each diamond, such as
its price, length, depth, or weight. The dataset, however, features
three quality attributes: The quality of the cut, the clarity, and the
color. Suppose we are interested in examining properties of diamonds of
the highest quality only, across colors. Therefore, we select only those
diamonds from the dataset that have an *ideal* cut and the best (*IF*)
clarity. Now only the color quality gives a context to each diamonds and
its attributes.

This constellation now allows us to examine differences across
differently colored diamonds. For example, there are considerable
differences in price. We find that only the group of diamonds of the
best color is significantly different from the other groups. This
example is available as a downloadable dataset (Hönel 2023c).

------------------------------------------------------------------------

# Datasets

Metrics As Scores can use existing and own datasets. Please keep reading
to learn how.

## Use Your Own

Metrics As Scores has a built-in wizard that lets you import your own
dataset! There is another wizard that bundles your dataset so that it
can be shared with others. You may [**contribute your
dataset**](https://github.com/MrShoenel/metrics-as-scores/blob/master/CONTRIBUTING.md)
so we can add it to the curated list of known datasets (see next
section). If you do not have an own dataset, you can use the built-in
wizard to download any of the known datasets, too!

Note that Metrics As Scores supports you with all tools necessary to
create a publishable dataset. For example, it carries out the common
statistical tests:

- ANOVA (John M. Chambers 2017): Analysis of variance of your data
  across the available contexts.
- Tukey’s Honest Significance Test (TukeyHSD; Tukey (1949)): This test
  is used to gain insights into the results of an ANOVA test. While the
  former only allows obtaining the amount of corroboration for the null
  hypothesis, TukeyHSD performs all pairwise comparisons (for all
  possible combinations of any two contexts).
- Two-sample T-test: Compares the means of two samples to give an
  indication whether or not they appear to come from the same
  distribution. Again, this is useful for comparing contexts. Tukey’s
  test is used to gain insights into the results of an ANOVA test. While
  the former only allows obtaining the amount of corroboration for the
  null hypothesis, TukeyHSD performs all pairwise comparisons (for all
  possible combinations of any two contexts).

It also creates an **automatic report** based on these tests that you
can simply render into a PDF using Quarto.

A publishable dataset must contain parametric fits and pre-generated
densities (please check the wizard for these two). Metrics As Scores can
fit approximately **120** continuous and discrete random variables using
`Pymoo` (Blank and Deb 2020). Note that Metrics As Scores also
automatically carries out a number of goodness-of-fit tests. The type of
test also depends on the data (for example, not each test is valid for
discrete data, such as the KS two-sample test). These tests are then
used to select some best fitting random variable for display in the web
application.

- Cramér-von Mises- (Cramér 1928) and Kolmogorov–Smirnov one-sample
  (Stephens 1974) tests: After fitting a distribution, the sample is
  tested against the fitted parametric distribution. Since the fitted
  distribution cannot usually accommodate all of the sample’s
  subtleties, the test will indicate whether the fit is acceptable or
  not.
- Cramér-von Mises- (Anderson 1962), Kolmogorov–Smirnov-, and
  Epps–Singleton (Epps and Singleton 1986) two-sample tests: After
  fitting, we create a second sample by uniformly sampling from the
  `PPF`. Then, both samples can be used in these tests. The
  Epps–Singleton test is also applicable for discrete distributions.

## Known Datasets

The following is a curated list of known, publicly available datasets
that can be used with Metrics As Scores. These datasets can be
downloaded using the text-based user interface.

- Metrics and Domains From the Qualitas.class corpus (Hönel 2023b). 10
  GB. <https://doi.org/10.5281/zenodo.7633949>.
- ELISA Spectrophotometer Samples (Hönel 2023a). 266 MB.
  <https://doi.org/10.5281/zenodo.7633989>.
- Price, weight, and other properties of over 1,200 ideal-cut and
  best-clarity diamonds (Hönel 2023c). 508 MB.
  <https://doi.org/10.5281/zenodo.7647596>.

------------------------------------------------------------------------

# Personalizing the Web Application

The web application *“[Metrics As
Scores](https://metrics-as-scores.ml/)”* is located in the directory
[`src/metrics_as_scores/webapp/`](https://github.com/MrShoenel/metrics-as-scores/blob/master/src/metrics_as_scores/webapp/).
The app itself has three vertical blocks: a header, the interactive
part, and a footer. Header and footer can be easily edited by modifing
the files
[`src/metrics_as_scores/webapp/header.html`](https://github.com/MrShoenel/metrics-as-scores/blob/master/src/metrics_as_scores/webapp/header.html)
and
[`src/metrics_as_scores/webapp/footer.html`](https://github.com/MrShoenel/metrics-as-scores/blob/master/src/metrics_as_scores/webapp/footer.html).

Note that when you create your own dataset, you get to add sections to
the header and footer using two HTML fragments. This is recommended over
modifying the web application directly.

If you want to change the title of the application, you will have to
modify the file
[`src/metrics_as_scores/webapp/main.py`](https://github.com/MrShoenel/metrics-as-scores/blob/master/src/metrics_as_scores/webapp/main.py)
at the very end:

``` python
# Change this line to your desired title.
curdoc().title = "Metrics As Scores"
```

**Important**: If you modify the web application, you must always
maintain two links: one to <https://metrics-as-scores.ml/> and one to
this repository, that is,
<https://github.com/MrShoenel/metrics-as-scores>.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-anderson1962" class="csl-entry">

Anderson, T. W. 1962. “<span class="nocase">On the Distribution of the
Two-Sample Cramer-von Mises Criterion</span>.” *The Annals of
Mathematical Statistics* 33 (3): 1148–59.
<https://doi.org/10.1214/aoms/1177704477>.

</div>

<div id="ref-pymoo2020" class="csl-entry">

Blank, Julian, and Kalyanmoy Deb. 2020. “Pymoo: Multi-Objective
Optimization in Python.” *IEEE Access* 8: 89497–509.
<https://doi.org/10.1109/ACCESS.2020.2990567>.

</div>

<div id="ref-cramer1928" class="csl-entry">

Cramér, Harald. 1928. “On the Composition of Elementary Errors.”
*Scandinavian Actuarial Journal* 1928 (1): 13–74.
<https://doi.org/10.1080/03461238.1928.10416862>.

</div>

<div id="ref-epps1986" class="csl-entry">

Epps, T. W., and Kenneth J. Singleton. 1986. “An Omnibus Test for the
Two-Sample Problem Using the Empirical Characteristic Function.”
*Journal of Statistical Computation and Simulation* 26 (3-4): 177–203.
<https://doi.org/10.1080/00949658608810963>.

</div>

<div id="ref-gil2016software" class="csl-entry">

Gil, Joseph Yossi, and Gal Lalouche. 2016. “When Do Software Complexity
Metrics Mean Nothing? - When Examined Out of Context.” *J. Object
Technol.* 15 (1): 2:1–25. <https://doi.org/10.5381/jot.2016.15.5.a2>.

</div>

<div id="ref-dataset_elisa" class="csl-entry">

Hönel, Sebastian. 2023a. “Metrics As Scores Dataset: Elisa
Spectrophotometer Positive Samples.” Zenodo.
<https://doi.org/10.5281/zenodo.7633989>.

</div>

<div id="ref-dataset_qcc" class="csl-entry">

———. 2023b. “<span class="nocase">Metrics As Scores Dataset: Metrics and
Domains From the Qualitas.class corpus</span>.” Zenodo.
<https://doi.org/10.5281/zenodo.7633949>.

</div>

<div id="ref-dataset_diamonds-ideal-if" class="csl-entry">

———. 2023c. “<span class="nocase">Metrics As Scores Dataset: Price,
weight, and other properties of over 1,200 ideal-cut and best- clarity
diamonds</span>.” Zenodo. <https://doi.org/10.5281/zenodo.7647596>.

</div>

<div id="ref-honel2022qrs" class="csl-entry">

Hönel, Sebastian, Morgan Ericsson, Welf Löwe, and Anna Wingkvist. 2022.
“Contextual Operationalization of Metrics as Scores: Is My Metric Value
Good?” In *22nd IEEE International Conference on Software Quality,
Reliability and Security, QRS 2022, Guangzhou, China, December 5-9,
2022*, 333–43. IEEE. <https://doi.org/10.1109/QRS57517.2022.00042>.

</div>

<div id="ref-chambers2017" class="csl-entry">

John M. Chambers, Richard M. Heiberger, Anne E. Freeny. 2017. “Analysis
of Variance; Designed Experiments.” In *Statistical Models in S*, edited
by John M. Chambers and Trevor J. Hastie, 1st ed. Routledge.
<https://doi.org/10.1201/9780203738535>.

</div>

<div id="ref-stephens1974" class="csl-entry">

Stephens, M. A. 1974. “EDF Statistics for Goodness of Fit and Some
Comparisons.” *Journal of the American Statistical Association* 69
(347): 730–37. <https://doi.org/10.1080/01621459.1974.10480196>.

</div>

<div id="ref-tempero2010qualitas" class="csl-entry">

Tempero, Ewan D., Craig Anslow, Jens Dietrich, Ted Han, Jing Li, Markus
Lumpe, Hayden Melton, and James Noble. 2010. “The Qualitas Corpus: A
Curated Collection of Java Code for Empirical Studies.” In *17th Asia
Pacific Software Engineering Conference, APSEC 2010, Sydney, Australia,
November 30 - December 3, 2010*, edited by Jun Han and Tran Dan Thu,
336–45. IEEE Computer Society. <https://doi.org/10.1109/APSEC.2010.46>.

</div>

<div id="ref-terra2013qcc" class="csl-entry">

Terra, Ricardo, Luis Fernando Miranda, Marco Túlio Valente, and Roberto
da Silva Bigonha. 2013. “Qualitas.class Corpus: A Compiled Version of
the Qualitas Corpus.” *ACM SIGSOFT Softw. Eng. Notes* 38 (5): 1–4.
<https://doi.org/10.1145/2507288.2507314>.

</div>

<div id="ref-tukey1949anova" class="csl-entry">

Tukey, John W. 1949. “Comparing Individual Means in the Analysis of
Variance.” *Biometrics* 5 (2): 99–114.
<http://www.jstor.org/stable/3001913>.

</div>

<div id="ref-ggplot2" class="csl-entry">

Wickham, Hadley. 2016. *<span class="nocase">ggplot2</span>: Elegant
Graphics for Data Analysis*. Springer-Verlag New York.
<https://ggplot2.tidyverse.org>.

</div>

</div>
