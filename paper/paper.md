---
affiliations: 
  - 
    index: 1
    name: "Department of Computer Science and Media Technology, Linnaeus University, Sweden"
authors: 
  - 
    affiliation: 1
    name: "Sebastian Hönel^[corresponding author]"
    orcid: 0000-0001-7937-1645
  - 
    affiliation: 1
    equal-contrib: false
    name: "Morgan Ericsson"
    orcid: 0000-0003-1173-5187
  - 
    affiliation: 1
    equal-contrib: false
    name: "Welf Löwe"
    orcid: 0000-0002-7565-3714
  - 
    affiliation: 1
    equal-contrib: false
    name: "Anna Wingkvist"
    orcid: 0000-0002-0835-823X
bibliography: refs.bib
date: "29 September 2022"
tags: 
  - Python
  - "Multiple ANOVA"
  - "Distribution fitting"
  - "Inverse sampling"
  - "Empirical distributions"
  - "Kernel density estimation"
title: "Metrics As Scores: A Tool- and Analysis Suite and Interactive Application for Exploring Context-Dependent Distributions"
---


\newcommand{\mas}{\textsf{MAS}\xspace}
\newcommand\tightto{\!\to\!}
\newcommand\tightmapsto{\!\mapsto\!}
\newcommand{\tight}[1]{\,{#1}\,}
\newcommand{\utight}[1]{{#1}\,}


# Summary
<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->
<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
<!-- Short introduction into the problem, then -->

\emph{Metrics As Scores} can be thought of as an interactive, \emph{multiple} analysis of variance [abbr. "ANOVA", @chambers2017statistical].
An ANOVA might be used to estimate the *goodness-of-fit* of a statistical model.
Beyond ANOVA, which is used to analyze the differences among hypothesized group means for a single quantity (feature), Metrics As Scores seeks to answer the question of whether a sample of a certain feature is more or less common across groups.
This approach to data visualization and -exploration has been used previously [e.g., @Jiang2022anova].
Beyond this, Metrics As Scores can determine what might constitute a good/bad, acceptable/alarming, or common/extreme value, and how distant the sample is from that value, for each group.
This is expressed in terms of a percentile (a standardized scale of $\left[0,1\right]$), which we call ***score***.
Considering all available features among the existing groups furthermore allows the user to assess how different the groups are from each other, or whether they are indistinguishable from one another.


The name \emph{Metrics As Scores} was derived from its initial application: examining differences of software metrics across application domains [@honel2022mas].
A software metric is an aggregation of one or more raw features according to some well-defined standard, method, or calculation.
In software processes, such aggregations are often counts of events or certain properties [@carleton1999].
However, without the aggregation that is done in a quality model, raw data (samples) and software metrics are rarely of great value to analysts and decision-makers. This is because quality models are conceived to establish a connection between software metrics and certain quality goals [@kaner2004software].
It is, therefore, difficult to answer the question "is my metric value good?".


With Metrics As Scores we present an approach that, given some \emph{ideal} value, can transform any sample into a score, given a sample of sufficiently many relevant values.
While such ideal values for software metrics were previously attempted to be derived from, e.g., experience or surveys [@benlarbi2000thresh], benchmarks [@alves2010thresh], or by setting practical values [@grady1992practical], with Metrics As Scores we suggest deriving ideal values additionally in non-parametric, statistical ways.
To do so, data first needs to be captured in a \emph{relevant} context (group).
A feature value might be good in one context, while it is less so in another.
Therefore, we suggest generalizing and contextualizing the approach taken by @UlanLEW21, in which a score is defined to always have a range of $[0,1]$ and linear behavior.
This means that scores can now also be compared and that a fixed increment in any score is equally valuable among scores.
This is not the case for raw features, otherwise.


Metrics As Scores consists of a tool- and analysis suite and an interactive application that allows researchers to explore and understand differences in scores across groups.
The operationalization of features as scores lies in gathering values that are context-specific (group-typical), determining an ideal value non-parametrically or by user preference, and then transforming the observed values into distances.
Metrics As Scores enables this procedure by unifying the way of obtaining probability densities/masses and conducting appropriate statistical tests.
More than $120$ different parametric distributions (approx. $20$ of which are discrete) are fitted through a common interface.
Those distributions are part of the `scipy` package for the Python programming language, which Metrics As Scores makes extensive use of [@scipy].
While fitting continuous distributions is straightforward using maximum likelihood estimation, many discrete distributions have integral parameters. For these, Metrics As Scores solves a mixed-variable global optimization problem using a genetic algorithm in `pymoo` [@pymoo].
Additionally to that, empirical distributions (continuous and discrete) and smooth approximate kernel density estimates are available. Applicable statistical tests for assessing the goodness-of-fit are automatically performed.
<!-- -->
These tests are used to select some best-fitting random variable in the interactive web application.
As an application written in Python, Metrics As Scores is made available as a package that is installable using the Python Package Index (PyPI): `pip install metrics-as-scores`.
As such, the application can be used in a stand-alone manner and does not require additional packages, such as a web server or third-party libraries.





# Statement Of Need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

Metrics As Scores is a supplement to existing analyses that enables the exploration of differences a  mong groups in a novel, mostly interactive way.
Raw features are seldomly useful as, e.g., indicators of quality.
Only the transformation to scores enables an apples-to-apples comparison of different quantities (features) across contexts (groups).
This is particularly true for software metrics, which often cannot be compared directly, because due to their different scales and distributions, there does not exist a mathematically sound way to do so [@ulan2018jointprobs].
<!-- -->
While some have attempted to associate blank software metrics with quality [e.g., @basili1996validation], most often applications have to resort to using software metrics as, e.g., fault indicators [@caulo2019metricsfault; @aziz2019metrics], or as indicators of reliability and complexity [@chidamber1994metrics].
<!-- -->
Furthermore, none of the existing approaches that attempted to associate software metrics with quality paid great attention to the fact that software metrics have different distributions and, therefore, different statistical properties across application domains.
Therefore, the operationalization of software metrics as scores ought to be conditional on the application domain.



# MAS -- The Tool- and Analysis Suite
<!-- Here, we go into detail about distribution fitting and statistical tests. -->
The main purpose of the Metrics As Scores tool- and analysis suite for Python is to approximate or estimate, enable the exploration of, and sample from context-dependent distributions.
Three principal types of distributions are supported: empirical and parametric (both continuous and discrete), as well as kernel density estimates.
These are all unified using the class `Density`, which provides access to the probability density/mass function (PDF/PMF), the cumulative distribution function (CDF) and its complement (CCDF) for scores, and the percent point function (PPF).
As a unified representation for all these we choose line plots, as these are most commonly used for continuous densities.
Instead of, e.g., histograms for discrete data, the plotting will fall back to using step-wise linear functions.
<!--
-->
Metrics As Scores carries out a number of statistical tests for fitted distributions.
The results for each test are stored in a separate spreadsheet after the fitting process and may be used to further investigate how well certain distributions fit and what the alternatives are.
The carried out tests are: Cramér--von Mises [@Cramr1928] and Kolmogorov--Smirnov one-sample [@Stephens1974] tests, Cramér--von Mises [@Anderson1962], Kolmogorov--Smirnov, and Epps--Singleton [@Epps1986] two-sample tests.
The second sample required for the two-sample test is obtained by uniformly sampling from the fitted distribution's PPF.
The best-fitting distribution is selected for pre-generating densities that are used by the web application, such that only the single best fit is used for visualization.
The Epps--Singleton two-sample test is compatible with discrete data and is used for discrete distributions.
For continuous data, the one-sample Kolmogorov--Smirnov test is used.


Metrics As Scores supports the transformation of samples into distances using ideal values that are computed non-parametrically.
Given a sample $X$ from an arbitrary population and an ideal value $i_X$, the corresponding distance, $D$, is obtained as $D=\lvert X-i_X \rvert$.
In order to obtain a discrete ideal value (e.g., when transforming a discrete sample in order to fit a discrete probability distribution), the expectation (mean), median, infimum, and supremum can be obtained in a straightforward way and then rounded.
A discrete value for the mode (most common value) is determined using `scipy`.
When a continuous ideal value is required, we first estimate a kernel density $f_{\mathcal{X}}$ using a Gaussian kernel.
Then, the expectation is obtained as $\mathbb{E}\left[\mathcal{X}\right]\tight{=}\int_{-\infty}^{\infty}\,t\,f_{\mathcal{X}}(t)\,dt$.
The mode of a sample $X$ is obtained by solving $\hat{x}\tight{=}\mathrm{arg\,\!max}_{x\tight{\in}\mathcal{X}}\,f_{\mathcal{X}}(x)$.
In order to approximate the median, we obtain a large sample from the kernel density and compute its median using `numpy` [@numpy].


In order to understand whether or not the available groups in the data matter before obtaining any of these distributions, Metrics As Scores supports additional tools for generating and outputting results for three other statistical tests.
The ANOVA test is used to analyze differences among sample means (which, e.g., stem from the same feature in different groups).
Tukey's Honest Significance Test [abbr. "TukeyHSD", @Tukey1949] is used to gain insights into the results of an ANOVA test.
While the former only allows obtaining the amount of corroboration for the null hypothesis, TukeyHSD performs all pairwise comparisons (for all possible combinations of any two groups).
Lastly, Welch's two-sample t-test (which does not assume equal population variances) compares the means of two samples to give an indication of whether or not they appear to come from the same distribution [@welch1947].


Metrics As Scores includes a scientific template for generating a report for a dataset that exploits the results of these analyses [for example, see @dataset_qcc].
Users are encouraged to import their own datasets and have Metrics As Scores conduct all necessary analyses, generate a report, and bundle a publishable dataset.
The application comes with a rich text-based user interface, which offers wizards that afford completely code-free interactions.
These interactions include, for example, showing installed datasets, downloading of known datasets from a curated list, creating own datasets, automatically attempting to fit more than $120$ random variables, report creation and bundling of own datasets, pre-generating densities for the interactive web application, and running the web application with a locally available dataset.



# MAS -- The Interactive Application
<!-- Here, we will introduce the actual application. Also, the application hosted under https://metrics-as-scores.ml/ is *an* actual application of the QCC. -->

![Main plot area of the application "Metrics As Scores". Using the Qualitas.class corpus, software metrics values of own applications can be scored against the corpus' groups (application domains). Shown are the CCDFs (scores) of the fitted parametric distributions for the metric NBD transformed using the domain-specific expectation as ideal value.](MAS.png){#fig:mas}

The interactive application is partially shown in Figure \ref{fig:mas}. Not shown are the header, UI controls, a tabular with numerical data for the current selection, and the footer which contains help.
The application supports all transforms, continuous and discrete distributions, obtaining scores for own features/sampling from inverse CDFs (PPFs), and grouping of features into discrete/continuous.
The main tool, the plot, allows the user to zoom, pan, select, enable/disable contexts, and manually hover the graphs to obtain precise $x$/$y$-values.
The web application can be launched with any of the available datasets (manually created or downloaded).
The interactive application is built using Bokeh and facilitates customization using a few steps described in the software's manual [@bokeh].




<!-- # Related Work
<!-- We will not have this section unless otherwise requested, because the text is already richely interspersed with references where appropriate. -->
<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->



# Applications

The Metrics As Scores tool- and analysis suite and interactive application have initially been used to study the "Qualitas.class corpus" of software metrics [@terra2013qualitas].
The results of studying the software metrics of the corpus show that, for example, the context (application domain) software metrics were captured in is always of importance and must not be neglected.
In addition, some of the software metrics in the corpus are \emph{never} similar across application domains and must be applied with great care when used in quality models [@honel2022mas].
Evidently, the approach offered by Metrics As Scores enables not only to examine and compare samples but also the contexts these are embedded in as a whole.

Metrics As Scores has since been extended to work with almost arbitrary datasets.
Three well-known datasets have been added: the Iris flower dataset [@dataset_iris], the Diamonds dataset [@dataset_diamonds-ideal-if], and the Elisa Spectrophotometer Positive Samples dataset [@dataset_elisa].
While these datasets are well understood, Metrics As Scores can reveal additional insights.
For example, visual inspection of the Iris flower dataset shows that the probability densities for the features of flower petals do only overlap somewhat or not at all across the three species *setosa*, *versicolor*, and *virginica*.
This is corroborated by the generated report which confirms that these features are not statistically significantly similar across species.







# Related Work

Metrics As Scores finds itself among other visualization tools related to statistical analysis and learning.
Some existing tools support a visual and interactive approach to exploring the results of an ANOVA.
In @sturm2005anova, the goal is to enable a what-if analysis by allowing the user to assume arbitrary groups in the data.
@fox2009hyptests provide a package for `R` to visually test hypotheses of linear models (as is ANOVA).
A number of packages for creating non-interactive ANOVA visualizations exists [e.g., @rpkg2023granova].
To the best of our knowledge, however, Metrics As Scores is the first application to enable the interactive exploration of differences among groups.
It appears that it is also the first tool to enable the transformation of samples into scores and to produce and aggregate group-related results derived from these.
CorpusViz by @slater2019corpusviz is a tool that exclusively targets the Qualitas corpus [@tempero2010qualitas], *not* the Qualitas.class corpus.
CorpusViz attempts to satisfy the three primary requirements of composite viewing of multiple visualizations, the ability to change between software systems and versions, as well as allowing the user to configure the visualizations.




# Acknowledgments
<!-- Acknowledgement of any financial support. -->

The authors would like to sincerely express their gratitude towards the reviewers of the Journal of Open Source Software for their invaluable comments.

This work is supported by the [Linnaeus University Centre for Data Intensive Sciences and Applications (DISA)](https://lnu.se/forskning/sok-forskning/linnaeus-university-centre-for-data-intensive-sciences-and-applications) High-Performance Computing Center.


# References