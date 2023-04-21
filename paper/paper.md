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
Beyond ANOVA, which is used to analyze the differences among hypothesized group means for a single feature, Metrics As Scores seeks to answer the question of whether a sample of a certain quantity (feature) is more or less common across groups.
For each group, we can determine what might constitute a good/bad, acceptable/alarming, or common/extreme value, and how distant the sample is from that value.
This is expressed in terms of a percentile (a standardized scale of $\left[0,1\right]$), which we call \textbf{\emph{score}}.
Considering all available features among the existing groups furthermore allows to assess how different the groups are from each other, or whether they are indistinguishable from one another.


The name \emph{Metrics As Scores} was derived from its initial application: Examining differences of software metrics across application domains [@honel2022mas].
A software metric is an aggregation of a raw quantity according to some well-defined standard, method, or calculation.
In software processes, such aggregations are often counts of events or certain properties [@carleton1999].
However, without the aggregation that is done in a quality model, raw data and software metrics are rarely of great value to analysts and decision-makers. This is because quality models are conceived to establish a connection between software metrics and certain quality goals [@kaner2004software].
It is, therefore, difficult to answer the question "is my metric value good?".


With Metrics As Scores we present an approach that, given some \emph{ideal} value, can transform any feature into a score, given an observation of sufficiently many relevant values.
While such ideal values for software metrics were previously attempted to be derived from, e.g., experience or surveys [@benlarbi2000thresh], benchmarks [@alves2010thresh], or by setting practical values [@grady1992practical], with Metrics As Scores we suggest deriving ideal values additionally in non-parametric, statistical ways.
To do so, data first needs to be captured in a \emph{relevant} context (group).
A feature value might be good in one context, while it is less so in another.
Therefore, we suggest generalizing and contextualizing the approach taken by @UlanLEW21, in which a score is defined to always have a range of $[0,1]$ and linear behavior.
This means that scores can now also be compared and that a unit change in any score is equally valuable among scores.
This is not the case for raw features, otherwise.


Metrics As Scores consists of a tool- and analysis suite and an interactive application that allows researchers to explore and understand differences in scores across groups.
The operationalization of raw data or features as scores lies in gathering values that are context-specific (group-typical), determining an ideal value non-parametrically or by user preference, and then transforming the observed values into distances.
Metrics As Scores enables this procedure by unifying the way of obtaining probability densities/masses and conducting appropriate statistical tests.
More than $120$ different parametric distributions (approx. $20$ of which are discrete) are fitted through a common interface.
Those distributions are part of the `scipy` package for the Python programming language, which Metrics As Scores makes extensive use of [@scipy].
While fitting continuous distributions is straightforward using maximum likelihood estimation, many discrete distributions have integral parameters. For these, Metrics As Scores solves a mixed-variable global optimization problem using a genetic algorithm and Pymoo [@pymoo].
Additionally to that, empirical distributions (continuous and discrete) and smooth approximate kernel density estimates are available. Applicable statistical tests for assessing the goodness-of-fit are automatically performed.
<!-- -->
These tests are used to select some best-fitting random variable in the interactive web application.





# Statement Of Need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

Metrics As Scores is a supplement to existing analyses that enables the exploration of differences among groups in a novel, mostly interactive way.
Raw features are seldomly useful as, e.g., indicators of quality.
Only the transformation to scores allows for an apples-to-apples comparison of different quantities (features) across contexts (groups).
This is particularly true for software metrics, which often cannot be compared directly, because due to their different scales and distributions, there does not exist a mathematical sound way to do so [@ulan2018jointprobs].
<!-- -->
While some have attempted to associate blank software metrics with quality (e.g, @basili1996validation), most often applications have to resort to using software metrics as, e.g., fault indicators [@caulo2019metricsfault; @aziz2019metrics], or as indicators of reliability and complexity [@chidamber1994metrics].
<!-- -->
Furthermore, none of the existing approaches that attempted to associate software metrics with quality paid great attention to the fact that software metrics have different distributions and, therefore, different statistical properties across application domains.
Therefore, the operationalization of software metrics as scores ought to be conditional on the application domain.



# MAS -- The Tool- and Analysis Suite
<!-- Here, we go into detail about distribution fitting and statistical tests. -->
The main purpose of the Metrics As Scores tool- and analysis suite for Python is to approximate or estimate, enable the exploration of, and sample from context-dependent distributions.
Three principal types of distributions are supported: Empirical and Parametric (both continuous and discrete), as well as kernel density estimates.
These are all unified using the class `Density`, which provides access to the PDF/PMF, CDF/CCDF (for scores), and the PPF.
<!--
-->
Metrics As Scores carries out a number of statistical tests for fitted distributions.
The results for each test are stored in a separate spreadsheet after the fitting process and may be used to further investigate how well certain distributions fit and what the alternatives may are.
The carried out tests are: Cramér--von Mises [@Cramr1928] and Kolmogorov--Smirnov one-sample [@Stephens1974] tests, Cramér--von Mises [@Anderson1962], Kolmogorov--Smirnov, and Epps--Singleton [@Epps1986] two-sample tests.
The second sample required for the two-sample test is obtained by uniformly sampling from the fitted distribution's PPF.
The best-fitting distribution is selected for pre-generating densities that are used by the web application, such that only the single best fit is used for visualization.
The Epps--Singleton two-sample test is compatible with discrete data and is used for discrete distributions.
For continuous data, the one-sample Kolmogorov--Smirnov test is used.


Metrics As Scores supports to transform samples into distances using ideal values that are computed non-parametrically.
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
The application comes with a rich text-based user interface, which offers wizards that allow for completely code-free interactions.
These interactions include, for example, showing installed datasets, downloading of known datasets from a curated list, creating own datasets, automatically attempting to fit more than $120$ random variables, report creation and bundling of own datasets, pre-generating densities for the interactive web application, and running the web application with a locally available dataset.



# MAS -- The Interactive Application
<!-- Here, we will introduce the actual application. Also, the application hosted under https://metrics-as-scores.ml/ is *an* actual application of the QCC. -->

![Main plot area of the application "Metrics As Scores". Using the Qualitas.class corpus, software metrics values of own applications can be scored against the corpus' groups (application domains). Shown are the CCDFs (scores) of the fitted parametric distributions for the metric TLOC transformed using the infimum (per domain). Available online: <https://metrics-as-scores.ml/>.](MAS.png){#fig:mas}

The interactive application is partially shown in Figure \ref{fig:mas}. Not shown are the header, UI controls, a tabular with numerical data for the current selection, and the footer which contains help.
The application supports all transforms, continuous and discrete distributions, obtaining scores for own features/sampling from inverse CDFs (PPFs), and grouping of features into discrete/continuous.
The main tool, the plot, allows the user to zoom, pan, select, enable/disable contexts, and manually hover the graphs to obtain precise $x$/$y$-values.
The web application can be launched with any of the available datasets (manually created or downloaded).
The interactive application is built using Bokeh and allows for customization using a few steps described in the software's manual [@bokeh].




<!-- # Related Work
<!-- We will not have this section unless otherwise requested, because the text is already richely interspersed with references where appropriate. -->
<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->



# Applications

The Metrics As Scores tool- and analysis suite and interactive application have been used to study the "Qualitas.class corpus" of software metrics [@terra2013qualitas].
The results of studying the software metrics of the corpus show that the context (application domain) software metrics were captured in is always of importance and must not be neglected.
In addition, some of the software metrics in the corpus are \emph{never} similar across application domains and must be applied with great care when used in quality models [@honel2022mas].
The Metrics As Scores-approach enables these and similar insights and supports decision-makers in selecting and comparing scores.



# Acknowledgments
<!-- Acknowledgement of any financial support. -->

The authors would like to sincerely express their gratitude towards the reviewers of the Journal of Open Source Software for their invaluable comments.

This work is supported by the [Linnaeus University Centre for Data Intensive Sciences and Applications (DISA)](https://lnu.se/forskning/sok-forskning/linnaeus-university-centre-for-data-intensive-sciences-and-applications) High-Performance Computing Center.


# References