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
Raw data can be aggregated according to some well-defined standard, method, or calculation, which makes it a metric.
In software processes, such aggregations are often counts of events or certain properties [@carleton1999].
However, without the aggregation that is done in a quality model, raw data and metrics are rarely of great value to analysts and decision-makers. This is because quality models are conceived to establish a connection between metrics and certain quality goals [@kaner2004software].
It is, therefore, difficult to answer the question ``is my metric value good?''.
With \emph{Metrics As Scores} (\mas) we present an approach that, given some \emph{ideal} value, can transform any metric into a score, given an observation of sufficiently many relevant metrics values.
While such ideal values were previously attempted to be derived from, e.g., experience or surveys [@benlarbi2000thresh], benchmarks [@alves2010thresh], or by setting practical values [@grady1992practical], with \mas we suggest deriving ideal values additionally in non-parametric, statistical ways.
To do so, data first needs to be captured in a \emph{relevant} context. A metric value might be good in one context, while it is less so in another.
Therefore, we suggest generalizing and contextualizing the approach taken by [@UlanLEW21], in which a score is defined to always have a range of $[0,1]$ and linear behavior.
This means that scores can now also be compared and that a unit change in any score is equally valuable among scores.
This is not the case for raw data or metrics.


\mas consists of a tool- and analysis suite and an interactive application that allows researchers to explore and understand differences in scores across contexts.
The operationalization of raw data or metrics as scores lies in gathering values that are context-specific (usual), determining an ideal value non-parametrically or by user preference, and then transforming the observed values into distances.
\mas enables this procedure by unifying the way of obtaining probability densities/masses and conducting appropriate statistical tests.
More than $120$ different parametric distributions (approx. $20$ of which are discrete) are fitted through a common interface.
While fitting continuous distributions is straightforward using maximum likelihood estimation, many discrete distributions have integral parameters. For these, \mas solves a mixed-variable global optimization problem using a genetic algorithm and Pymoo [@pymoo].
Additionally to that, empirical distributions (continuous and discrete) and smooth approximate Kernel density estimates are available. Applicable statistical tests, such as the Cramér--von Mises- or Epps--Singleton-tests, are automatically performed.
\mas was created for and includes the ``Qualitas.class'' corpus of software metrics [@terra2013qualitas].
However, we include a guide for using own datasets. By following three (two) steps (one is optional), own data formatted as plain CSV files can be imported and leveraged.



# Statement Of Need
<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->
Without the transformation to scores, raw data and metrics are seldomly useful as indicators of quality.
While some have attempted to associate blank metrics with quality (e.g, @basili1996validation), most often applications have to resort to using metrics as, e.g., fault indicators [@caulo2019metricsfault; @aziz2019metrics], or as indicators fo reliability and complexity [@chidamber1994metrics].
Most often metrics cannot be compared directly, because due to their different scales and distributions, there does not exist a mathematical sound way to do so [@ulan2018jointprobs].
Furthermore, none of the existing approaches that attempted to associate metrics with quality paid great attention to the fact that metrics have different distributions and therefore different statistical properties across contexts.
Therefore, the operationalization of metrics as scores ought to be conditional on the context.
The results of studying the metrics of the Qualitas.class corpus show that the context metrics were captured in is always of importance and must not be neglected.
In addition, some of the metrics in the corpus are \emph{never} similar across contexts and must be applied with great care when used in quality models. [@honel2022mas].
The \mas approach enables these and similar insights and supports decision-makers in selecting and comparing scores.


# MAS -- The Tool- and Analysis Suite
<!-- Here, we go into detail about distribution fitting and statistical tests. -->
The main purpose of the \mas tool- and analysis suite for Python is to approximate or estimate, and to enable the exploration of context-dependent distributions.
Three principal types of distributions are supported: Empirical and Parametric (both continuous and discrete), as well as Kernel density estimates.
These are all unified using the class `Density`, which provides access to the PDF/PMF, CDF/CCDF (for scores), and the PPF.
When obtaining any of these types of distributions for a univariate sample, the following statistical tests are carried out automatically (if applicable): Cramér--von Mises- [@Cramr1928] and Kolmogorov--Smirnov one-sample [@Stephens1974] tests, Cramér-von Mises- [@Anderson1962], Kolmogorov–Smirnov-, and Epps–Singleton [@Epps1986] two-sample tests.
Currently, the following non-parametric transforms are supported for all distributions: The expectation $\mathbb{E}\left[\mathcal{X}\right]\tight{=}\int_{-\infty}^{\infty}\,x\,f_{\mathcal{X}}(x)\,dx$, the mode $\hat{x}\tight{=}\mathrm{arg\,\!max}_{x\tight{\in}\mathcal{X}}\,f_{\mathcal{X}}(x)$, the median, and observed infimum/supremum.
The latter two transforms are useful for attaching an explicit meaning of smaller/larger is better.

In order to understand whether or not the available contexts in the data matter before obtaining any of these distributions, \mas supports additional tools for generating and outputting results for three other statistical tests.
The analysis of variance (ANOVA) test is used to analyze differences among sample means (which, e.g., stem from the same metric in different contexts) [@chambers2017statistical].
Tukey's Honest Significance Test (TukeyHSD) is used to gain insights into the results of an ANOVA test. While the former only allows obtaining the amount of corroboration for the null hypothesis, TukeyHSD performs all pairwise comparisons (for all possible combinations of any two contexts) [@Tukey1949].
Lastly, the two-sample T-test compares the means of two samples to give an indication of whether or not they appear to come from the same distribution.



# MAS -- The Interactive Application
<!-- Here, we will introduce the actual application. Also, the application hosted under https://metrics-as-scores.ml/ is *an* actual application of the QCC. -->

![Main plot area of the application ``Metrics As Scores''. Using the Qualitas.class corpus, metrics values of own applications can be scored against the corpus' domains. Shown are the CCDFs (scores) of the fitted parametric distributions for the metric TLOC transformed using the infimum (per domain). Available online: <https://metrics-as-scores.ml/>.](MAS.png){#fig:mas}

The interactive application is partially shown in Figure \ref{fig:mas}. Not shown are the header, UI controls, a tabular with numerical data for the current selection, and the footer which contains help.
The application supports all transforms, continuous and discrete distributions, obtaining scores for own metrics/sampling from inverse CDFs (PPFs), and grouping of metrics into discrete/continuous.
The main tool, the plot, allows the user to zoom, pan, select, enable/disable contexts, and manually hover the graphs to obtain precise $x$/$y$-values.
The interactive application is built using Bokeh [@bokeh] and allows for customization using a few steps described in the software's manual.




<!-- # Related Work
<!-- We will not have this section unless otherwise requested, because the text is already richely interspersed with references where appropriate. -->
<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->


# Acknowledgments
<!-- Acknowledgement of any financial support. -->
This work is supported by the [Linnaeus University Centre for Data Intensive Sciences and Applications (DISA)](https://lnu.se/forskning/sok-forskning/linnaeus-university-centre-for-data-intensive-sciences-and-applications) High-Performance Computing Center.


# Applications

The \mas tool- and analysis suite and interactive application have been used to study the Qualitas.class corpus of software metrics [@honel2022mas].



# References