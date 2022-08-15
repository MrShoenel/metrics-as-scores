Files
======

The original [`metrics.zip`](http://web.archive.org/web/20220814110913/http://java.labsoft.dcc.ufmg.br/qualitas.class/corpus/metrics.zip) (84.2mb) from [http://java.labsoft.dcc.ufmg.br/qualitas.class/download.html](https://web.archive.org/web/20191223234321/http://java.labsoft.dcc.ufmg.br/qualitas.class/download.html) was repacked into `metrics.7z` (30.1mb) using PPMd.

The metrics values from all these files (systems/projects) have been previously extracted into separate CSV files and merged into one large file. Those are stored under [`../csv/metrics.csv.7z`](../csv/metrics.csv.7z) (496kb).

You can, however, extract your own metrics using methods of the class `QualitasCorpusMetricsExtractor`. The merged file from above does not retain any other information than the system, the metric, and the value, because the primary purpose of this repository is to approximate distributions.
