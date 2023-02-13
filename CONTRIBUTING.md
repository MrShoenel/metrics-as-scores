Contributing to Metrics As Scores
=================================

- [Contributing to Metrics As Scores](#contributing-to-metrics-as-scores)
- [1. How To Contribute to *Metrics As Scores*](#1-how-to-contribute-to-metrics-as-scores)
	- [1.1. Contributions of New Datasets](#11-contributions-of-new-datasets)
		- [1.1.1. Registering Your Dataset](#111-registering-your-dataset)
	- [1.2. Contribute to the Software](#12-contribute-to-the-software)


# 1. How To Contribute to *Metrics As Scores*


Metrics As Scores is open source, and the datasets it can be used with are open access.
Therefore, we welcome all contributions, but first and foremost we welcome additional datasets.


## 1.1. Contributions of New Datasets

Any new dataset must be created using the Text User Interface (TUI) that comes with Metrics As Scores.
It ensures compatibility with the application.
After installing Metrics As Scores, simply type

```{shell}
> mas
```
at a prompt to bring up the interface.


You should then upload your dataset at, e.g., Zenodo.
Have a look at the [Qualitas.class corpus dataset](https://doi.org/10.5281/zenodo.7633950) to get an idea.
The `mas` command will help you create a bundle that can be uploaded.

### 1.1.1. Registering Your Dataset

Lastly, if you want to make available your dataset to others by having it added to the [`known-datasets`](/metrics-as-scores/blob/master/datasets/known-datasets.json), open an issue and ask for inclusion.
Please ascertain that your dataset fully works with the Web Application prior to opening an issue.


## 1.2. Contribute to the Software

If you wish to contribute to the Metrics As Scores application, you are welcome to [open a new issue](./issues) for

- Suggesting new features
- Reporting errors
- Discuss currently existing features
- Any other questions related to the application

There is no issue template. However, when reporting bugs, make sure to disclose the exact version used, both of Metrics As Scores, your platform, the dataset, etc., and to provide, if possible, a minimum reproducible example.

The most common way to contribute new features, bug fixes, and maintenance patches is to fork the repository and [open a pull request](./pulls).

