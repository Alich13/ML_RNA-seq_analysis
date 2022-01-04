RNA-seq_analysis
==============================

Implementation of Machine learning Algorithms in order to classify tumor samples based on their transcriptomic expression profile

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Data
The data used from this projects come from a [this](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq) UCI - Machine Learning Repositry dataset with the following description reported in the website:  
  
This collection of data is part of the RNA-Seq (HiSeq) PANCAN data set, it is a random extraction of gene expressions of patients having different types of tumor: 
  
* **BRCA**: Breast Invasive Carcinoma 
* **KIRC**: Kidney Renal Clear Cell Carcinoma
* **COAD**: Colon Adenocarcinoma
* **LUAD**: Lung Adenocarcinoma
* **PRAD**: Prostate Adenocarcinoma

### Characteristics
Samples are stored rows. attributes of each sample are RNA-Seq gene expression levels measured by illumina HiSeq platform. 

**Number of samples** : 801  
**Number of attributes (features )**: 20531  
 
  
**Gene Expression Table**
|            | **gene_0** | ... | **gene_20530** |
|:----------:|:----------:|-----|----------------|
|  sample_0  |      0     | ... | 0.6            |
|     ...    |     0.6    | ... | 0              |
| sample_800 |     0.3    | ... | 0.2            |  
  
**Labels Table**  
|            | **Class** |
|:----------:|:---------:|
|  sample_0  |    PRAD   |
|     ...    |    ...    |
| sample_800 |    LUAD   |


### Source
Samuele Fiorini, University of Genoa, redistributed under [Creative Commons license](http://creativecommons.org/licenses/by/3.0/legalcode) from [here](https://www.synapse.org/#!Synapse:syn4301332)
