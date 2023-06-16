In this repostory, we will train models that leverage validation data generated for the OPERA DSWx-HLS product and produce open surface water / not water inferences on the entire Planetscope image from which the validation data was generated. 

We then follow a previously established workflow ([see here](https://github.com/OPERA-Cal-Val/DSWx-HLS-Requirement-Verification/blob/05ac35701d506ce6d1a1e886fc6c1198003e7eff/0-Verify_Requirements.ipynb)) to compare the DSWx product to the generated classification, and assign a pass/fail score based on the metrics required by requirements

# Prerequisites
The repository provides an environment file which will create an Anaconda environment and install the requisite python packages. This environment is created and activated by the following commands:

```
conda env create -f environment.yml
conda activate expand-validation
```
We recommend using [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) to speed up the process of creating the python environment

# Repository layout
    .
    ├── data
    │   └─ validation_table.csv   # File describing Planet, HLS, and DSWx image IDs, along with S3 bucket links to the DSWx and validation data
    ├── notebooks
    │   ├─ .env         # Location for Planet API key
    │   ├─ 1-Download-datasets.ipynb
    │   ├─ 2-Train-and-infer-all-chips.ipynb
    │   ├─ 3a-Recalculate-validation-metrics.ipynb
    │   └─ 3b-Recalculate-validation-metrics-all-images.ipynb
    │
    ├── environment.yml # YAML file describing environment dependencies
    └── README.md       # This file

# Downloading data

This repository relies on the access of data from AWS S3 buckets and through the Planet API. The DSWx tiles and validation data are hosted on publicly accessible S3 buckets, and do not require any access credentials. 
Access to Planet data requires an API key, which is specified in a `.env` file in the `notebooks` folder. The content of the file is a single line specifying the API key:
```
PLANET_API_KEY='YOUR KEY HERE'
```

Planet data is made available through the NASA Commercial Smallsat Data Acquisition (CSDA) program. The access request form is available through the [CSDA website](https://www.earthdata.nasa.gov/esds/csda).