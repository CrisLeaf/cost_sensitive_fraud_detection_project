# Fraud Detection Project (with Imbalanced Data)

### Installation

This project requires the following libraries:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Seaborn](http://seaborn.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/install.html)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [CatBoost](https://catboost.ai/)

You will also need to have a software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

To install the packages, on a Python 3.9.4 virtual environment run :

```
pip install -r requirements.txt
```


### Dataset

The data was downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It is the result of applying a PCA (dimensionality reduction) algorithm into a full dataset of transactions, to protect the customers privacy.

Feature Columns:
- `V1-V28`: numeric masked columns.
- `Time`: time in seconds after the first transaction.
- `Amount`: the transaction money.

Target Column:
- `Class`: label with 0 for real and 1 for fraud transactions.


### Code

The code provided by the `fraud_detection.ipynb` notebook file, describes an Amount sensitive attempt to build a model for fraud detection. After a short Statistical Analysis a construction of the most used tree-based classification algorithms and a performance evaluation is presented.


### Run


In a terminal window, run one of the followings:

```
ipython notebook fraud_detection.ipynb
```  
or
```
jupyter notebook fraud_detection.ipynb
```
or open with Jupyter Lab
```
jupyter lab
```


### Results


A model was found that allows saving up to 76% of money involved in fraudulent transactions. It consists of using the LightGBM Classifier trained with augmented minor class data.