"""
30-Jun-21 9:35am

This is the main file utility functions and classes for sklearn data pre-processing. At the moment, it is targeted at my own projects. At this stage, it has minimal documentation and no testing or exception handling, and really should be refactored to be mode generic/parameterisable. At some point, I might address these and make it look much more like a proper package/module for being used by other people, even if just for good practice.

It depends on sklearn pipelines and column transformers - I have to check what is the mininum requirement of sklearn version.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.datasets import make_regression


class ScalerOHEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    to do:

    * deal with both `category` and `object` for categorical encoding.

    * this works well for a less flexible transformer - numeric, one-hot encoding, pass-through (in that order). I will try it with `xgboost`. For `catboost` and `lightgbm`, I would like to try their categorical feature support, so the transformer will change slightly (e.g., ordinal encoding). Some refactoring could be useful here at some point.

    * inspired from:

        - https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62

        - https://github.com/klemag/PyconUS_2019-model-interpretability-tutorial/blob/master/01-interpretability_eli5-skeleton.ipynb

    * briefly, the idea is to have a transformer that wraps around a typical ColumnTransformer for categorical (ohe) and numerical features so as to output a DataFrame with meaningful column names. Applying the ColumnTransformer leaves us with numpy arrays, which make it difficult to push forward in a data science and machine learning pipeline.

    * there a few assumptions that I will have to make here so that the code is functional right now. Ideally, it would be refactored for having customisable options for the transformers.

    * the assumptions are:
        * one StandardScaler for numeric features. The latter are recovered via `select_dtype(include='number').columns`. Probably not the most efficient way, but I am having issues using masks based on `dtypes.isin([int,float,pd.Int8Dtype]).

        * one hot-encoder for categorical features - the latter are determined by `select_dtype(include='category').columns`.

        * all other features are passed through (`remainder='passthrough').

    * the order of the features of the input DataFrame, when `transformed`, is changed. Firstly, we get the numeric features, then the categorical ones and, finally, the ones passed through. The code makes sure the alignment of feature names and columns is maintained (dangerously error-prone, though).
    """
    
    def __init__(self, num_scaler=MinMaxScaler(), remainder='passthrough'):
        """
        20-jun-20: just added num_scaler as an argument
        should the categorical and numeric transformers need to be customised,
        `cat_kws` and `num_kws` can be provided (as dictionaries, later expanded).
        """
        self.remainder = remainder
        self.num_scaler = num_scaler


    def fit(self, X, y=None):
        self._num_features = X.select_dtypes('number').columns
        # could it be done for strings as well (i.e., `object`)?
        self._cat_features = X.select_dtypes('category').columns 
        self._other_features = (X.columns
            .difference(self._cat_features)
            .difference(self._num_features)
        )

        # the pipeline for pre-process numeric features at the moment, really, just
        # a standard scalar as for categorical values, a simple imputer for dealing
        # for missing values has been included
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        #     ('scaler', MinMaxScaler())
            ('scaler', self.num_scaler)
        ])

        # my pipeline for a one-hot encoder. To highlight:
        # unlikely that I will face missing data here, but I am adding a
        # simple imputer anyway (useful for reference)
        # one-encoding has been instructed to ignore unknown categories. This is 
        # useful when a fitted transformer is being applied on validation data.
        # I do have the option of enabling `drop='first'`?!
        # will use num_kws are some point?!
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy='constant', fill_value='missing')
            ),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore', dtype='int', sparse=False)
            )
        ])
        # my pre-processor pipeline consists of a ColumnTransformer 
        # the latter decides which (sub-)pipeline to apply to each subset of
        # features
        self._col_trans = ColumnTransformer(
            transformers=[
                ('numeric', num_transformer, self._num_features),
                ('categorical', cat_transformer, self._cat_features)
            ], 
            remainder=self.remainder
        )

        self._col_trans.fit(X)

        return self



    def transform(self, X):
        """
        this is where the wrapping magic takes place. Inspired from https://github.com/klemag/PyconUS_2019-model-interpretability-tutorial/blob/master/01-interpretability_eli5-skeleton.ipynb

        """

        """
        update 12-May-20: need to allow for DataFrames not to have categorical
        data - at the moment, if this is the case, this code breaks as, among other things, `.named_steps['onehot']` would not exist.
        """

        # if there is no .categories_ feature, it probably means that
        # there are no categorical features to be encoded.
        # could have used hasattr too...
        if 'categories_' in \
            self._col_trans.named_transformers_['categorical']\
                .named_steps['onehot'].__dict__:
            ohe_categories = (self._col_trans
                .named_transformers_['categorical']
                .named_steps['onehot']
                .categories_
            )

            # update 9-Jul-20
            # xgboost doesn't seem to like features with characters such
            # as `, [ ] >` (e.g., "<=40").
            # I should replace <,>,<=,>= by '_gt_', '_lt_', '_lte_', 
            # '_gte_', respectively

            # features names prefix those given by feature values
            new_ohe_features = [
                f'{col}_{val}'
                for col, vals
                in zip(self._cat_features, ohe_categories)
                for val in [
                    c.replace('>=', '_gte_').replace('<=', '_lte_')
                     .replace('>', '_gt_').replace('<', '_lt_') 
                    for c in vals
                ]
            ]
        else:
            # well, it just seems that there are no categorical features,
            # but I will need an empty list fo represent that
            new_ohe_features = []

        # alignment of columns is performed on the assumption that the
        # column transformer outputs numeric features, followed by one-not
        # encoded categorical features and, then, the remainder passed through
        df = pd.DataFrame(
                self._col_trans.transform(X), columns=
                    self._num_features.tolist() + new_ohe_features + 
                    self._other_features.tolist()
        )

        return df


    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


