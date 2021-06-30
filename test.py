from skalpp import (
    ScalerOHEncoderWrapper
)

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

def test_1():
    """
    test the scaler/encoder with some tiny toy datasets
    """

    tiny_data = [
        { 'name': 'Bizcuit', 'age': 29, 'city': 'Manchester', 
            'gender': 'M', 'start_date': '2019-05-10' },
        { 'name': 'Eczmax', 'age': 23, 'city': 'Liverpool', 
            'gender': 'F', 'start_date': '2019-07-08' },
        { 'name': 'Boeing', 'age': 33, 'city': 'Manchester', 
            'gender': 'M', 'start_date': '2019-08-11' }
    ]
    
    tiny_df = pd.DataFrame(tiny_data, 
        columns=[ 'name', 'age', 'city', 'gender', 'start_date' ])
    
    """
    >>> tiny_df
          name  age        city gender  start_date
    0  Bizcuit   29  Manchester      M  2019-05-10
    1   Eczmax   23   Liverpool      F  2019-07-08
    2   Boeing   33  Manchester      M  2019-08-11
    """
    
    tiny_df['age'] = tiny_df['age'].astype(pd.Int8Dtype())
    tiny_df[['city','gender']] = \
        tiny_df[['city','gender']].astype('category')
    tiny_df['start_date'] = pd.to_datetime(tiny_df['start_date'])
    
    """
    >>> tiny_df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   name        3 non-null      object
     1   age         3 non-null      Int8
     2   city        3 non-null      category
     3   gender      3 non-null      category
     4   start_date  3 non-null      datetime64[ns]
    dtypes: Int8(1), category(2), datetime64[ns](1), object(1)
    memory usage: 380.0+ bytes

    >>> tiny_df
          name  age        city gender start_date
    0  Bizcuit   29  Manchester      M 2019-05-10
    1   Eczmax   23   Liverpool      F 2019-07-08
    2   Boeing   33  Manchester      M 2019-08-11
    """

    scaler_encoder = ScalerOHEncoderWrapper()
    scaler_encoder.fit(tiny_df)
    scaler_encoder.transform(tiny_df)
    """
            age city_Liverpool city_Manchester gender_F gender_M     name start_date
    0  0.162221              0               1        0        1  Bizcuit 2019-05-10
    1  -1.29777              1               0        1        0   Eczmax 2019-07-08
    2   1.13555              0               1        0        1   Boeing 2019-08-11    """

    scaler_encoder._col_trans.named_transformers_['categorical'].named_steps['onehot'].inverse_transform([[ 0, 1, 0, 1]])
    """
    array([['Manchester', 'M']], dtype=object)
    """



def test_2():
    """
    test the scaler/encoder with some tiny toy datasets
    """

    # from from https://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html#sphx-glr-auto-examples-linear-model-plot-huber-vs-ridge-py
    rng = np.random.RandomState(0)
    X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0)
    tiny_data = X
    tiny_df = pd.DataFrame(tiny_data, columns=[ 'x' ])
    """
    >>> tiny_df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20 entries, 0 to 19
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   x       20 non-null     float64
    dtypes: float64(1)
    memory usage: 288.0 bytes
    """

    scaler_encoder = ScalerOHEncoderWrapper()
    scaler_encoder.fit(tiny_df)



def main():
    test_1()
    test_2()


if __name__ == '__main__':
    main()


