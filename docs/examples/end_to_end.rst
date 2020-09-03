End to End Example
========================
.. code-block:: python

    import pandas as pd
    
    from tensorflow.keras import models, layers, callbacks, Input
    
    df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv').iloc[:100,:]