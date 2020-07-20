import importlib.util
spec = importlib.util.spec_from_file_location("KerasTools", "KerasTools/__init__.py")
KerasTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KerasTools)

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv')
helper = KerasTools.keras_tools(data = df, ts_n_y_vals = 5, debug=True)

returndf = helper.train_test_split(split_type='sample')
print(returndf)
# helper.scale(scaler = "minmax")

# print(type(helper.X_train))