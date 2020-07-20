import importlib.util
spec = importlib.util.spec_from_file_location("KerasTools", "KerasTools/__init__.py")
KerasTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KerasTools)

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv')
helper = KerasTools.keras_tools(data = df.iloc[:100,:], ts_n_y_vals = 5, debug=True)

helper.train_test_split(split_type='sequential')
print(f"X_train: {helper.X_train.shape}")
print(f"y_train: {helper.y_train.shape}")
print(f"X_test: {helper.X_test.shape}")
print(f"y_test: {helper.y_test.shape}")
print(f"X_valid: {helper.X_valid.shape}")
print(f"y_valid: {helper.X_valid.shape}")
# helper.scale(scaler = "minmax")

# print(type(helper.X_train))