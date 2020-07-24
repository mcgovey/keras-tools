import importlib.util
spec = importlib.util.spec_from_file_location("KerasTools", "KerasTools/__init__.py")
KerasTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KerasTools)

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv').iloc[:100,:]
helper = KerasTools.keras_tools(data = df, ts_n_y_vals = 5, debug=True)

helper.train_test_split(split_type='sample')

print(f"test_df: {helper.test_df.shape}")
print(f"train_df: {helper.train_df.shape}")
print(f"valid_df: {helper.valid_df.shape}")
print(f"test_df: {helper.test_df}")
print(f"train_df: {helper.train_df}")

# print(f"X_train: {helper.X_train.shape}")

# helper.rnn_transform(split_pct = 0.3,
# 						val_split_pct = 0.1,
# 						step = 1,
# 						sample_size = 1)
# print(f"X_train: {helper.X_train.shape}")
# print(f"y_train: {helper.y_train.shape}")
# print(f"X_test: {helper.X_test.shape}")
# print(f"y_test: {helper.y_test.shape}")
# print(f"X_valid: {helper.X_valid.shape}")
# print(f"y_valid: {helper.X_valid.shape}")
# helper.scale(scaler = "minmax")

# print(f"y_train: {helper.y_train}")

# print(df.shape)
# print(type(helper.X_train))