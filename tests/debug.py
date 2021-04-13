import importlib.util
spec = importlib.util.spec_from_file_location("KerasTools", "KerasTools/__init__.py")
KerasTools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KerasTools)

import pandas as pd

from tensorflow.keras import models, layers, callbacks, Input

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv').iloc[:100,:]

n_y_vals = 5
feature_list = [1,2]

helper = KerasTools.keras_tools(data = df, 
                                    features = feature_list, 
                                    index = 0, 
                                    ts_n_y_vals = n_y_vals, 
                                    debug=True)

helper.train_test_split(split_type='overlap')

print(f"test_df: {helper.test_df.shape}")
print(f"train_df: {helper.train_df.shape}")
print(f"valid_df: {helper.valid_df.shape}")
print(f"test_df: {helper.test_df}")
print(f"train_df: {helper.train_df}")


step = 3
sample_size = 1


helper.reshape_ts(step = step,
					sample_size = sample_size,
					scaler = "minmax")

print(f"X_train: {helper.X_train.shape}")
print(f"y_train: {helper.y_train.shape}")
print(f"X_test: {helper.X_test.shape}")
print(f"y_test: {helper.y_test.shape}")
print(f"X_valid: {helper.X_valid.shape}")
print(f"y_valid: {helper.X_valid.shape}")


input_shape = helper.get_input_shape()

timeseries_input = Input(shape=input_shape, dtype='float32', name='timeseries')

ts_layer = layers.LSTM(units=16, 
                       activation='relu')(timeseries_input)
                       
output = layers.Dense(len(feature_list) * n_y_vals, activation=None)(ts_layer)
output = layers.Reshape((len(feature_list), n_y_vals))(output)

model = models.Model(timeseries_input, output)
model.compile(optimizer='adam',
             loss='mse',
             metrics=['mse', 'mae', 'mape'])
             
history = model.fit(helper.X_train, 
                        helper.y_train,
                        validation_data = (helper.X_test, helper.y_test),
                        steps_per_epoch=5,
                        validation_steps = 5,
                        epochs=10,
                        verbose=1)
                        
# helper.model_summary(model, history, show_charts=True)
# print(f"X_train: {helper.X_train}")
# print(f"y_train: {helper.y_train}")
print(helper.X_valid)
preds = helper.predict_ts(helper.X_valid, model = model)
print(preds)