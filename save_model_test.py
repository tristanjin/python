
'''
Created on 2018. 11. 6.

@author: tristanjin

'''

from __future__ import print_function
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.framework.dtypes import string

tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None

DJI = pd.read_csv("./data/_DJI.csv")
CPIAUCSL = pd.read_csv("./data/CPIAUCSL.csv")
DGS10 = pd.read_csv("./data/DGS10.csv")
fed = pd.read_csv("./data/fed-funds-rate-historical-chart.csv")
LNS = pd.read_csv("./data/LNS14000024.csv")
#DJI["DATE"]=pd.datetime(DJI["DATE"])
DJI["DATE"]=pd.to_datetime(DJI['DATE'])

CPIAUCSL["DATE"]=pd.to_datetime(CPIAUCSL['DATE'])
DGS10["DATE"]=pd.to_datetime(DGS10['DATE'])
fed["DATE"]=pd.to_datetime(fed['DATE'])
LNS["DATE"]=pd.to_datetime(LNS['DATE'])

merged_data=pd.merge(DJI,CPIAUCSL, on="DATE", how="left")
merged_data=pd.merge(merged_data,DGS10, on="DATE", how="left")
merged_data=pd.merge(merged_data,fed, on="DATE", how="left")
merged_data=pd.merge(merged_data,LNS, on="DATE", how="left")
#merged_data=merged_data.replace(np.nan,0)

merged_data=merged_data.replace(".",np.nan)
merged_data[['Close']] = merged_data[['Close']].astype(float)
merged_data[['DGS10']] = merged_data[['DGS10']].astype(float)
merged_data['Close']=merged_data['Close']/1000

#merged_data.to_csv("merged.csv",mode="a",header=True)
#DATE Open High Low Close Adj Close Volume CPIAUCSL DGS10 value LNS14000024
#merged_data_tensor=tf.convert_to_tensor(merged_data)
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#merged_data_tensor=sess.run(merged_data_tensor.replace(np.nan,0))
#merged_data.to_csv("merged.csv",mode="a",header=True)
#for x in range(0, merged_data.shape[0]):
 #if x%100 == 0 :
# print(merged_data.iloc[x]["LNS14000024"])
#if merged_data.loc[x]["LNS14000024"] == 0.0 and x != 0 :
#print(merged_data.loc[x-1]["LNS14000024"])
# temp = merged_data.loc[x-1]["LNS14000024"] 
# print(temp)
# merged_data.loc[x]["LNS14000024"] = temp
#print(merged_data.loc[x]["LNS14000024"])
merged_data=merged_data.fillna(method='ffill') 
merged_data=merged_data.fillna(method='bfill')
#for x in range(0, merged_data.shape[0]): 
# print(merged_data.iloc[x]["LNS14000024"])
print(merged_data.tail(100))
def preprocess_features(df):
    selected_features = df[[
    # "DATE",
    # "Open",
    # "High",
    # "Low",
    # "Adj",
     "Close",
    # "Volume",
    # "CPIAUCSL",
    # "DGS10",
     "value"
    # "LNS14000024" 
    ]]

    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(df):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.

    output_targets["Close"] = (
    df["Close"])

    return output_targets

train_example = preprocess_features(merged_data.head(5000))
train_target = preprocess_targets(merged_data.head(5000))
validation_example = preprocess_features(merged_data[5001:6500])
validation_target = preprocess_targets(merged_data[5001:6500])
test_example = preprocess_features(merged_data.tail(2000))
test_target = preprocess_targets(merged_data.tail(2000))
correlation_dataframe = train_example.copy()
correlation_dataframe["target"] = train_target["Close"]

print(correlation_dataframe.corr())
merged_data=merged_data.reindex()

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
# Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()} 
     # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
     # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


feature_spec = {'Close': tf.FixedLenFeature([1], dtype=tf.float32),
                 'value': tf.FixedLenFeature([1], dtype=tf.float32)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

 # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=construct_feature_columns(training_examples),
    optimizer=my_optimizer
    )

 # Create input functions.
    training_input_fn = lambda: my_input_fn(
                training_examples, 
                training_targets["Close"], 
                batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
                training_examples, 
                training_targets["Close"], 
                num_epochs=1, 
                shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(
            validation_examples, validation_targets["Close"], 
            num_epochs=1, 
            shuffle=False)
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
        )

        #linear_regressor.train.export_savedmodel("./models", serving_input_receiver_fn, assets_extra, as_text, checkpoint_path)
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
       
        # Compute training and validation loss.

        training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # Occasionally print the current loss.
        print(" period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    
    
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    export_dir_base="C:/workspace/test1/multiLinear/saved_model"
    linear_regressor.export_savedmodel(export_dir_base, serving_input_receiver_fn,strip_default_attrs=True)
#     plt.ylabel("RMSE")
#     plt.xlabel("Periods")
#     plt.title("Root Mean Squared Error vs. Periods")
#     plt.tight_layout()
#     plt.plot(training_rmse, label="training")
#     plt.plot(validation_rmse, label="validation")
#     plt.legend()
#     plt.show()
    return linear_regressor

linear_regressor = train_model(
    learning_rate=0.0001,
    steps=1000,
    batch_size=1,
    #feature_columns=construct_feature_columns(train_example),
    training_examples=train_example,
    training_targets=train_target,
    validation_examples=validation_example,
    validation_targets=validation_target)

predict_test_input_fn = lambda: my_input_fn(
    test_example, 
    test_target["Close"], 
    num_epochs=1, 
    shuffle=False)

# test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
# test_predictions = np.array([item['predictions'][0] for item in test_predictions])
# root_mean_squared_error = math.sqrt(
#     metrics.mean_squared_error(test_predictions, test_target))
# print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

 
