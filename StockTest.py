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

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

DJI = pd.read_csv("./data/_DJI.csv")
CPIAUCSL = pd.read_csv("./data/CPIAUCSL.csv")
DGS10 = pd.read_csv("./data/DGS10.csv")
fed = pd.read_csv("./data/fed-funds-rate-historical-chart.csv")
LNS = pd.read_csv("./data/LNS14000024.csv")

merged_data=pd.merge(DJI,CPIAUCSL, on="DATE", how="left")
merged_data=pd.merge(merged_data,DGS10, on="DATE", how="left")
merged_data=pd.merge(merged_data,fed, on="DATE", how="left")
merged_data=pd.merge(merged_data,LNS, on="DATE", how="left")
merged_data=merged_data.replace(np.nan,0)
merged_data=merged_data.replace(".",0)
merged_data[["value"]]=merged_data[["value"]].astype(float)
merged_data[["AdjClose"]]=merged_data[["AdjClose"]].astype(float)
#merged_data.to_csv("./merged_data.csv")
#merged_data_tensor=tf.convert_to_tensor(merged_data)

target_str = "AdjClose"
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#merged_data_tensor=sess.run(merged_data_tensor.replace(np.nan,0))

#DATE    Open    High    Low    Close    Adj Close    Volume    CPIAUCSL    DGS10     value    LNS14000024

print(merged_data)
def preprocess_features(stock):
  selected_features = stock[
    [
     "value",
#     "DGS10"
     "AdjClose"     
     ]]
  processed_features = selected_features.copy()
  return processed_features

def preprocess_targets(stock):
  output_targets = pd.DataFrame()
  output_targets[target_str] = (
    stock[target_str]/1000)
  return output_targets



def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
        
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
     # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(1000)
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
  
 
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
          training_targets[target_str], 
          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
          training_examples, 
          training_targets[target_str], 
          num_epochs=1, 
          shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
          validation_examples, validation_targets[target_str], 
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
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        print("Model training finished.")
    
    
      # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()
    
    return linear_regressor



training_examples = preprocess_features(merged_data.head(5000))
#training_examples.describe()
print(training_examples)
training_targets = preprocess_targets(merged_data.head(5000))
#training_targets.describe()

#validation_examples = preprocess_features(grade.tail(35))
#validation_examples.describe()
validation_examples = preprocess_features(merged_data[5000:7000])
print(validation_examples)
#validation_targets = preprocess_targets(grade.tail(35))
#validation_targets.describe()
validation_targets = preprocess_targets(merged_data[5000:7000])

linear_regressor = train_model(
    learning_rate=0.00001,
    steps=5000,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
   
#   
test_examples = preprocess_features(merged_data[7000:])
test_targets = preprocess_targets(merged_data[7000:])
 
predict_test_input_fn = lambda: my_input_fn(
       test_examples, 
       test_targets[target_str], 
       num_epochs=1, 
       shuffle=False)
 
test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])
 
root_mean_squared_error = math.sqrt(
     metrics.mean_squared_error(test_predictions, test_targets))
 
print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

