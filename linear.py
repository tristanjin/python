from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

grade = pd.read_csv("./grade.csv", sep=",")

print(grade)

def preprocess_features(grade):
  selected_features = grade[
    ["high_GPA",
     "math_SAT",
     "verb_SAT",
     "comp_GPA",
     "univ_GPA"
     ]]
  processed_features = selected_features.copy()
  return processed_features

def preprocess_targets(grade):
  
  output_targets = pd.DataFrame()
  output_targets["univ_GPA"] = (
    grade["univ_GPA"])
  return output_targets


training_examples = preprocess_features(grade.head(80))
#training_examples.describe()
print(training_examples)
training_targets = preprocess_targets(grade.head(80))
#training_targets.describe()

validation_examples = preprocess_features(grade.tail(30))
#validation_examples.describe()
print(validation_examples)
validation_targets = preprocess_targets(grade.tail(30))
#validation_targets.describe()

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
          training_targets["univ_GPA"], 
          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
          training_examples, 
          training_targets["univ_GPA"], 
          num_epochs=1, 
          shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
          validation_examples, validation_targets["univ_GPA"], 
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
    
    return linear_regressor

linear_regressor = train_model(
    learning_rate=0.03,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
  
  
  