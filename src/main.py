import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.utils import resample

import models as models

def undersample_dataset(x_set, y_set):
  min_num_samples = min([np.sum(y_set == i) for i in range(10)])

  x_undersampled = []
  y_undersampled = []
  for i in range(10):
    # Separate samples by class
    class_i = x_set[y_set == i]
    
    # Undersample majority class
    class_i_undersampled = resample(class_i, replace=False, n_samples=min_num_samples, random_state=42)
    
    # Append undersampled class to X_undersampled and y_undersampled
    x_undersampled.append(class_i_undersampled)
    y_undersampled.append(np.full(min_num_samples, i))
      
  # Combine undersampled classes into single arrays
  x_undersampled = np.vstack(x_undersampled)
  y_undersampled = np.hstack(y_undersampled)

  # Shuffle the data
  shuffle_idx = np.random.permutation(len(x_undersampled))
  x_undersampled = x_undersampled[shuffle_idx]
  y_undersampled = y_undersampled[shuffle_idx]
  
  return (x_undersampled, y_undersampled)

def main():
  # Load dataset
  print("Started. Loading MNIST dataset")
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  # Split into training and validation sets
  split_at = 0.2
  num_validation_samples = int(x_train.shape[0] * split_at)
  
  x_val = x_train[:num_validation_samples]
  y_val = y_train[:num_validation_samples]

  x_train = x_train[num_validation_samples:]
  y_train = y_train[num_validation_samples:]
  
  # Remove bias by undersampling
  (x_train, y_train) = undersample_dataset(x_train, y_train)
  (x_val, y_val) = undersample_dataset(x_val, y_val)
  
  # Flatten the 28x28 images into vectors of size 784
  num_pixels = x_train.shape[1] * x_train.shape[2]
  x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype("float32")
  x_val = x_val.reshape((x_val.shape[0], num_pixels)).astype("float32")
  x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype("float32")
  
  # Normalize the inputs from 0-255 to 0-1
  x_train = x_train / 255
  x_val = x_val / 255
  x_test = x_test / 255
  
  # One hot encode outputs
  y_train = np_utils.to_categorical(y_train)
  y_val = np_utils.to_categorical(y_val)
  y_test = np_utils.to_categorical(y_test)
  num_classes = y_test.shape[1]
  
  # To select the best version among all tuned and not tuned versions of the models use the following:
  # models.select_best_model((x_train, y_train), (x_val, y_val), (x_test, y_test))
  
  # To select the best tuned model use the following:
  # models.select_best_hp_model((x_train, y_train), (x_val, y_val), (x_test, y_test))
  
  # To select the best normal model use the following:
  # models.select_best_normal_model((x_train, y_train), (x_val, y_val), (x_test, y_test))
  
  # To test a specific model WITH hyperparameter tuning on use the following function.
  # Change the first argument to provide a model builder.
  models.test_specific_hp_model(models.create_model2_hp, (x_train, y_train), (x_val, y_val), (x_test, y_test))
  
  # To test a specific model WITHOUT hyperparameter tuning use the following function.
  # Change the first argument to provide a normal model.
  # models.test_specific_normal_model(models.create_model2, (x_train, y_train), (x_val, y_val), (x_test, y_test))
  
  print("Exiting.")

if __name__ == "__main__":
  main()
