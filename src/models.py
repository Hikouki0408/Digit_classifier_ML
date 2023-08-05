import keras_tuner as kt
import numpy as np
import os
import shutil

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv1D, Reshape, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def test_specific_hp_model(model_builder, train_set, validation_set, test_set):
    # if os.path.exists("../logs") and os.path.isdir("../logs"):
    #     shutil.rmtree("../logs")
    
    if not os.path.exists("../figures"):
        os.mkdir("../figures")
    
    save_path = "../models/tuned_model.h5"
    figure_path = "../figures/tuned_model.png"
    run_name = "khyperband_modeltest"
    
    (x_train, y_train) = train_set
    (x_val, y_val) = validation_set
    (x_test, y_test) = test_set
    
    tuner = kt.Hyperband(model_builder, objective="accuracy", max_epochs=10, factor=3, directory="../logs", project_name=run_name)
    tuner.search(x_val, y_val, epochs=10, callbacks=[EarlyStopping(monitor="loss", patience=5)])
    
    best_hps = tuner.get_best_hyperparameters()[0]
    
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=10, validation_data=validation_set, batch_size=200, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    
    print("Model Summary: ")
    print("Best hyperparameters: " + str(best_hps.values))
    print("Loss: " + str(scores[0]))
    print("Accuracy: " + str(scores[1]))
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    fig = plt.figure()
    plt.suptitle("Tuned hyperparameters, bias removed.")
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history.history['loss'])), np.arange(1, len(history.history['loss'])+1))
    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history.history['loss'])), np.arange(1, len(history.history['loss'])+1))
    plt.legend(['Training', 'Validation'], loc='upper right')
    
    fig.tight_layout()
    
    fig.savefig(figure_path)
    print("Saved accuracy and loss plots at %s " % figure_path)
    
    model.save(save_path)
    print("Saved trained tuned model at %s " % save_path)

def test_specific_normal_model(modelFn, train_set, validation_set, test_set):
    save_path = "../models/normal_model.h5"
    figure_path = "../figures/normal_model.png"
    
    if not os.path.exists("../figures"):
        os.mkdir("../figures")
    
    (x_train, y_train) = train_set
    (x_test, y_test) = test_set
    
    model = modelFn()
    
    history = model.fit(x_train, y_train, validation_data=validation_set, epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    
    print("Model Summary: ")
    print("Loss: " + str(scores[0]))
    print("Accuracy: " + str(scores[1]))
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    fig = plt.figure()
    fig.suptitle("Default hyperparameters, bias removed.")
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history.history['loss'])), np.arange(1, len(history.history['loss'])+1))
    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history.history['loss'])), np.arange(1, len(history.history['loss'])+1))
    plt.legend(['Training', 'Validation'], loc='upper right')
    
    fig.tight_layout()
    
    fig.savefig(figure_path)
    print("Saved accuracy and loss plots at %s " % figure_path)
    
    model.save(save_path)
    print("Saved trained normal model at %s " % save_path)

# Select the best model overall
def select_best_model(train_set, validation_set, test_set):
    (x_test, y_test) = test_set
    
    best_model = None
    
    best_hp_model = select_best_hp_model(train_set, validation_set, test_set)
    best_normal_model = select_best_normal_model(train_set, validation_set, test_set)
    
    scores_hp_model = best_hp_model.evaluate(x_test, y_test, verbose=0)
    scores_normal_model = best_normal_model.evaluate(x_test, y_test, verbose=0)
    
    error_hp_model = 100 - scores_hp_model[1] * 100
    error_normal_model = 100 - scores_normal_model[1] * 100
        
    print("Normal model acc: %f" % error_normal_model)
    print("Tuned model acc: %f" % error_hp_model)
    if error_hp_model > error_normal_model:
        print("normal model is more accurate")
    else:
        print("tuned version is more accurate")
    return best_normal_model if error_hp_model > error_normal_model else best_hp_model

# Select the best model out of models with tuned hyperparams
def select_best_hp_model(train_set, validation_set, test_set):
    models = get_hp_models()
    
    (x_train, y_train) = train_set
    (x_val, y_val) = validation_set
    (x_test, y_test) = test_set
    
    best_model = None
    best_error = 100.0
    
    for count, model_builder in enumerate(models):
        run_name = "khyperband_model%d" % count
        
        tuner = kt.Hyperband(model_builder, objective="accuracy", max_epochs=10, factor=3, directory="../logs", project_name=run_name)
        tuner.search(x_val, y_val, epochs=10, callbacks=[EarlyStopping(monitor="loss", patience=5)])
        
        best_hps = tuner.get_best_hyperparameters()[0]
        
        model = tuner.hypermodel.build(best_hps)
        model.fit(x_train, y_train, epochs=10, validation_data=validation_set, batch_size=200, verbose=2)
        scores = model.evaluate(x_test, y_test, verbose=0)
        
        error = 100 - scores[1] * 100
        
        print("Model Summary: ")
        print("Best hyperparameters: " + str(best_hps.values))
        print("Loss: " + str(scores[0]))
        print("Accuracy: " + str(scores[1]))
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        
        if error < best_error:
            print("Found better tuned model: %s" % model.modelname)
            best_error = error
            best_model = model
    
    print("Found the best tuned model: %s." % best_model.modelname)
    return best_model

# Select the best model out of models with default hyperparams
def select_best_normal_model(train_set, validation_set, test_set):
    models = get_normal_models()
    
    (x_train, y_train) = train_set
    (x_test, y_test) = test_set
    
    best_model = None
    best_error = 100.0
    
    for model in models:
        model.fit(x_train, y_train, validation_data=validation_set, epochs=10, batch_size=200, verbose=2)
        scores = model.evaluate(x_test, y_test, verbose=0)
        
        error = 100 - scores[1] * 100
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        if error < best_error:
            print("Found better normal model: %s" % model.modelname)
            best_error = error
            best_model = model

    print("Found the best normal model: %s." % best_model.modelname)
    return best_model

# Get all models with a set of hyperparams
def get_hp_models():
    models = []

    # Baseline models
    models.append(create_model1_hp)
    models.append(create_model2_hp)
    
    # Convolutional models
    # models.append(create_model_conv1d(num_pixels, num_classes))
    # models.append(create_model_conv2d(28, num_classes))

    return models

# Get all normal models that do not require to be tuned
def get_normal_models():
    models = []

    # Baseline models
    models.append(create_model1())
    models.append(create_model2())

    return models

def create_model1():
    # Create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer="normal", activation="relu"))
    model.add(Dense(10, kernel_initializer="normal", activation="softmax"))

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.modelname = "Model1"
    return model

def create_model1_hp(hp):
    # Hyperparams setting
    hp_activation = hp.Choice("activation", values=["relu", "sigmoid", "tanh", "selu"])
    hp_learning_rate = hp.Choice("learning_rate", values=[0.1, 0.01, 0.001, 0.0001])
    
    # Create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer="normal", activation=hp_activation))
    model.add(Dense(10, kernel_initializer="normal", activation="softmax"))

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=hp_learning_rate), metrics=["accuracy"])
    
    model.modelname = "Model1_hp"
    return model

def create_model2():
    # Create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer="normal", activation="relu"))
    model.add(Dense(784, kernel_initializer="normal", activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer="normal", activation="softmax"))

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.modelname = "Model2"
    return model

def create_model2_hp(hp):
    # Hyperparams setting
    hp_units = hp.Int("units", min_value=16, max_value=784, step=16)
    hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    hp_activation = hp.Choice("activation", values=["relu", "sigmoid", "tanh", "selu"])
    hp_learning_rate = hp.Choice("learning_rate", values=[0.1, 0.01, 0.001, 0.0001])
    
    # Create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer="normal", activation=hp_activation))
    model.add(Dense(hp_units, kernel_initializer="normal", activation=hp_activation))
    model.add(Dropout(hp_dropout))
    model.add(Dense(10, kernel_initializer="normal", activation="softmax"))

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=hp_learning_rate), metrics=["accuracy"])
    # model.compile(loss="categorical_crossentropy", optimizer=hp_optimizer, metrics=["accuracy"])
    
    model.modelname = "Model2_hp"
    return model

# def create_model_conv1d(num_pixels, num_classes):
#     # Create model
#     model = Sequential()
#     model.add(Reshape(target_shape=(1, 784), input_shape=(784,)))
#     model.add(Conv1D(1, 3, input_dim=num_pixels, activation="relu", kernel_initializer="he_uniform"))
#     model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
#     model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))

#     # Compile model
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
#     model.modelname = "Convolutional 1D Model"
#     return model

# def create_model_conv2d(num_pixels, num_classes):
#     # Create model
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(num_pixels, num_pixels, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
#     model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))

#     # Compile model
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
#     model.modelname = "Convolutional 2D Model"
#     return model