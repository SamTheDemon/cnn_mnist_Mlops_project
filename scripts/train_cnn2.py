# scripts/train_cnn.py
import sys
import os
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from load_data import load_mnist

def create_cnn_model_v1(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2, 2), strides=1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=1))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2, 2), strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_cnn_model_v2(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def main(model_version):
    mlflow.start_run()
    
    # Load Data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Preprocess Data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Create Model
    input_shape = (28, 28, 1)
    if model_version == 'v1':
        model = create_cnn_model_v1(input_shape, num_classes)
    else:
        model = create_cnn_model_v2(input_shape, num_classes)
    
    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Log parameters
    mlflow.log_param('model_version', model_version)
    mlflow.log_param('optimizer', 'adam')
    mlflow.log_param('loss_function', 'categorical_crossentropy')
    
    # Train Model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    
    # Evaluate Model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    mlflow.log_metric('test_loss', test_loss)
    mlflow.log_metric('test_accuracy', test_acc)
    
    # Log the model
    mlflow.keras.log_model(model, "model")
    
    mlflow.end_run()

if __name__ == "__main__":
    model_version = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    main(model_version)
