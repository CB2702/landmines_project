from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, log_softmax
from keras.optimizers import Adam, SGD
from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

def build_binary_classifier_model():
    model = Sequential([
    Dense(64, activation='relu', name = 'input', input_dim = 7),
    Dense(32, activation= 'relu', name = 'hidden1'),
    Dense(1, activation = 'sigmoid', name = 'output')])

    model.compile(optimizer='adam', metrics=['accuracy'], loss = BinaryCrossentropy(from_logits=False))
    
    return model

def build_multiclass_classifier_model():
    model = Sequential([
    Dense(512, activation='relu', input_dim = 7,  name = 'input'),
    Dense(256, activation= 'relu', name = 'hidden1'),
    Dense(128, activation= 'relu', name = 'hidden2'),
    Dense(4, activation = 'softmax', name = 'output')])

    model.compile(optimizer='adam', metrics=['accuracy'], loss = SparseCategoricalCrossentropy(from_logits=False))
    
    return model


def prepare_dataset(df: pd.DataFrame, drop_cols: list, target_var: str, val_test_size: float, test_size: float, scale = False, encode = False, seed = 69):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    if target_var == 'M':
        df = df.loc[df['is_M'] == 1]

    # Prepare X
    X = df.drop(columns = drop_cols)
    if scale:
        X = scaler.fit_transform(X)

    # Prepare y
    y = df[target_var]
    if encode:
        y = encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size, random_state = seed)
    X_train = tf.convert_to_tensor(X_train)
    X_val = tf.convert_to_tensor(X_val)
    y_train = np.array(y_train)
    y_test = np.array(y_val)

    return df, X_train, X_val, y_train, y_val

def train_model(model, X_train, y_train, epochs = 100):
    # Change this at will!
    model.fit(X_train, y_train, epochs = epochs)

    return model

def evaluate_model(model, X_val, y_val):
    results = model.evaluate(X_val, y_val)
    return results

def build_conf_matrix(model, X_val, y_val, fig_name):
    y_pred_prob = model.predict(X_val)
    y_pred_class = np.argmax(y_pred_prob, axis = 1)
    conf_matrix = confusion_matrix(y_true=y_val, y_pred=y_pred_class)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(f'results/{fig_name}')

    return None

