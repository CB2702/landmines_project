import pandas as pd
import numpy as np
from packages.landmines.ml_ops.data_import.data_import import run_import_data
from packages.landmines.ml_ops.data_processing.data_processing import run_data_processing
from packages.landmines.ml_source.training.train import prepare_dataset, train_model, perform_inference, build_binary_classifier_model, build_multiclass_classifier_model
import plotly_express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

### TITLE SECTION ###
st.title("Landmine Detection Project")
st.subheader("Using TensorFlow Neural Networks to Detect and Assort Landmines")
### PAGE BODY CONTAINER ###

st.header("Data Import and Exploration")
if "raw_data" not in st.session_state:
    path = 'data\Mine_Dataset.xls'
    raw_data = run_import_data(path = path, sheet_name = 'Normalized_Data', name = 'landmines')
    st.session_state['raw_data'] = raw_data
else:
    raw_data = st.session_state['raw_data']

st.subheader("Imported Data")
selection_list = list(raw_data.columns)
info_dict = {'V': 'Output voltage value of FLC sensor due to magnetic distortion',
             'H': 'Height of the sensor from the ground',
             'S': 'Soil type [0: dry and sandy, 0.2: dry and humus, 0.4: dry and limy, 0.6: humid and sandy, 0.8: humid and humus, 1: humid and limy]',
             'M': 'Mine type [1: no mine, 2: anti tank, 3: anti personnel, 4: booby trapped anti personnel, 5: m14 anti personnel]'
             }

selectbox_info = st.selectbox('Select a column for more info', selection_list)

st.write(f"Column Description - {selectbox_info}: {info_dict[selectbox_info]}")

st.dataframe(raw_data)

st.subheader("Exploration")
st.write("Use the selection box below to explore the distribution of the dataframe columns.")
selectbox_col = st.selectbox("Select a column", selection_list)
fig = px.histogram(
    raw_data, 
    x=selectbox_col, 
    nbins=30,
    title=f"Histogram of column {selectbox_col}"
)

# 3. Display in Streamlit

st.plotly_chart(fig, use_container_width=True)

st.write("Please select a mine class using the selectbox below to view the datapoint distribution of a single point")
mine_selection_list = list(raw_data['M'].unique())
selectbox_mine = st.selectbox("Please select a column", mine_selection_list)

df_mine = raw_data.loc[raw_data['M'] == selectbox_mine]

fig = px.scatter(
    df_mine, 
    x='V',
    y = 'H',
    color='S',
    title=f"Scatter of mine {selectbox_mine} hued by soil type"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.scatter_3d(
    raw_data,
    x = 'V',
    y = 'H',
    z = 'S',
    color='M',
    title=f"3D Scatter of mine {selectbox_mine}"
)

st.plotly_chart(fig)

st.header("Modelling Process")
st.subheader("Data Processing")

st.write("""
        The following data processing steps have been performed:
         1) Enforcing categorical variable types for 'S' and 'M'
         2) One-hot encoding 'S' as a feature, removing S = 0 to reduce degrees of freedom
         3) Defining a binary categorical variable (M == 1) for definition of 'is_mine'
         4) Standard scale data (performed during training process)
        """)

if 'processed_data' not in st.session_state:
    processed_data = run_data_processing(raw_data)
    st.session_state['processed_data'] = processed_data
else:
    processed_data = st.session_state['processed_data']

st.write("See below table for processed data.")
st.dataframe(processed_data)

st.subheader("Modelling Phase 1: Binary Classifier")
st.write("We will now use a binary classifier to assert whether a given datapoint is in fact a mine or not.")
st.write("We have built a binary classifier neural network using TensorFlow for this task.")

if 'binary_classifier' not in st.session_state:
    binary_classifier = build_binary_classifier_model()
    st.session_state['binary_classifier'] = binary_classifier
else:
    binary_classifier = st.session_state['binary_classifier']
seed = np.random.random_integers(0, 100)
df_bc, X_train_bc, X_val_bc, y_train_bc, y_val_bc, X_cols = prepare_dataset(processed_data, ['is_M', 'M'], 'is_M', 0.3, 0.5, scale = True, encode=False, seed = seed)
df_mc, X_train_mc, X_val_mc, y_train_mc, y_val_mc, X_cols = prepare_dataset(processed_data, ['is_M', 'M'], 'M', 0.3, 0.5, scale = True, encode=True, seed=seed)
if 'trained_binary_classifier' not in st.session_state:
    binary_classifier_trained = train_model(binary_classifier, X_train_bc, y_train_bc)
    st.session_state['trained_binary_classifier'] = binary_classifier_trained
else:
    binary_classifier_trained = st.session_state['trained_binary_classifier']

st.write("See below for binary classifier results")

results = perform_inference(binary_classifier_trained, X_val_bc, y_val_bc, X_cols)
y_pred_class = np.round(results['y_hat'])
conf_matrix = confusion_matrix(y_true=y_val_bc, y_pred=y_pred_class)
labels = [0, 1]

st.dataframe(results)

fig = px.imshow(
    conf_matrix, 
    x=labels, 
    y=labels, 
    text_auto=True,          # Automatically adds counts to cells
    aspect="auto", 
    color_continuous_scale='Blues',
    labels=dict(x="Predicted", y="Actual", color="Count")
)

fig.update_layout(title='Confusion Matrix of Mine vs. Not Mine')

st.plotly_chart(fig)

st.subheader("Modelling Phase 2: Multiclass Classifier")
st.write("We will now use a binary classifier to classify which mine type a mine is (not including non-mine datapoints)")
st.write("We have built a multiclass classifier neural network with sparse categorical entropy using TensorFlow for this task.")

if 'multiclass_classifier' not in st.session_state:
    multiclass_classifier = build_multiclass_classifier_model()
    st.session_state['multiclass_classifier'] = multiclass_classifier
else:
    multiclass_classifier = st.session_state['multiclass_classifier']

if 'trained_multiclass_classifier' not in st.session_state:
    multiclass_classifier_trained = train_model(multiclass_classifier, X_train_mc, y_train_mc)
    st.session_state['trained_multiclass_classifier'] = multiclass_classifier_trained
else:
    multiclass_classifier_trained = st.session_state['trained_multiclass_classifier']

results_mc = multiclass_classifier_trained.predict(X_val_mc)
y_pred_class_mc = np.argmax(results_mc, axis = 1)
conf_matrix = confusion_matrix(y_true=y_val_mc, y_pred=y_pred_class_mc)
labels = [0,1,2,3]


fig = px.imshow(
    conf_matrix, 
    x=labels, 
    y=labels, 
    text_auto=True,          # Automatically adds counts to cells
    aspect="auto", 
    color_continuous_scale='Blues',
    labels=dict(x="Predicted", y="Actual", color="Count")
)


st.plotly_chart(fig)






