import pkg_resources
import subprocess
import sys


from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import pymatgen
import matminer

#Import Libraries
from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import SiteCollection
from matminer.featurizers.composition.alloy import Miedema, WenAlloys,YangSolidSolution
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.data import MixingEnthalpy, DemlData
from matminer.utils import data_files #for importing "Miedema.csv" present inside package of Matminer library
from matplotlib.ticker import MultipleLocator # for minor tick lines
import seaborn as sns

import tensorflow as tf

pd.set_option('display.max_columns', None)
import numpy as np
import os
import matplotlib.pyplot as plt

ef= ElementFraction()
stc = StrToComposition()

# Add the function.py file
from functions import *

# Set Streamlit app title
st.title('Piezoelectric Material Selection')

# Add an option to manually input a formula
next_input = st.checkbox('Add next Piezo-Material')

# Create a DataFrame to store the selected formulas
data = {'S.N': [], 'formula_pretty': []}
df_selected_formulas = pd.DataFrame(data)

# Add a dropdown to select a pre-defined formula
predefined_formulas = ['Ba0.85Ca0.15Ti0.92Zr0.07Hf0.01O3', 'Ba0.84Ca0.15Sr0.01Ti0.90Zr0.10O3', 'BaTiO3']
# selected_predefined_formula = st.selectbox('Select a pre-defined formula', predefined_formulas)
selected_predefined_formula = st.sidebar.selectbox('Select a pre-defined formula', predefined_formulas)
if selected_predefined_formula:
    df_selected_formulas = pd.concat([df_selected_formulas, pd.DataFrame({'S.N': [len(df_selected_formulas) + 1], 'formula_pretty': [selected_predefined_formula]})], ignore_index=True)

# If manual input is selected, display an input box for the custom formula
if next_input:
    custom_formula = st.text_input('Enter the custom formula')
    if custom_formula:
        df_selected_formulas = pd.concat([df_selected_formulas, pd.DataFrame({'S.N': [len(df_selected_formulas) + 1], 'formula_pretty': [custom_formula]})], ignore_index=True)

# Display the selected formulas
if not df_selected_formulas.empty:
    st.write('Selected Formulas:')
    st.dataframe(df_selected_formulas)
    
df_piezo = df_selected_formulas
# df_piezo = df_selected_formulas
# 'Piezo Materials' == 'formula_pretty'




"""

# Welcome to PiezoTensorNet!


"""
#############################################################################################################################
# Add the prediction files
#df_piezo = pd.read_csv('csv/For_Prediction.csv')
#df_piezo = df_piezo.head(50)
############################################################    Added input compositions
#df_piezo = pd.DataFrame({'formula_pretty': [selected_formula]})
df_piezo = stc.featurize_dataframe(df_piezo, 'formula_pretty',ignore_errors=True,return_errors=True)
df_piezo = ef.featurize_dataframe(df_piezo, "composition",ignore_errors=True,return_errors=True)

# In[4]:
from matminer.featurizers.composition import ElementProperty
featurizer = ElementProperty.from_preset('magpie')
df_piezo = featurizer.featurize_dataframe(df_piezo, col_id='composition')
#y = bg_data_featurized['gap expt']

print("Feature Calculation in Progress")
# In[5]:
# get_ipython().run_line_magic('run', 'functions.ipynb')
df, df_input_target = properties_calculation(df_piezo)

# In[6]:
magpie_list = ['MagpieData minimum Number',
 'MagpieData maximum Number',
 'MagpieData range Number',
 'MagpieData mean Number',
 'MagpieData avg_dev Number',
 'MagpieData mode Number',
 'MagpieData minimum MendeleevNumber',
 'MagpieData maximum MendeleevNumber',
 'MagpieData range MendeleevNumber',
 'MagpieData mean MendeleevNumber',
 'MagpieData avg_dev MendeleevNumber',
 'MagpieData mode MendeleevNumber',
 'MagpieData minimum AtomicWeight',
 'MagpieData maximum AtomicWeight',
 'MagpieData range AtomicWeight',
 'MagpieData mean AtomicWeight',
 'MagpieData avg_dev AtomicWeight',
 'MagpieData mode AtomicWeight',
 'MagpieData minimum MeltingT',
 'MagpieData maximum MeltingT',
 'MagpieData range MeltingT',
 'MagpieData mean MeltingT',
 'MagpieData avg_dev MeltingT',
 'MagpieData mode MeltingT',
 'MagpieData minimum Column',
 'MagpieData maximum Column',
 'MagpieData range Column',
 'MagpieData mean Column',
 'MagpieData avg_dev Column',
 'MagpieData mode Column',
 'MagpieData minimum Row',
 'MagpieData maximum Row',
 'MagpieData range Row',
 'MagpieData mean Row',
 'MagpieData avg_dev Row',
 'MagpieData mode Row',
 'MagpieData minimum CovalentRadius',
 'MagpieData maximum CovalentRadius',
 'MagpieData range CovalentRadius',
 'MagpieData mean CovalentRadius',
 'MagpieData avg_dev CovalentRadius',
 'MagpieData mode CovalentRadius',
 'MagpieData minimum Electronegativity',
 'MagpieData maximum Electronegativity',
 'MagpieData range Electronegativity',
 'MagpieData mean Electronegativity',
 'MagpieData avg_dev Electronegativity',
 'MagpieData mode Electronegativity',
 'MagpieData minimum NsValence',
 'MagpieData maximum NsValence',
 'MagpieData range NsValence',
 'MagpieData mean NsValence',
 'MagpieData avg_dev NsValence',
 'MagpieData mode NsValence',
 'MagpieData minimum NpValence',
 'MagpieData maximum NpValence',
 'MagpieData range NpValence',
 'MagpieData mean NpValence',
 'MagpieData avg_dev NpValence',
 'MagpieData mode NpValence',
 'MagpieData minimum NdValence',
 'MagpieData maximum NdValence',
 'MagpieData range NdValence',
 'MagpieData mean NdValence',
 'MagpieData avg_dev NdValence',
 'MagpieData mode NdValence',
 'MagpieData minimum NfValence',
 'MagpieData maximum NfValence',
 'MagpieData range NfValence',
 'MagpieData mean NfValence',
 'MagpieData avg_dev NfValence',
 'MagpieData mode NfValence',
 'MagpieData minimum NValence',
 'MagpieData maximum NValence',
 'MagpieData range NValence',
 'MagpieData mean NValence',
 'MagpieData avg_dev NValence',
 'MagpieData mode NValence',
 'MagpieData minimum NsUnfilled',
 'MagpieData maximum NsUnfilled',
 'MagpieData range NsUnfilled',
 'MagpieData mean NsUnfilled',
 'MagpieData avg_dev NsUnfilled',
 'MagpieData mode NsUnfilled',
 'MagpieData minimum NpUnfilled',
 'MagpieData maximum NpUnfilled',
 'MagpieData range NpUnfilled',
 'MagpieData mean NpUnfilled',
 'MagpieData avg_dev NpUnfilled',
 'MagpieData mode NpUnfilled',
 'MagpieData maximum NdUnfilled',
 'MagpieData range NdUnfilled',
 'MagpieData mean NdUnfilled',
 'MagpieData avg_dev NdUnfilled',
 'MagpieData mode NdUnfilled',
 'MagpieData maximum NfUnfilled',
 'MagpieData range NfUnfilled',
 'MagpieData mean NfUnfilled',
 'MagpieData avg_dev NfUnfilled',
 'MagpieData minimum NUnfilled',
 'MagpieData maximum NUnfilled',
 'MagpieData range NUnfilled',
 'MagpieData mean NUnfilled',
 'MagpieData avg_dev NUnfilled',
 'MagpieData mode NUnfilled',
 'MagpieData minimum GSvolume_pa',
 'MagpieData maximum GSvolume_pa',
 'MagpieData range GSvolume_pa',
 'MagpieData mean GSvolume_pa',
 'MagpieData avg_dev GSvolume_pa',
 'MagpieData mode GSvolume_pa',
 'MagpieData minimum GSbandgap',
 'MagpieData maximum GSbandgap',
 'MagpieData range GSbandgap',
 'MagpieData mean GSbandgap',
 'MagpieData avg_dev GSbandgap',
 'MagpieData mode GSbandgap',
 'MagpieData maximum GSmagmom',
 'MagpieData range GSmagmom',
 'MagpieData mean GSmagmom',
 'MagpieData avg_dev GSmagmom',
 'MagpieData mode GSmagmom',
 'MagpieData minimum SpaceGroupNumber',
 'MagpieData maximum SpaceGroupNumber',
 'MagpieData range SpaceGroupNumber',
 'MagpieData mean SpaceGroupNumber',
 'MagpieData avg_dev SpaceGroupNumber',
 'MagpieData mode SpaceGroupNumber']


# In[7]:
#df_fs_magpie = df.iloc[:,list(range(92,220))]
# df_fs_magpie = df.iloc[:,list(range(85,213))]
df_fs_magpie =df.loc[:, magpie_list]

# In[8]:
df_input_target = df_input_target.drop(['No of Components'], axis=1)

# In[9]:
df_input_target= df_input_target.iloc[:,list(range(0,13))]

# In[10]:
df_features = pd.concat([df_fs_magpie,df_input_target], axis=1)

# # Classification Predictions

# In[11]:
path='model_files//nn_model//classification//'

# In[12]:
import pickle
scaler = pickle.load(open(path+'scaler.pkl','rb'))

df_std = scaler.transform(df_features)

pca_1 = pickle.load(open(path+'pca_1.pkl','rb'))
#std_test = scaler.transform(X_test_std)
#test_pca_1 =  pca_1.transform(std_test)

df_pca =  pca_1.transform(df_std)

print("Classificastion in Progress")

# In[13]:
from tensorflow import keras
model_cat = keras.models.load_model('model_files/nn_model/classification/model_cat.h5')
model_cata = keras.models.load_model('model_files/nn_model/classification/model_cata.h5')
model_catb = keras.models.load_model('model_files/nn_model/classification/model_catb.h5')

y_cat = model_cat.predict(df_pca)

# In[14]:
category = np.where(y_cat[:, 0] > 0.5, 'A', 'B')
####################################################################
"""
## The Crysrtal Structure



"""
st.write("Category :", category[0])
#####################################################################
subcategories = []
y_tensor = []

if np.any(category == 'A'):
    y_subcat = model_cata.predict(df_pca)
    
    for subcat in y_subcat:
        subcategory = []
        y_target = []
        if subcat[0] > 0.33:
            subcategory.append('cubic')
#             y_value = ensemble_model(model_path='model_files/nn_model/cubic/')
            
        elif subcat[1] > 0.33:
            subcategory.append('tetra42m')
#             y_value = ensemble_model(model_path='model_files/nn_model/tetra42m/')
            
        elif subcat[2] > 0.33:
            subcategory.append('ortho222')
#             y_value = ensemble_model(model_path='model_files/nn_model/ortho222/')
            
        subcategories.append(subcategory)
#         y_tensor.append(y_value)
        
else:
    y_subcat = model_catb.predict(df_pca)
    for subcat in y_subcat:
        subcategory = []
        y_target = []
        
        if subcat[0] > 0.5:
            subcategory.append('orthomm2')
#             y_value = ensemble_model(model_path='model_files/nn_model/orthomm2/')
            
        elif subcat[1] > 0.5:
            subcategory.append('hextetramm')
#             y_value = ensemble_model(model_path='model_files/nn_model/hextetramm/')
            
        subcategories.append(subcategory)
#         y_tensor.append(y_value)

####################################################################
"""


"""
st.write("Crystal Structure is :", subcategories[0][0])
#####################################################################
# In[15]:
print("Tensor Predictionsin Progress")

import multiprocessing
import os
import numpy as np
import pickle
import keras

# Define the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a shared dictionary to cache models
manager = multiprocessing.Manager()
model_cache = manager.dict()

def ensemble_model(df_pred, model_path= 'model_files/nn_model/cubic/'):
    
    # Assuming your data is stored in 'data' variable
    df_pred = df_pred.reshape(1, -1)
    # Check if the model is already in the cache
    if model_path in model_cache:
        return model_cache[model_path]

    scaler = pickle.load(open(model_path+'scaler_reg.pkl', 'rb'))
    df_std = scaler.transform(df_pred)

    pca_1 = pickle.load(open(model_path+'pca_reg.pkl', 'rb'))
    df_pca = pca_1.transform(df_std)

    model1 = keras.models.load_model(model_path+'model_1.h5')
    model2 = keras.models.load_model(model_path+'model_2.h5')
    model3 = keras.models.load_model(model_path+'model_3.h5')
    model4 = keras.models.load_model(model_path+'model_4.h5')
    model5 = keras.models.load_model(model_path+'model_5.h5')

    predictions = []

    for model in [model1, model2, model3, model4, model5]:
        pred = model.predict(df_pca)  # Assuming the models have a predict() method
        predictions.append(pred)

    ensemble_prediction = np.mean(predictions, axis=0)  # Average the predicted probabilities across models
    ensemble_prediction = ensemble_prediction.tolist()
    ensemble_prediction = ensemble_prediction[0]

    # Store the prediction in the cache
    model_cache[model_path] = ensemble_prediction
    return ensemble_prediction

# # In[16]:
# df_pca = df_features.values
# ensemble_model(df_pca[5], model_path='model_files/nn_model/hextetramm/')

# In[17]:
print("This has to be printed")
df_predict = df_features.values
y_tensor = []
y_value = []
# subcategories =  []
for item in range(df_pca.shape[0]):
#     print(item, y_cat[item], subcategories[item])
    if y_cat[item][0] > 0.5 and subcategories[item] == ['cubic']:
#         if subcategories[item] == 'cubic':
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/cubic/')
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[0]]]
            
    elif y_cat[item][0] > 0.5 and subcategories[item] == ['tetra42m']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/tetra42m/') 
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[1]]]
            
    elif y_cat[item][0] > 0.5 and subcategories[item] == ['ortho222']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/ortho222/') 
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[1], 0], [0, 0, 0, 0, 0, y[2]]]
            
    elif y_cat[item][0] < 0.5 and subcategories[item] == ['orthomm2']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/orthomm2/')
        y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[1], 0, 0], [y[2], y[3], y[4], 0, 0, 0]]
            
    elif y_cat[item][0] < 0.5 and subcategories[item] == ['hextetramm']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/hextetramm/') 
        y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[0], 0, 0], [y[1], y[1], y[2], 0, 0, 0]]

    y_tensor.append(y_value)

# In[18]:
# y_tensor
my_tensor = np.array(y_tensor[0])
my_tensor_visual = np.around(my_tensor, decimals=3)
my_tensor_visual = np.where(my_tensor_visual == 0.0, '0', my_tensor_visual)
# my_tensor = np.trim_zeros(my_tensor_visual.flatten(), 'b').reshape(my_tensor_visual.shape)
# my_df = pd.dataframe(y_tensor[0])
####################################################################
"""
## The Piezo Tensor is


"""

# Define the matrix size
rows = 3
cols = 6

# Create the LaTeX matrix string
matrix = r"\begin{pmatrix}"

# Append the values to the matrix string
for i in range(rows):
    for j in range(cols):
        matrix += str(my_tensor_visual[i, j])
        if j < cols - 1:
            matrix += " & "
        else:
            matrix += r" \\"

# Close the matrix string
matrix += r"\end{pmatrix}"

# Display LaTeX matrix using st.latex()
st.latex(matrix)

#####################################################################
# In[19]:
# y_tensor[1]
##################################################################################################
# End of Prediction

######################################################################################################

import subprocess

# Define the path to the file
file_path = "~/Sachin Research/Piezoelectric_Research/Piezoelectric_codes/Codes_May 1/Piezoelectric Tensors/"

# Create a NumPy array
# my_array = np.array([1, 2, 3, 4, 5])

# Save the NumPy array as a tensor file
tensor_path = 'my_tensor.npy'
np.save(tensor_path, my_tensor)

# Add a download button to the Streamlit app
if st.button("Download NumPy Array as Tensor"):
    with open(tensor_path, "rb") as file:
        contents = file.read()
        st.download_button(label="Click here to download", data=contents, file_name="my_array.npy")
        
#####################################################################################################
# Crystal Rotations
from crystal_rotation import *

# Take an input from the user
crystal_rotations = st.sidebar.checkbox("Perform Crystal rotation for rotated tensor")
if crystal_rotations:
   # Create input fields in a compact and aligned layout
    cols = st.sidebar.columns([1, 1, 1, 1])
    cols[0].write('<p style="margin-bottom: 0.2em;">Euler Angle:</p>', unsafe_allow_html=True)  
    cols[1].write("Psi /n(*)")
    cols[2].write("Theta")
    cols[3].write("Phi")
    psi = cols[1].text_input("", value="30")
    theta = cols[2].text_input("", value="90")
    phi = cols[3].text_input("", value="150")


    # Perform tensor rotation with crystal rotations
    tensor_prime = tensor_rotation(my_tensor, psi=float(psi), theta=float(theta), phi=float(phi))

    tensor_rotation_plot(my_tensor, phi_vals = 30,order=[1,1])

    # st.write("Crystal Rotation :", tensor_prime)

    # Define the matrix size
    tensor_prime_visual = np.around(tensor_prime, decimals=3)
    tensor_prime_visual = np.where(tensor_prime_visual == 0.0, '0', tensor_prime_visual)
    # Create the LaTeX matrix string
    matrix = r"\begin{pmatrix}"
    cols =6
    # Append the values to the matrix string
    for i in range(rows):
        for j in range(cols):
            matrix += str(tensor_prime_visual[i, j])
            if j < cols - 1:
                matrix += " & "
            else:
                matrix += r" \\"

    # Close the matrix string
    matrix += r"\end{pmatrix}"

    # Display LaTeX matrix using st.latex()
    st.latex(matrix)
