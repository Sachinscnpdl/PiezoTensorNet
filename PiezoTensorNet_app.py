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
from prediction_ML import *
import streamlit as st
import pandas as pd

# Add a dropdown to select a pre-defined formula
import streamlit as st
import pandas as pd

# Create two tabs in the sidebar
tab_options = ["New piezoelectric design", "Rapid Piezo-performance design"]
selected_tab = st.sidebar.radio("Select Tab", tab_options)

# Initialize empty DataFrame to store selected formulas
df_selected_formulas = pd.DataFrame()

# Add input block for "New piezoelectric design"
if selected_tab == "New piezoelectric design":
    # Add a dropdown to select a pre-defined formula
    predefined_formulas = ['Ba0.85Ca0.15Ti0.92Zr0.07Hf0.01O3', 'Ba0.84Ca0.15Sr0.01Ti0.90Zr0.10O3', 'BaTiO3']
    selected_predefined_formula = st.selectbox('Select a pre-defined formula', predefined_formulas)

    # If a pre-defined formula is selected, add it to the DataFrame
    if selected_predefined_formula:
        df_selected_formulas = pd.concat([df_selected_formulas, pd.DataFrame({'S.N': [len(df_selected_formulas) + 1], 'formula_pretty': [selected_predefined_formula]})], ignore_index=True)

# Add input block for "Rapid Piezo-performance design"
if selected_tab == "Rapid Piezo-performance design":
    # Add an input box for the custom formula
    custom_formula = st.text_input('Enter the custom formula')

    # If a custom formula is entered, add it to the DataFrame
    if custom_formula:
        df_selected_formulas = pd.concat([df_selected_formulas, pd.DataFrame({'S.N': [len(df_selected_formulas) + 1], 'formula_pretty': [custom_formula]})], ignore_index=True)

# Display the selected formulas
if not df_selected_formulas.empty:
    st.write('Selected Formulas:')
    st.dataframe(df_selected_formulas)

df_piezo = df_selected_formulas


cat, subcategories, y_tensor = prediction_model(df_piezo, cat = 'B', point='')

"""

# Welcome to PiezoTensorNet!


"""


st.write("Crystal Structure is :", subcategories[0][0])
#####################################################################
# In[15]:
print("Tensor Predictionsin Progress")

my_tensor = np.array(y_tensor[0])
# my_tensor_visual = np.around(my_tensor, decimals=3)
# my_tensor_visual = np.where(my_tensor_visual == 0.0, '0', my_tensor_visual)
# my_tensor = np.trim_zeros(my_tensor_visual.flatten(), 'b').reshape(my_tensor_visual.shape)
# my_df = pd.dataframe(y_tensor[0])
####################################################################
"""
## The Piezo Tensor is


"""

# # Define the matrix size
# rows = 3
# cols = 6

# # Create the LaTeX matrix string
# matrix = r"\begin{pmatrix}"

# # Append the values to the matrix string
# for i in range(rows):
#     for j in range(cols):
#         matrix += str(my_tensor_visual[i, j])
#         if j < cols - 1:
#             matrix += " & "
#         else:
#             matrix += r" \\"

# # Close the matrix string
# matrix += r"\end{pmatrix}"

# latex_tanser_visual(my_tensor)

# Display LaTeX matrix using st.latex()
st.latex(latex_tanser_visual(my_tensor))
#####################################################################

# Piezo Tensor Representations
if subcategories[0][0]=='cubic':
    image = "plots/CAT_A.png"
elif subcategories[0][0]=='tetra42m':
    image = "plots/CAT_A.png"    
elif subcategories[0][0]=='ortho222':
    image = "plots/CAT_A.png"
    
elif subcategories[0][0]=='orthomm2':
    image = "plots/orthomm2.png"
elif subcategories[0][0]=='hextetramm':
    image = "plots/hextetra.png"

caption = "Piezoelectric Tensor Visualization"
size = (300, 200)  # Custom size in pixels
position = "right"  # Options: "left", "centered", "right"

st.image(image,width=200, caption=caption)

# st.image(image, caption="Image Caption", width=10, use_column_width=True)

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
    cols[0].write('<p style="margin-bottom: -0.2em;">Euler Angle:</p>', unsafe_allow_html=True)  
    cols[1].write("Psi ")
    cols[2].write("Theta")
    cols[3].write("Phi")
    psi = cols[1].text_input("", value="30")
    theta = cols[2].text_input("", value="90")
    phi = cols[3].text_input("", value="150")

    # Perform tensor rotation with crystal rotations
    tensor_prime = tensor_rotation(my_tensor, psi=float(psi), theta=float(theta), phi=float(phi))

#     tensor_rotation_plot(my_tensor, phi_vals = 30,order=[1,1])

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
    
    phi = st.sidebar.slider("Enter angle phi:", min_value=0.0, max_value=360.0, value=0.0, step=10.0)

    block = st.sidebar.columns([1, 1, 1])
    block[0].write('<p style="margin-bottom: -0.2em;">Show plot of Rotated tensor:</p>', unsafe_allow_html=True)
    block[1].write("Row element", unsafe_allow_html=True)
    block[2].write("Column element", unsafe_allow_html=True)

    i = block[1].text_input("Row", value="1", key="row_element")
    j = block[2].text_input("Column", value="1", key="column_element")

    i = int(i)
    j = int(j)


  
    
#     i = st.sidebar.number_input("Enter matrix order i:", min_value=1, max_value=3, value=1, step=1)
#     j = st.sidebar.number_input("Enter matrix order j:", min_value=1, max_value=6, value=1, step=1)
    
    crystal_plot, max_e, max_theta, max_psi, phi_vals = tensor_rotation_plot(my_tensor, phi = phi, order=[i-1,j-1])
    st.plotly_chart(crystal_plot)
    
    rot_optimization = tensor_rotation_optimization(my_tensor, order=[i-1,j-1])
    st.plotly_chart(rot_optimization)
