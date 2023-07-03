from functions import *

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

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import matplotlib.pyplot as plt

ef= ElementFraction()
stc = StrToComposition()

def prediction_model(df_piezo, cat = 'B', point=''):   
##############################################
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

    def ensemble_modelt(df_pred, model_path= 'model_files/nn_model/cubic/'):

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

#         for model in [model1, model2, model3, model4, model5]:
        for model in [model1]:
            pred = model.predict(df_pca)  # Assuming the models have a predict() method
            print("pred", pred)
            predictions.append(pred)

        ensemble_prediction = np.mean(predictions, axis=0)  # Average the predicted probabilities across models
        ensemble_prediction = ensemble_prediction.tolist()
        ensemble_prediction = ensemble_prediction[0]

        # Store the prediction in the cache
    #     model_cache[model_path] = ensemble_prediction
        return ensemble_prediction
        

##############################################################################################
################################################################################################

    def input_features(df_piezo):
        import numpy as np

        df_piezo = stc.featurize_dataframe(df_piezo, "formula_pretty",ignore_errors=True,return_errors=True)
        df_piezo = ef.featurize_dataframe(df_piezo, "composition",ignore_errors=True,return_errors=True)

        from matminer.featurizers.composition import ElementProperty
        featurizer = ElementProperty.from_preset('magpie')
        df_piezo = featurizer.featurize_dataframe(df_piezo, col_id='composition')
        #y = bg_data_featurized['gap expt']

#         %run functions.ipynb
        df, df_input_target = properties_calculation(df_piezo)

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

        df_fs_magpie =df.loc[:, magpie_list]
        df_input_target = df_input_target.drop(['No of Components'], axis=1)
        df_input_target= df_input_target.iloc[:,list(range(0,13))]
        df_features = pd.concat([df_fs_magpie,df_input_target], axis=1)
        return df_features

##############################################################################################
################################################################################################

    df_features= input_features(df_piezo)
    path='model_files//nn_model//classification//'

    import pickle
    scaler = pickle.load(open(path+'scaler.pkl','rb'))
    df_std = scaler.transform(df_features)
    pca_1 = pickle.load(open(path+'pca_1.pkl','rb'))
    df_pca =  pca_1.transform(df_std)

    from tensorflow import keras
    model_cat = keras.models.load_model('model_files/nn_model/classification/model_cat.h5')
    model_cata = keras.models.load_model('model_files/nn_model/classification/model_cata.h5')
    model_catb = keras.models.load_model('model_files/nn_model/classification/model_catb.h5')

    y_cat = model_cat.predict(df_pca)
    if cat=='NA':
        category = np.where(y_cat[:, 0] > 0.5, 'A', 'B')
    else: 
        category = np.full(y_cat.shape[0], cat)
        
        
    subcategories = []
    y_tensor = []
    if np.any(category == 'A'):
        y_subcat = model_cata.predict(df_pca)

        for subcat in y_subcat:
            subcategory = []
            y_target = []
            if subcat[0] > 0.33:
                subcategory.append('cubic')
            elif subcat[1] > 0.33:
                subcategory.append('tetra42m')
            elif subcat[2] > 0.33:
                subcategory.append('ortho222')

            subcategories.append(subcategory)

    else:
        y_subcat = model_catb.predict(df_pca)
        for subcat in y_subcat:
            subcategory = []
            y_target = []

            if subcat[0] > 0.5:
                subcategory.append('orthomm2')
            elif subcat[1] > 0.5:
                subcategory.append('hextetramm')

            subcategories.append(subcategory)

    print(subcategories)
    if point!= '':
        # Replace all values in `subcategory` with "new"
        subcategories = np.where(subcategories != '', point, subcategories)

    print("* * * ")
    print(subcategories)
    df_predict = df_features.values
    y_tensor = []
    y_value = []
    for item in range(df_pca.shape[0]):
        if category[item]=='A' and subcategories[item] == ['cubic']:
            y = ensemble_modelt(df_predict[item], model_path='model_files/nn_model/cubic/')
            y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[0]]]

        elif category[item]=='A' and subcategories[item] == ['tetra42m']:
            y = ensemble_modelt(df_predict[item], model_path='model_files/nn_model/tetra42m/') 
            y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[1]]]

        elif category[item]=='A' and subcategories[item] == ['ortho222']:
            y = ensemble_modelt(df_predict[item], model_path='model_files/nn_model/ortho222/') 
            y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[1], 0], [0, 0, 0, 0, 0, y[2]]]

        elif category[item]=='B' and subcategories[item] == ['orthomm2']:
            y = ensemble_modelt(df_predict[item], model_path='model_files/nn_model/orthomm2/')
            y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[1], 0, 0], [y[2], y[3], y[4], 0, 0, 0]]

        elif category[item]=='B' and subcategories[item] == ['hextetramm']:
            y = ensemble_modelt(df_predict[item], model_path='model_files/nn_model/hextetramm/') 
            y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[0], 0, 0], [y[1], y[1], y[2], 0, 0, 0]]
       
        y_tensor.append(y_value)
    return category, subcategories, y_tensor

#####################################################################################################################################
########################################################################################################################################

def tensor_rotation_optimization(eo, phi_vals, order=[0, 0]):
    # Define the angles theta and psi
    theta_vals = np.linspace(0, np.pi, 50)
    psi_vals = np.linspace(0, 2 * np.pi, 50)

    # Initialize arrays to store the maximum values
    max_e11_vals = []
    max_theta_vals = []
    max_psi_vals = []

    # Iterate over each phi value
    for phi in phi_vals:
        # Initialize array to store the e'11 matrix elements for each combination of psi, theta, and phi
        e_prime_11 = np.zeros((len(psi_vals), len(theta_vals)))

        # Calculate the e'11 matrix elements for each combination of psi, theta, and phi
        for i, psi in enumerate(psi_vals):
            for j, theta in enumerate(theta_vals):
                # Compute the elements of matrix A
                A = np.array([
                    [np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                     np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi),
                     np.sin(theta) * np.sin(phi)],
                    [-np.cos(theta) * np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi),
                     np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                     np.cos(theta) * np.sin(psi)],
                    [np.sin(theta) * np.sin(psi), -np.cos(phi) * np.sin(theta), np.cos(theta)]
                ])

                # Compute the elements of matrix N
                N = np.array([
                    [A[0, 0]**2, A[1, 0]**2, A[2, 0]**2, 2 * A[1, 0] * A[2, 0], 2 * A[2, 0] * A[0, 0], 2 * A[0, 0] * A[1, 0]],
                    [A[0, 1]**2, A[1, 1]**2, A[2, 1]**2, 2 * A[1, 1] * A[2, 1], 2 * A[2, 1] * A[0, 1], 2 * A[0, 1] * A[1, 1]],
                    [A[0, 2]**2, A[1, 2]**2, A[2, 2]**2, 2 * A[1, 2] * A[2, 2], 2 * A[2, 2] * A[0, 2], 2 * A[0, 2] * A[1, 2]],
                    [A[0, 1] * A[0, 2], A[1, 1] * A[1, 2], A[2, 1] * A[2, 2], A[1, 1] * A[2, 2] + A[2, 1] * A[1, 2], A[0, 1] * A[2, 2] + A[2, 1] * A[0, 2], A[1, 1] * A[0, 2] + A[0, 1] * A[1, 2]],
                    [A[0, 2] * A[0, 0], A[1, 2] * A[1, 0], A[2, 2] * A[2, 0], A[1, 2] * A[2, 0] + A[2, 2] * A[1, 0], A[2, 2] * A[0, 0] + A[0, 2] * A[2, 0], A[0, 2] * A[1, 0] + A[0, 0] * A[1, 2]],
                    [A[0, 0] * A[0, 1], A[1, 0] * A[1, 1], A[2, 0] * A[2, 1], A[1, 0] * A[2, 1] + A[2, 0] * A[1, 1], A[2, 0] * A[0, 1] + A[0, 0] * A[2, 1], A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1]]
                ])

                # Compute the elements of the e' matrix
                e_prime = np.zeros((3, 6))
                for l in range(3):
                    for m in range(6):
                        for n in range(3):
                            for o in range(6):
                                e_prime[l, m] += A[l, n] * eo[n, o] * N[o, n]

                # Store the e'11 matrix element at the corresponding indices
                e_prime_11[i, j] = e_prime[order[0], order[1]]

        # Find the maximum point and its value for each order
        max_index = np.unravel_index(np.argmax(e_prime_11), e_prime_11.shape)
        max_theta = np.degrees(theta_vals[max_index[1]])
        max_psi = np.degrees(psi_vals[max_index[0]])
        max_e11 = e_prime_11[max_index]

        # Append the maximum values to the respective arrays
        max_e11_vals.append(max_e11)
        max_theta_vals.append(max_theta)
        max_psi_vals.append(max_psi)

    # Plot the maximum values for each order as a function of phi
    orders = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
              [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]]


    fig = go.Figure()
    for i, order in enumerate(orders):
        fig.add_trace(go.Scatter(x=np.degrees(phi_vals), y=np.array(max_e11_vals[i]).flatten(), mode='lines', name=f'Order {order}'))

    fig.update_layout(title='Maximum e11 values as a function of phi', xaxis_title='Phi (degrees)', yaxis_title='e11')
    fig.show()



#####################################################################################################################
# Latex matrix
# Define the matrix size
def latex_tensor_visual(my_tensor):
    
    my_tensor_visual = np.around(my_tensor, decimals=3)
    my_tensor_visual = np.where(my_tensor_visual == 0.0, '0', my_tensor_visual)

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
    return matrix
#########################################################################################################################
