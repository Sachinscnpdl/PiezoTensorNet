from functions import *


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

        %run functions.ipynb
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
