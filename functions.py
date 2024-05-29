#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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
import numpy as np
import pandas as pd

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import matplotlib.pyplot as plt

ef= ElementFraction()
stc = StrToComposition()

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



def elements_occurance(df):
    df = df.loc[:, (df != 0).any(axis=0)]

    cols = list(df.columns.values)    #Make a list of all of the columns in the df
    set = df.astype(bool).sum(axis=0) # Extract the occurance of each element in the alloys

    element_df = set.to_frame()      # Convert extracted the occurance of each element in dataframe

    element_occurancy = element_df[7:]
    element_occurancy.columns =['Occurance']
    return df


# In[ ]:


def df_element_number(df):
    # Add a column of "Number of component" & "component" in each alloy system
    prop = []
    for number in range(len(df['formula_pretty'])):
        mpea = df['composition'][number]
        element = list(Composition(mpea).as_dict().keys()) # List element present in Alloys ['Al', 'Cr', 'Fe', 'Ni', 'Mo']
        prop.append([len(element), " ".join(element)])

        prop_data = pd.DataFrame(prop, columns=['No of Components', 'Component'])
    df = pd.concat([df, prop_data], axis = 1)
    return df


# In[ ]:


def element_number(df, fig_title='Elements Number', fig_name='element_number'):
    import os
    import matplotlib
    import matplotlib.ticker as tck
    import seaborn as sns
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    ax = sns.countplot(x='No of Components', data=df)
    
    plt.rcParams.update({'font.size': 20})
    
    ax.set_title('Number of elements', fontdict={'size': 24, 'color': 'blue'})
    #ax.bar_label(ax.containers[0], fontproperties={'size': 18})
        
    ax.set_xlabel('Element Numbers', fontdict={'size': 20, 'color': 'r'})
    ax.set_ylabel('Count', fontdict={'size': 20, 'color': 'r'})
    
    plt.ylim(0, 820,400)
    
    plt.tick_params(axis='both', which='both', length=5, width=1.5,color='black')
    
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    
    plt.savefig("plots//element_number",dpi=1200, bbox_inches='tight')

    plt.show()


# In[1]:


def element_occurrence(df,limit_value=8, fig_title='Hardness/ Elongation', fig_name='element_occurrence'):
    
    df = df.loc[:, (df != 0).any(axis=0)]

    cols = list(df.columns.values)    #Make a list of all of the columns in the df
    set = df.astype(bool).sum(axis=0) # Extract the occurance of each element in the alloys

    element_df = set.to_frame()      # Convert extracted the occurance of each element in dataframe

    element_occurancy = element_df[limit_value:-150]
    element_occurancy.columns =['Occurrence']
    element_occurancy = element_occurancy.sort_values('Occurrence')
    
    
    plt.figure(figsize=(36,20))
    
    df = element_occurancy
    print(df)
    
    mask = df['Occurrence'] <= 29
    df1 = df[mask]
    df2_3 = df[~mask]
    
    mask2 = df2_3['Occurrence'] < 75
    df2 = df2_3[mask2]
    df3 = df2_3[~mask2]
    
    import matplotlib
    #matplotlib.use('agg')
    def plot_hor_bar(subplot, data, title = 'title', xlabel = 'Occurrence'):
        print('lenght:  ',len(data))
        plt.subplot(1,3,subplot)
        ax = sns.barplot(x=data['Occurrence'],y=data.index, data=data)
        #ax.bar_label(ax.containers[0], fontproperties={'size': 24})
        plt.title(title,
                  fontsize=36, color='b')
        plt.xlabel(xlabel, fontsize=42, color='b')
        plt.xticks(fontsize=36)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        plt.ylabel(None)
        plt.yticks(fontsize=40,color='black')
        plt.tick_params(axis='both', which='both', length=15, width=5,color='black')
        #plt.pause(.01)
        sns.despine(left=True)
        ax.grid(False)
        ax.tick_params(bottom=True, left=False)

        return None
 
    
    plot_hor_bar(1, df1, title = 'Elements Occurring Less than 25', xlabel = ' ')
    plot_hor_bar(2, df2, title = 'Elements Occurring from 25 to 75', xlabel = 'Occurrence counts')
    plot_hor_bar(3, df3, title = 'Elements Occurring more than 75', xlabel = ' ')
    #plt.title('Elements Occurrence counts')
    
    plt.savefig("plots//element_occurrence",dpi=200, bbox_inches='tight')


    plt.show()


# In[2]:


def properties_calculation(dataframe):
    
    # Import csv files "Midema" to calculte input features
    elem_prop_data = pd.read_csv('csv/Miedema.csv')
    VEC_elements = elem_prop_data.set_index('element')['valence_electrons'].to_dict()
    shear_modulus_g = elem_prop_data.set_index('element')['shear_modulus'].to_dict()
    bulk_modulus_b = elem_prop_data.set_index('element')['compressibility'].to_dict()
    
    # Input featurs calculation
    df = dataframe
    properties = []
    for number in range(len(df['formula_pretty'])):

        mpea = df['composition'][number]
        #print(mpea)
        #print(Composition(mpea).as_dict().keys())
        element = list(Composition(mpea).as_dict().keys()) # List element present in Alloys ['Al', 'Cr', 'Fe', 'Ni', 'Mo']
        #print(element)
        fraction_composition = list(Composition(mpea).as_dict().values()) # List Fraction composition of corresponding element in an Alloy eg. [1.0, 1.0, 1.0, 1.0, 1.0]
        #print(fraction_composition)
        total_mole = sum(fraction_composition) # Sum of elemental composition
        #print(total_mole)

        atomic_number = []
        bulk_modulus = []
        shear_modulus = []
        molar_heat = []
        thermal_conductivity = []
        mole_fraction = []
        X_i = []
        r_i = []
        Tm_i = []
        VEC_i= []
        R = 8.314

        for i in element:

            atomic_number.append(Element(i).Z)
            #molar_heat.append(Cp_dict[i])

            bulk_b =Element(i).bulk_modulus

            if type(bulk_b) == type(None):
                for j in bulk_modulus_b: bulk_b = (bulk_modulus_b.get(j))       

            bulk_modulus.append(bulk_b)

            #print(bulk_modulus)

            shear_g = (Element(i).rigidity_modulus)
            if type(shear_g) == type(None):
                for s in shear_modulus_g: shear_g = ((shear_modulus_g.get(s)))
            shear_modulus.append(shear_g)

            thermal_conductivity.append(Element(i).thermal_conductivity)
            mole_fraction.append(Composition(mpea).get_atomic_fraction(i)) # Calculates mole fraction of mpea using "Composition" functions

            X_i.append(Element(i).X) # Calculate individual electronegativity using "Element" function

            r_i.append(Element(i).atomic_radius) if Element(i).atomic_radius_calculated == None else r_i.append(Element(i).atomic_radius_calculated) # There are two functions present in Element␣class of pymatgen, so here checking using if conditional in both functions␣to not miss any value
            Tm_i.append(Element(i).melting_point) # Calculating melting point of every element using "Element" class and function
            
            try: VEC_i.append(DemlData().get_elemental_property(Element(i),"valence")) # VEC is also present in 2 locations in matminer, first is the␣function "DemlData()"
            except KeyError:
                if i in VEC_elements: VEC_i.append(float(VEC_elements.get(i))) #In case data is not present in "DemlData()" function, there is a csv file␣inside matminer opened earlier as "elem_prop_data" in the very first cell
                if i=='Xe': VEC_i.append(float(2))
            #print(number, VEC_i)
            #print('VEC: ',i, VEC_elements.get('Xe'))
        
        # Average Atomic Number
        AN = sum(np.multiply(mole_fraction, atomic_number))
        #print(AN)

        # Average Molar Heat coefficient
        #Cp_bar = sum(np.multiply(mole_fraction, molar_heat))    
        #print(Cp_bar)

        #term_Cp = (1-np.divide(molar_heat, Cp_bar))**2
        #del_Cp = sum(np.multiply(mole_fraction, term_Cp))**0.5 

        # Thermal Conductivity
        k = sum(np.multiply(mole_fraction, thermal_conductivity))

        # Bulk Modolus
        bulk = sum(np.multiply(mole_fraction, bulk_modulus)) # Bulk modulus of 'Zr' not present
        
        # Bulk modolus asymmetry
        term_bulk = (1-np.divide(bulk_modulus, bulk))**2
        del_bulk = sum(np.multiply(mole_fraction, term_bulk))**0.5         

        # Shear Modolus
        shear= sum(np.multiply(mole_fraction, shear_modulus))
        
        # Shear modolus asymmetry
        term_shear = (1-np.divide(shear_modulus, shear))**2
        del_shear = sum(np.multiply(mole_fraction, term_shear))**0.5         

        # Calculation of Atomic Radius Difference (del)

        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2
        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5 
        #print(number,element,mole_fraction,r_i,r_bar,term,atomic_size_difference)


        # Electronegativity (del_X)
        X_bar = sum(np.multiply(mole_fraction, X_i))
        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i,X_bar))**2)))**0.5
        #term_X = (1-np.divide(X_i, X_bar))**2
        #del_Chi = (sum(np.multiply(mole_fraction, term_X)))**0.5 

        # Difference Melting Temperature
        T_bar = sum(np.multiply(mole_fraction, Tm_i))
        del_Tm =(sum(np.multiply(mole_fraction, (np.subtract(Tm_i,T_bar))**2)))**0.5

        # Average Melting Temperature
        Tm = sum(np.multiply(mole_fraction, Tm_i))    

        # Valence Electron Concentration
        VEC = sum(np.multiply(mole_fraction, VEC_i))
        #print(VEC)

        # Entropy of mixing
        #del_Smix = -WenAlloys().compute_configuration_entropy(mole_fraction)*1000 #WenAlloys class imported from matminer library
        del_Smix = -R*sum(np.multiply(mole_fraction, np.log(mole_fraction)))


        # Geometrical parameters
        if atomic_size_difference == 0:  atomic_size_difference = 1e-9
        lemda = np.divide(del_Smix, (atomic_size_difference)**2)
        #print(number,del_Smix,atomic_size_difference, lemda)

        #parameter = Tm*del_Smix/abs(del_Hmix) 
        #print(number,"lemda, parameter", lemda, parameter)


        #properties.append([len(element), " ".join(element), " ".join(list(map(str, fraction_composition))),total_mole, round(sum(mole_fraction),1), atomic_size_difference, round(del_Chi, 4),del_Tm, Tm, VEC, AN, k, bulk,del_bulk,shear,del_shear, round(del_Smix, 4),round(lemda,4), round(del_Hmix, 4),round(parameter,4)])
        properties.append([len(element), " ".join(element), " ".join(list(map(str, fraction_composition))),total_mole, round(sum(mole_fraction),1), atomic_size_difference, round(del_Chi, 4),del_Tm, Tm, VEC, AN, k, bulk,del_bulk,shear,del_shear, round(del_Smix, 4),round(lemda,4)])
    #prop_data = pd.DataFrame(properties, columns=['No of Components','Component','Moles of individual Components', 'Total Moles', 'Sum of individual MoleFractions', '$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K','B', 'ΔB','G', 'ΔG','ΔSmix','$\lambda$', 'ΔHmix','$\Omega$'])
    prop_data = pd.DataFrame(properties, columns=['No of Components','Component','Moles of individual Components', 'Total Moles', 'Sum of individual MoleFractions', '$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K','B', 'ΔB','G', 'ΔG', 'ΔSmix','$\lambda$',])

    df = pd.concat([df, prop_data], axis = 1)
    
    df_input_target = df.iloc[:,[-18,-13,-12,-11,-10, -9, -8, -7, -6,-5, -4, -3, -2, -1,4,5,2,6,7]]
    
    return(df,df_input_target)


# In[ ]:


def properties_calculation_old(dataframe):
    
    # Import csv files "Midema" to calculte input features
    elem_prop_data = pd.read_csv('csv/Miedema.csv')
    VEC_elements = elem_prop_data.set_index('element')['valence_electrons'].to_dict()
    shear_modulus_g = elem_prop_data.set_index('element')['shear_modulus'].to_dict()
    bulk_modulus_b = elem_prop_data.set_index('element')['compressibility'].to_dict()
    
    # Input featurs calculation
    df = dataframe
    properties = []
    for number in range(len(df['formula_pretty'])):

        mpea = df['composition'][number]
        #print(mpea)
        #print(Composition(mpea).as_dict().keys())
        element = list(Composition(mpea).as_dict().keys()) # List element present in Alloys ['Al', 'Cr', 'Fe', 'Ni', 'Mo']
        #print(element)
        fraction_composition = list(Composition(mpea).as_dict().values()) # List Fraction composition of corresponding element in an Alloy eg. [1.0, 1.0, 1.0, 1.0, 1.0]
        #print(fraction_composition)
        total_mole = sum(fraction_composition) # Sum of elemental composition
        #print(total_mole)

        atomic_number = []
        bulk_modulus = []
        shear_modulus = []
        molar_heat = []
        thermal_conductivity = []
        mole_fraction = []
        X_i = []
        r_i = []
        Tm_i = []
        VEC_i= []
        R = 8.314

        for i in element:

            atomic_number.append(Element(i).Z)
            #molar_heat.append(Cp_dict[i])

            bulk_b =Element(i).bulk_modulus

            if type(bulk_b) == type(None):
                for j in bulk_modulus_b: bulk_b = (bulk_modulus_b.get(j))       

            bulk_modulus.append(bulk_b)

            #print(bulk_modulus)

            shear_g = (Element(i).rigidity_modulus)
            if type(shear_g) == type(None):
                for s in shear_modulus_g: shear_g = ((shear_modulus_g.get(s)))
            shear_modulus.append(shear_g)

            thermal_conductivity.append(Element(i).thermal_conductivity)
            mole_fraction.append(Composition(mpea).get_atomic_fraction(i)) # Calculates mole fraction of mpea using "Composition" functions

            X_i.append(Element(i).X) # Calculate individual electronegativity using "Element" function

            r_i.append(Element(i).atomic_radius) if Element(i).atomic_radius_calculated == None else r_i.append(Element(i).atomic_radius_calculated) # There are two functions present in Element␣class of pymatgen, so here checking using if conditional in both functions␣to not miss any value
            Tm_i.append(Element(i).melting_point) # Calculating melting point of every element using "Element" class and function
            
            try: VEC_i.append(DemlData().get_elemental_property(Element(i),"valence")) # VEC is also present in 2 locations in matminer, first is the␣function "DemlData()"
            except KeyError:
                if i in VEC_elements: VEC_i.append(float(VEC_elements.get(i))) #In case data is not present in "DemlData()" function, there is a csv file␣inside matminer opened earlier as "elem_prop_data" in the very first cell
                if i=='Xe': VEC_i.append(float(2))
            #print(number, VEC_i)
            #print('VEC: ',i, VEC_elements.get('Xe'))
        
        # Average Atomic Number
        AN = sum(np.multiply(mole_fraction, atomic_number))
        #print(AN)

        # Average Molar Heat coefficient
        #Cp_bar = sum(np.multiply(mole_fraction, molar_heat))    
        #print(Cp_bar)

        #term_Cp = (1-np.divide(molar_heat, Cp_bar))**2
        #del_Cp = sum(np.multiply(mole_fraction, term_Cp))**0.5 

        # Thermal Conductivity
        k = sum(np.multiply(mole_fraction, thermal_conductivity))

        # Bulk Modolus
        bulk = sum(np.multiply(mole_fraction, bulk_modulus)) # Bulk modulus of 'Zr' not present
        
        # Bulk modolus asymmetry
        term_bulk = (1-np.divide(bulk_modulus, bulk))**2
        del_bulk = sum(np.multiply(mole_fraction, term_bulk))**0.5         

        # Shear Modolus
        shear= sum(np.multiply(mole_fraction, shear_modulus))
        
        # Shear modolus asymmetry
        term_shear = (1-np.divide(shear_modulus, shear))**2
        del_shear = sum(np.multiply(mole_fraction, term_shear))**0.5         

        # Calculation of Atomic Radius Difference (del)

        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2
        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5 
        #print(number,element,mole_fraction,r_i,r_bar,term,atomic_size_difference)


        # Electronegativity (del_X)
        X_bar = sum(np.multiply(mole_fraction, X_i))
        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i,X_bar))**2)))**0.5
        #term_X = (1-np.divide(X_i, X_bar))**2
        #del_Chi = (sum(np.multiply(mole_fraction, term_X)))**0.5 

        # Difference Melting Temperature
        T_bar = sum(np.multiply(mole_fraction, Tm_i))
        del_Tm =(sum(np.multiply(mole_fraction, (np.subtract(Tm_i,T_bar))**2)))**0.5

        # Average Melting Temperature
        Tm = sum(np.multiply(mole_fraction, Tm_i))    

        # Valence Electron Concentration
        #print(mole_fraction.shape, VEC_i.shape)
        #print(number)
        #print(mole_fraction)
        #print(VEC_i)
        VEC = sum(np.multiply(mole_fraction, VEC_i))
        #print(VEC)

        # Entropy of mixing
        #del_Smix = -WenAlloys().compute_configuration_entropy(mole_fraction)*1000 #WenAlloys class imported from matminer library
        del_Smix = -R*sum(np.multiply(mole_fraction, np.log(mole_fraction)))


        HEA = element
        #print(len(mole_fraction), len(HEA))


        # Enthalpy of mixing
        AB = []
        C_i_C_j = []
        del_Hab = []
        for item in range(len(HEA)):
            for jitem in range(item, len(HEA)-1):
                AB.append(HEA[item] + HEA[jitem+1])
                C_i_C_j.append(mole_fraction[item]*mole_fraction[jitem+1])
                #del_Hab.append(round(Miedema().deltaH_chem([HEA[item], HEA[jitem+1]], [0.5, 0.5], 'ss'),3)) # Calculating binary entropy of mixing at 0.5-0.5␣ (equal) composition using Miedema class of "matminer" library
                del_Hab.append(MixingEnthalpy().get_mixing_enthalpy(Element(HEA[item]), Element(HEA[jitem+1]))) # Matminer MixingOfEnthalpy
                #print(HEA)
                #print(del_Hab)
                #print(" ")

        omega = np.multiply(del_Hab, 4)
        del_Hmix = sum(np.multiply(omega, C_i_C_j))

        # Geometrical parameters
        if atomic_size_difference == 0:  atomic_size_difference = 1e-9
        lemda = np.divide(del_Smix, (atomic_size_difference)**2)
        #print(number,del_Smix,atomic_size_difference, lemda)

        #parameter = Tm*del_Smix/abs(del_Hmix) 
        #print(number,"lemda, parameter", lemda, parameter)


        #properties.append([len(element), " ".join(element), " ".join(list(map(str, fraction_composition))),total_mole, round(sum(mole_fraction),1), atomic_size_difference, round(del_Chi, 4),del_Tm, Tm, VEC, AN, k, bulk,del_bulk,shear,del_shear, round(del_Smix, 4),round(lemda,4), round(del_Hmix, 4),round(parameter,4)])
        properties.append([len(element), " ".join(element), " ".join(list(map(str, fraction_composition))),total_mole, round(sum(mole_fraction),1), atomic_size_difference, round(del_Chi, 4),del_Tm, Tm, VEC, AN, k, bulk,del_bulk,shear,del_shear, round(del_Smix, 4),round(lemda,4)])
    #prop_data = pd.DataFrame(properties, columns=['No of Components','Component','Moles of individual Components', 'Total Moles', 'Sum of individual MoleFractions', '$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K','B', 'ΔB','G', 'ΔG','ΔSmix','$\lambda$', 'ΔHmix','$\Omega$'])
    prop_data = pd.DataFrame(properties, columns=['No of Components','Component','Moles of individual Components', 'Total Moles', 'Sum of individual MoleFractions', '$\delta$', 'Δ$\chi$', 'ΔTm','Tm(K)', 'VEC', 'AN', 'K','B', 'ΔB','G', 'ΔG', 'ΔSmix','$\lambda$',])

    df = pd.concat([df, prop_data], axis = 1)
    
    df_input_target = df.iloc[:,[-18,-13,-12,-11,-10, -9, -8, -7, -6,-5, -4, -3, -2, -1,4,5,2,6,7]]
    
    return(df,df_input_target)


# In[ ]:





# In[ ]:


def input_target(datasets, input_name):
    inputs = datasets
    inputs = inputs.astype(float)

#     print(inputs.head(2))
#     print("..................................")
    #df_all = pd.DataFrame(inputs, columns = input_name+["e_ij_max"])
    #print(df_all.head())

    df_inputs = df_all.drop(['e_ij_max'], axis=1)
    df_targets = df_all['e_ij_max']
    return (df_all, df_inputs, df_targets)
    
def train_test_split(datasets, input_name):
    
    #df_all = datasets.astype(float)
    df_all = datasets
    df_inputs = df_all.drop(['e_ij_max','total'], axis=1)
    df_targets = df_all['total']
    


    # Split dataset in train-test
#     print("..................................")
#     print(df_all.head(2))
#     print("..................................")
#     print(df_inputs.head(2))
#     print("..................................")
#     print(df_targets.head())

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_inputs, df_targets, test_size=0.1, random_state=33)

    #X_train, X_test,  = train_test_split(df_inputs, test_size=0.1, random_state=0)

    #X_train_no_fab = X_train.drop(['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4','No of Components'], axis=1)
    #X_train_fab = X_train.loc[:,['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4']]

    #X_test_no_fab = X_test.drop(['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4','No of Components'], axis=1)
    #X_test_fab = X_test.loc[:,['Fab_1', 'Fab_2', 'Fab_3', 'Fab_4']]
    
    ##n_component = X_train.loc[:,['No of Components']]

    input_df = pd.concat([X_train, y_train], axis=1)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
        
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    return(X_train, X_test,y_train, y_test)


# In[ ]:


def train_test_distrubition(df,title="123"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker as tck

    df= df.to_frame()
    df = df.dropna().reset_index(drop=True)

    plt.rc('font', size=18)
    
    
    
    
    
    df["total"] = [np.array(m).astype('float64') for m in df["total"]]

    # create plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # create colormap
    norm = plt.Normalize(np.array(df["total"].values.tolist()).min(), np.array(df["total"].values.tolist()).max())
    cmap = plt.cm.plasma

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # plot surface
    for i, row in df.iterrows():
        x, y = np.meshgrid(range(6), range(3))
        z = row["total"]

        #ax.plot_surface(x+i*6, y, z, cmap=cmap, alpha=0.9,linewidth=0, rstride=1, cstride=1)
        ax.plot_surface(x, y+i*3, z, cmap=cmap, alpha=0.9,linewidth=0, rstride=1, cstride=1)

    # add colorbar
    clb = fig.colorbar(mappable, ax=ax,  shrink=0.45, pad = -0.05)
    clb.ax.tick_params(labelsize=16) 
    clb.set_label(r'$\alpha$', rotation=0)
    # Label and coordinate
    #ax.text(5, 10,10 , r"$\alpha$", color='red', fontsize=12)
    ax.text(4, max(ax.get_ybound()),1.1*max(ax.get_zbound()), r"$\alpha ^{  n}_{ij}$", color='red', fontsize=18)
    

    # set labels and title
    ax.set_xlabel('Columns', labelpad=12, fontsize=20,color='r')
    ax.set_ylabel('Rows x N', labelpad=20, fontsize=20,color='r')
    #ax.set_zlabel('Frequency', labelpad=8, fontsize=20,color='r')
    ax.set_title(title)

    ax.set_xticks(np.linspace(0, 6, 4))
    #ax.set_xticklabels(['1', '2', '6','2'])

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))

    
    ax.zaxis.set_major_locator(MaxNLocator(1))
    ax.get_zaxis().set_visible(False)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    #ax.set_zticklabels(ax.get_zticklabels(), fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # set camera angle and distance
    #ax.view_init(elev=20, azim=-40)
    ax.view_init(elev=20, azim=-40)
    #ax.dist=12

    # set background color and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    
    ax.set_zticks([]) # Remove the tick labels on the z-axis
    ax.set_zticklabels([])

    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) # Set the z-axis line color to transparent

    # show plot
    plt.show()


# In[ ]:


def new_split(df,title="123"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.cm import ScalarMappable

    df= df.to_frame()
    df = df.dropna().reset_index(drop=True)

    plt.rc('font', size=18)

    df["total"] = [np.array(m).astype('float64') for m in df["total"]]

    # create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # create colormap
    norm = plt.Normalize(np.array(df["total"].values.tolist()).min(), np.array(df["total"].values.tolist()).max())
    cmap = plt.cm.plasma

    # create ScalarMappable object based on frequency values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(df["total"])

    # plot surface with colors based on frequency values
    for i, row in df.iterrows():
        x, y = np.meshgrid(range(6), range(3))
        z = row["total"]

        ax.plot_surface(x, y+i*3, z, cmap=cmap, alpha=0.9, linewidth=0, rstride=1, cstride=1)

    # add colorbar
    fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.04)

    # set labels and title
    ax.set_xlabel('j x n', labelpad=12, fontsize=20,color='r')
    ax.set_ylabel('Rows', labelpad=20, fontsize=20,color='r')
    ax.set_zlabel('Frequency', labelpad=8, fontsize=20,color='r')
    ax.set_title(title)

    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # set camera angle and distance
    ax.view_init(elev=15, azim=-30)

    # set background color and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)

    # set plot size and scale
    fig.set_size_inches(10, 8)
    x_scale = 0.8
    y_scale = 1.2
    z_scale = 0.8
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0/scale.max())
    scale[3,3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj

    # show plot
    plt.show()


# In[ ]:


def n18_split(df,title="123"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.cm import ScalarMappable

    df= df.to_frame()
    df = df.dropna().reset_index(drop=True)

    plt.rc('font', size=18)

    df["total"] = [np.array(m).astype('float64') for m in df["total"]]

    # create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # create colormap
    norm = plt.Normalize(np.array(df["total"].values.tolist()).min(), np.array(df["total"].values.tolist()).max())
    cmap = plt.cm.plasma

    # create ScalarMappable object based on frequency values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(df["total"])

    # plot surface with colors based on frequency values
    for i, row in df.iterrows():
        x, y = np.meshgrid(range(18), range(2520))
        z = row["total"]

        ax.plot_surface(x, y, z, cmap=cmap, alpha=0.9, linewidth=0, rstride=1, cstride=1)

    # add colorbar
    fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.04)

    # set labels and title
    ax.set_xlabel('j x n', labelpad=12, fontsize=20,color='r')
    ax.set_ylabel('Rows', labelpad=20, fontsize=20,color='r')
    ax.set_zlabel('Frequency', labelpad=8, fontsize=20,color='r')
    ax.set_title(title)

    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # set camera angle and distance
    ax.view_init(elev=15, azim=-30)

    # set background color and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)

    # set plot size and scale
    fig.set_size_inches(10, 8)
    x_scale = 0.8
    y_scale = 1.2
    z_scale = 0.8
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0/scale.max())
    scale[3,3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj

    # show plot
    plt.show()


# In[ ]:


def data_distribution(data,text,test_annotate=0,test_annotate2=0,limit=1200,distance=0.15,ylabel = 'Hardness (HV)', title="Hardness Distribution",plot_path="plots\\hardness\\"):
    
    import seaborn as sns
    import matplotlib.ticker as tck
    
    mean=data.mean()
    median=np.median(data)
    ten_per = np.percentile(data, 10)
    ninety_per = np.percentile(data, 90)
    print(text,'\n Mean: ',mean,'\n Median: ',median, '\n 10 Percentile:',ten_per, '\n 90 Percentile:', ninety_per)
    print("Number of ",text, data.shape[0])
    print("----------------------")
    sns.set(style="ticks", color_codes=True,font_scale=1)
    
    fig , ax = plt.subplots(figsize=(4,4), dpi=400)
    sns.kdeplot( y=data, color="g",lw=0.5, shade=True, bw_adjust=1)
    
    # Plot Mean and Median
    plt.plot(distance,mean, marker="o", markersize=6, markeredgecolor="red", markerfacecolor="red", label="Mean",linestyle = 'None')
    plt.plot(distance,median, marker="^", markersize=6, markeredgecolor="blue", markerfacecolor="blue", label="Median",linestyle = 'None')
    
    # Plot 10 and 90 percentile
    plt.plot(distance,ten_per, marker="o", markersize=5, markeredgecolor="red", label="10% Percentile",linestyle = 'None')
    plt.plot(distance,ninety_per, marker="o", markersize=5, markeredgecolor="blue", label="90% Percentile",linestyle = 'None')

    x_values = [distance, distance]
    y_values = [ten_per, ninety_per]
    plt.plot(x_values, y_values, 'green', linestyle="-")
    
    # Annotations
    plt.text(distance*1.06, ten_per+test_annotate, str(round(ten_per,3))+ "(10%)", horizontalalignment='left', size='medium', color='b')
    plt.text(distance*1.06, ninety_per+test_annotate2, str(round(ninety_per,3))+ "(90%)", horizontalalignment='left', size='medium', color='b')

    #plt.gca().axes.get_xaxis().set_visible(False) # Remove x-axis lable
    
    plt.ylim(0, limit)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    
    plt.title(title+str(text)+" ("+str(data.shape[0])+" data)")
    plt.ylabel(ylabel)
    
    # Crop shaded region above max and below min value
    plt.axhspan(0,min(data), color='white')
    plt.axhspan(max(data),limit, color='white')

    plt.legend(frameon=False,loc='upper right')
    plt.savefig(plot_path+str(text)+ylabel+'_split.png',dpi=1200, bbox_inches='tight')
    #plt.legend(frameon=False);


# In[ ]:


def std_data(X_train):
    # Standarize the input features
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    
    scaler = MinMaxScaler()


    std_X_train = scaler.fit_transform(X_train)
    input_name = list(X_train.columns.values)

    std_df = pd.DataFrame(data=std_X_train, columns=input_name)
    #std_df['Hardness (HV)'] = y_train
    
    return(scaler, std_df)


# In[ ]:


def heatmap(std_train_df,name,prop='e_ij_max'):
    import seaborn as sns
    sns.set(style="ticks", color_codes=True,font_scale=2.2)
    plt.figure(figsize=(22,12))
    cmap = sns.diverging_palette(133,10,s=80, l=55, n=9, as_cmap=True)
    cor_train = std_train_df.corr()
    sns.heatmap(cor_train, annot=True, fmt='.2f',cmap=cmap) #

    plt.savefig(name+'_pcc_all.pdf',dpi=1200)
    plt.show()

def pcc_fs(std_df,y_train,input_pcc,name,prop='HV'):
    
    std_all_feature = np.column_stack((std_df,y_train))
    std_train_df=pd.DataFrame(data=std_all_feature, columns=input_name+[prop])
    heatmap(std_train_df,name)
    
    X_train_pcc = std_train_df.loc[:,input_pcc+[prop]]
    import seaborn as sns
    plt.figure(figsize=(13,6))
    cor_mid = X_train_pcc.corr()
    sns.heatmap(cor_mid, annot=True, cmap= plt.cm.CMRmap_r,fmt='.2f')
    plt.savefig(name+'_pcc_fs.png',dpi=1200,bbox_inches='tight')
    plt.show()
    


# In[ ]:


def vif_value(datasets):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(datasets.values,i) for i in range(datasets.shape[1])]

    vif['features'] = datasets.columns

    return(vif)


# In[ ]:


def pca_fs(std_df,name, title="a) Hardness PCA-1"):
    
    import seaborn as sns
    import matplotlib.ticker as tck
    
    # Plot PCA graph
    from sklearn.decomposition import PCA
    pca = PCA()
    sns.set(style="ticks", color_codes=True,font_scale=3)
    principalComponents = pca.fit_transform(std_df)
    plt.figure(figsize=(8,7))
    #plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_),color='purple', linewidth=3)
    plt.xlabel('No. of Principal Components')
    plt.ylabel('Cumulative EV')
    plt.title(title+ ' : Explained Variance ',color='blue', pad=20)
    
    plt.grid(alpha=0.75)
    plt.grid(True, which='minor')

    #x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #x=[2,4,6,8,10,12,14]
    plt.xlim(0, 150)
    #values = range(len(x))
    #plt.xticks(values, x)
    #plt.xticks()
    
    # Number of x ticks
    plt.xticks(range(0,152,50),rotation=0)
    plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    #plt.gca().xaxis.set_minor_locator(MultipleLocator(12))
    

    plt.rcParams.update({'font.size': 30})
    plt.tick_params(axis='both', which='both', length=3, width=1,color='black')
    
    import numpy
    plt.yticks(numpy.linspace(0.4, 1.0, num=4))
    
    plt.ylim(0.2, 1.05)
    #plt.yticks(range(0.3,0.9,0.3),rotation=0)
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    sns.set(style="ticks", color_codes=True,font_scale=2)

    plt.savefig(name+'_fs.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    # Principal components to capture 0.9 variance in data
    pca_1 = PCA(0.96)
    df_pca = pca_1.fit_transform(std_df)
    
    comp = pca_1.n_components_
    print('No. of components for PCA:' ,comp)
    
    pca_new = PCA(n_components = comp)
    df_pca_new = pca_new.fit_transform(std_df)
    
    print('Explained variance for 96% ', comp, 'components: ',pca_new.explained_variance_ratio_)
    print('Cumulative:', np.cumsum(pca_new.explained_variance_ratio_))
    
    return(pca_1,df_pca)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Machine Learning Model



# The value used in the function plays no role as the different hyperparameter value will be used while calling "create_model" function
def create_model(lyrs=6, neuron_size=64, act='selu', opt='Adam', dr=0.0, learning_rate=0.001,init_weights= 'he_uniform', weight_constraint = 3):
    import tensorflow as tf

    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Dropout
    from tensorflow.keras.constraints import max_norm

    import numpy as np
    import matplotlib.pyplot as plt
    
    # clear model
    tf.keras.backend.clear_session()

    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(neuron_size,input_dim=input_dim, activation=act))
    model.add(Dropout(dr))
    #tf.keras.layers.BatchNormalization(),
    
    # create additional hidden layers
    for i in range(1,lyrs):
        model.add(Dense(neuron_size, activation=act))
        model.add(Dropout(dr))
        
    model.add(Dense(neuron_size, activation='softmax'))    
        #tf.keras.layers.BatchNormalization(),
    # add dropout, default is none
    #model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(18, activation='sigmoid'))  # output layer
    opt = Adam(learning_rate=learning_rate)
    huber = tf.keras.losses.Huber(delta=1.5)
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mape','mae',tf.keras.metrics.RootMeanSquaredError()])
    
    return model


# In[ ]:


def matrics_plot(history,train_matrics,val_matrics,lable_name,model_name, data_of,plot_path):
    import matplotlib.ticker as tck
    all_train_mae_histories = []
    train_mae_history = train_matrics
    all_train_mae_histories.append(train_mae_history)
    average_train_mae_history = [
        np.mean([x[i] for x in all_train_mae_histories]) for i in range(max_epochs)]

    all_val_mae_histories = []
    val_mae_history = val_matrics
    all_val_mae_histories.append(val_mae_history)
    average_val_mae_history = [
        np.mean([x[i] for x in all_val_mae_histories]) for i in range(max_epochs)]
    
    loss = train_matrics
    val_loss = val_matrics
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', linewidth=2, label='Training '+lable_name)
    plt.plot(epochs, val_loss, '--r',  linewidth=1, label='Validation '+lable_name)
    plt.title('Training and Validation '+lable_name)
    plt.xlabel('Epochs')
    plt.ylabel(lable_name)
    plt.legend()
    #plt.savefig('mae_hardness.pdf',dpi=1200)
    plt.show()
    
    def smooth_curve(points, factor=0.9):
      smoothed_points = []
      for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
      return smoothed_points

    smooth_train_mae_history = smooth_curve(average_train_mae_history[5:])
    smooth_val_mae_history = smooth_curve(average_val_mae_history[5:])
    
    sns.set(style="ticks", color_codes=True,font_scale=2.25)
    fig, ax = plt.subplots(figsize=(6,5.5),dpi=600)
    #plt.ylim(20, 120)    # y-label range
    plt.gca().yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    plt.plot(range(1, len(smooth_train_mae_history) + 1), smooth_train_mae_history, 'b', label = 'Training '+lable_name)
    plt.plot(range(1, len(smooth_val_mae_history) + 1),smooth_val_mae_history, '--r', label = 'Validation '+lable_name)
    plt.xlabel('Epochs')
    plt.ylabel(''+lable_name)
    #plt.title('Smooth Training and Validation '+lable_name)
    plt.title(data_of+': '+model_name)
    plt.legend()
    plt.savefig(plot_path+data_of+'_'+lable_name+'.pdf',dpi=1200, bbox_inches='tight')
    plt.show()


# In[ ]:


#Predict on test data

def r2_plot(model,input_datasets,target_datasets,name,model_name,plot_path="plots\\hardness\\_"):
    sns.set(style="ticks", color_codes=True,font_scale=1.25)
    a=0.2 # Percentage error range
    predictions_datasets = model.predict(input_datasets)

    import sklearn.metrics
    from sklearn.metrics import r2_score
    r2_test = r2_score(target_datasets, predictions_datasets)
    plt.figure(figsize=(4,4),dpi=200)

    # plot x=y line 
    x_line = np.linspace(0, 50, 50)
    
    sns.lineplot(x=x_line, y=x_line,color='black',lw=0.75)

    print('Test R2 score: ', r2_test)
    


    test_r2 = sns.regplot(x=target_datasets,y=predictions_datasets,ci=None,scatter_kws=dict(s=8,color='r'),fit_reg=False)
    test_r2.set(title=str(model_name)+'Performance on Test data,'+' $R^2$ = ' +str(round(r2_test,3)))
    test_r2.set_xlabel("Real Targets"+"("+name+")", fontsize = 16)
    test_r2.set_ylabel("Predicted Value"+"("+name+")", fontsize = 16)

    Y1 = x_line*(1+a)
    Y2 = x_line*(1-a)

    sns.lineplot(x=x_line,y=Y1,lw=0.5,color='b',alpha=.2)
    sns.lineplot(x=x_line,y=Y2,lw=0.5,color='b',alpha=.2)

    test_r2.fill_between(x_line, Y1,x_line,color='b',alpha=.2)
    test_r2.fill_between(x_line, Y2,x_line,color='b',alpha=.2)
    
    # x and y ticks
    listOf_Yticks = np.arange(0, 40, 5)
    plt.yticks(listOf_Yticks)
    plt.xticks(listOf_Yticks)
    
    
    
    #ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))


    test_r2.figure.savefig('r2_hardness_'+str(name)+'.png',dpi=1200, bbox_inches='tight')


# In[ ]:





# In[ ]:


# The value used in the function plays no role as the different hyperparameter value will be used while calling "create_model" function
def create_model_class(opt='Adam', dr=0.0, learning_rate=0.001,init_weights= 'he_uniform', weight_constraint = 3):
    import tensorflow as tf

    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Dropout
    from tensorflow.keras.constraints import max_norm

    import numpy as np
    import matplotlib.pyplot as plt
    
    # clear model
    tf.keras.backend.clear_session()

    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(32,input_dim=input_dim, activation='selu', kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))
    #tf.keras.layers.BatchNormalization(),
    
    # create additional hidden layers
    model.add(Dense(24, activation = 'relu', kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))
        
    model.add(Dense(16, activation = 'LeakyReLU', kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))    
    
    model.add(Dense(12, activation = 'PReLU', kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))
    
    model.add(Dense(8, activation = 'relu', kernel_initializer = init_weights, kernel_constraint = max_norm(weight_constraint),kernel_regularizer='l2'))
    model.add(Dropout(dr))
    
    #tf.keras.layers.BatchNormalization(),
    # add dropout, default is none
    #model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(2, activation='softmax'))  # output layer

    model.compile(loss= 'categorical_crossentropy', optimizer=opt, metrics=['mse','accuracy'])
    
    return model


# In[ ]:


def ensemble_model(models, test_datasets,y_test):    
    import sklearn.metrics
    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score
    preds_array=[]
    #type(preds_df)
    for i in range(len(models)):
        preds = models[i].predict(test_datasets[i])
        accuracy_mean = accuracy_score(y_test,preds)
        preds_array.append(preds)

        print("R sq. for Model",i,accuracy_mean)
        #print(preds)
    preds_array=np.array(preds_array)    
    summed = np.sum(preds_array, axis=0)
    ensemble_prediction = np.argmax(summed, axis=1)
    mean_preds = np.mean(preds_array, axis=0)

    accuracy_mean = accuracy_score(y_test,mean_preds)
    print("Average accuracy:", accuracy_mean)

    # Weight calculations
    df = pd.DataFrame([])

    for w1 in range(0, 4):
        for w2 in range(0,4):
            for w3 in range(0,4):
                for w4 in range(0,4):
                    wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                    wted_preds1 = np.tensordot(preds_array, wts, axes=((0),(0)))
                    wted_ensemble_pred = np.mean(wted_preds1, axis=1)
                    weighted_r2 = accuracy_score(y_test, wted_ensemble_pred)
                    df = pd.concat([df,pd.DataFrame({'acc':weighted_r2,'wt1':wts[0],'wt2':wts[1], 
                                                 'wt3':wts[2],'wt4':wts[3] }, index=[0])], ignore_index=True)

    max_r2_row = df.iloc[df['acc'].idxmax()]
    print("Max $R^2$ of ", max_r2_row[0], " obained with w1=", max_r2_row[1]," w2=", max_r2_row[2], " w3=", max_r2_row[3], " and w4=", max_r2_row[4])  
    return(preds_array)


# In[ ]:


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
        for model in [model2, model3]:
            pred = model.predict(df_pca)  # Assuming the models have a predict() method
#             print("pred", pred)
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

    # from tensorflow import keras
    # model_cat = keras.models.load_model('model_files/nn_model/classification/model_cat.h5')
    # model_cata = keras.models.load_model('model_files/nn_model/classification/model_cata.h5')
    # model_catb = keras.models.load_model('model_files/nn_model/classification/model_catb.h5')

    from keras.models import load_model
    model_cat = load_model('model_files/nn_model/classification/model_cat.h5')
    model_cata = load_model('model_files/nn_model/classification/model_cata.h5')
    model_catb = load_model('model_files/nn_model/classification/model_catb.h5')

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

#     print(subcategories)
    if point!= '':
        # Replace all values in `subcategory` with "new"
        subcategories = np.where(subcategories != '', point, subcategories)

#     print("* * * ")
#     print(subcategories)
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


# In[ ]:


def two_dopants_ternary(base_composition = "(AlN)", element1="Mg", element2="Hg", cat='B', point = 'hextetramm', order=[2,2]):
    import pandas as pd
    import numpy as np

    # x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5])
    # y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4])

    x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0,0, 
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.2, 0.2, 0.2, 0.2, 0.2,
                        0.3, 0.4, 0.5, 0.6,
                        0.3, 0.3, 0.3,
                        0.4, 0.5,
                       0.4])
    y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                        0, 0, 0, 0, 0, 0, 0,0,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.3, 0.4, 0.5, 0.6,
                        0.2, 0.2, 0.2, 0.2,
                        0.3, 0.4, 0.5,
                        0.3, 0.3,
                       0.4])

    compositions = []
    for x, y in zip(x_array, y_array):
        composition = element1 + str(x) + element2 + str(y) + base_composition #"Al"+str(1-x-y)+"N"
        compositions.append(composition)

    df_composition = pd.DataFrame(compositions, columns=["formula_pretty"])
    
    ######################################################################################################
    
    # Do the required predictions
    cat, sub, tensor_eo = prediction_model(df_composition, cat= cat, point = point)
    ##########################################################################################################
    ##########################################################################################################
    
    # Ternary Plots
    import plotly.figure_factory as ff
    import numpy as np
    
#     Take the 3x3 of tensor

    target = []
    for itm in range(len(tensor_eo)):
#         trgt = tensor_eo[itm][order[0]][order[1]]
        trgt = np.sqrt(np.square(tensor_eo[itm][2][0]) + np.square(tensor_eo[itm][2][1]) +np.square(tensor_eo[itm][2][2]))
        target.append(trgt)

    target = np.array(target)

    # Calculate the remaining composition z
    comp = 1 - x_array - y_array
    pole_labels = [base_composition, element1, str(" .  ")+element2]
    colorscale = 'Rainbow' # Picnic Rainbow

    min_comp = np.min(comp)
    min_x = np.min(x_array)
    min_y = np.min(y_array)

    fig = ff.create_ternary_contour(np.array([comp, x_array, y_array]), target,
                                    pole_labels=pole_labels,
                                    interp_mode='cartesian',
                                    ncontours=40,
                                    colorscale=colorscale,
                                    showscale=True,
                                    width=1400, height=1050)

    fig.update_ternaries(baxis_nticks=5)
    fig.update_ternaries(aaxis_nticks=5)
    fig.update_ternaries(caxis_nticks=5)

    fig.update_layout(
        title_font_size=22,
#         xaxis_title="X Axis Title",
#         yaxis_title="Y Axis Title",
#         legend_title="Legend Title",
        font=dict(
            size=90,
            color="black",
            family="Gravitas One"
        ),
        margin=dict(l=200, r=200, t=200, b=220),  # Adjust the margins
        autosize=False,
        paper_bgcolor='white',  # Set the background color
        plot_bgcolor='white',  # Set the plot area color
    )

    fig.update_layout(
        ternary={
            'aaxis': {'ticklen': 18, 'tickwidth': 4},
            'baxis': {'ticklen': 18, 'tickwidth': 4},
            'caxis': {'ticklen': 18, 'tickwidth': 4}
        }
    )

    
    # Adjust color bar padding
    fig.update_coloraxes(colorbar=dict(
        tickmode='auto',
        thickness=50,
        dtick=1,


    ))

     #Increase linewidth of contour lines
    for contour_trace in fig['data']:
        if isinstance(contour_trace, go.Contour):
            contour_trace['line']['width'] = 10  # Increase linewidth for contour lines


    
    fig.update_ternaries(sum=1, aaxis_min=0.1, baxis_min=0.2, caxis_min=0)

    fig.show()


    # Locate the maximum value
    max_index = np.argmax(target)
    max_x = x_array[max_index]
    max_y = y_array[max_index]

    print(f"The maximum value is {round(max(target),2)} located at {element1}={max_x}, {element2}={max_y} and the index is {max_index}")
    
    return cat, sub, tensor_eo


# In[ ]:


def two_dopants(base_composition = "(AlN)", element1="Mg", element2="Hg", cat='B', point = 'hextetramm', order=[2,2]):
    import pandas as pd
    import numpy as np

    # x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5])
    # y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4])

    x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0,0, 
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.2, 0.2, 0.2, 0.2, 0.2,
                        0.3, 0.4, 0.5, 0.6,
                        0.3, 0.3, 0.3,
                        0.4, 0.5,
                       0.4])
    y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                        0, 0, 0, 0, 0, 0, 0,0,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.3, 0.4, 0.5, 0.6,
                        0.2, 0.2, 0.2, 0.2,
                        0.3, 0.4, 0.5,
                        0.3, 0.3,
                       0.4])

    compositions = []
    for x, y in zip(x_array, y_array):
        composition = element1 + str(x) + element2 + str(y) + base_composition #"Al"+str(1-x-y)+"N"
        compositions.append(composition)

    df_composition = pd.DataFrame(compositions, columns=["formula_pretty"])
    
    ######################################################################################################
    
    # Do the required predictions
    cat, sub, tensor_eo = prediction_model(df_composition, cat= cat, point = point)
    ##########################################################################################################
    ##########################################################################################################
    
    # Ternary Plots
    import plotly.figure_factory as ff
    import numpy as np
    
#     Take the 3x3 of tensor

    target = []
    target_33 = []
    target_31 = []

    for itm in range(len(tensor_eo)):
        trgt = np.sqrt(np.square(tensor_eo[itm][2][0]) + np.square(tensor_eo[itm][2][1]) +np.square(tensor_eo[itm][2][2]))
        target.append(trgt)
        trgt_33 = tensor_eo[itm][2][2]
        target_33.append(trgt_33)
        trgt_31 = tensor_eo[itm][2][0]
        target_31.append(trgt_31)        


    target = np.array(target)


    # Locate the maximum value
    max_index = np.argmax(target)
    max_x = x_array[max_index]
    max_y = y_array[max_index]
    
    max_index_33 = np.argmax(target_33)
    max_x_33 = x_array[max_index_33]
    max_y_33 = y_array[max_index_33]

    print(f"The max. value is {round(max(target),2)} and {round(target_33[max_index], 2)} {round(target_31[max_index], 2)}  at {element1}={max_x}, {element2}={max_y} and the index is {max_index}")
    print(f"The max. 3x3 value is {round(max(target_33),2)} and {round(target[max_index_33], 2)} {round(target_31[max_index_33], 2)}  at {element1}={max_x_33}, {element2}={max_y_33} and the index is {max_index_33}")

    return cat, sub, tensor_eo, target, target_33, target_31


# In[ ]:


def single_dopants_plots(base_composition = "(AlN)", element1="Mg", cat='B', point = 'hextetramm', order=[2,2]):
    import pandas as pd
    import numpy as np

    # x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5])
    # y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4])

    x_array = np.arange(0, 0.75, 0.025)
    y_array = np.arange(0, 0.75, 0.025)


    compositions = []
    for x, y in zip(x_array, y_array):
        composition = element1 + str(x) + base_composition #"Al"+str(1-x-y)+"N"
        compositions.append(composition)

    df_composition = pd.DataFrame(compositions, columns=["formula_pretty"])
    
    ######################################################################################################
    
    # Do the required predictions
    cat, sub, tensor_eo = prediction_model(df_composition, cat=cat, point = point)
    ##########################################################################################################
    ##########################################################################################################
    
#     target = []
#     for itm in range(len(tensor_eo)):
#         trgt = tensor_eo[itm][order[0]][order[1]]
#         target.append(trgt)
    
    target = []
    for itm in range(len(tensor_eo)):
        trgt = tensor_eo[itm][order[0]][order[1]]
        target.append(trgt)
        
    
    import matplotlib.pyplot as plt

    # Generate some sample data
    x = x_array
    y = target

    # Create a figure and axis objects
    fig, ax = plt.subplots()

    # Customize the line plot
    line_width = 2.5
    line_color = 'indigo'

    # Plot the data
    ax.plot(x, y, linewidth=line_width, color=line_color)

    # Customize the tick labels and font size
    tick_font_size = 18
    ax.tick_params(axis='both', labelsize=tick_font_size)

    # Customize other font sizes
    title_font_size = 20
    x_label_font_size = 18
    y_label_font_size = 18

    ax.set_title('Doping '+str(element1)+ " on "+str(base_composition), fontsize=title_font_size)
    ax.set_xlabel('Dopants composition (x)', fontsize=x_label_font_size)

    ax.set_ylabel("e' "+f"$_{order[0]}_{order[1]}$ "+r"$C/m^2$", fontsize=y_label_font_size)
    
    # Locate the maximum value
    max_index = np.argmax(target)
    max_x = x_array[max_index]
    max_y = y_array[max_index]
    
     # Add dashed vertical line from maximum value to y_max
    plt.axvline(x=max_x, ymin=0, ymax=max_y, color='gray',alpha=0.25, linestyle='--')
    plt.axhline(y=max_y, xmin=0, xmax=max_x, color='gray', alpha=0.25, linestyle='--')
    
    # Show the plot
    plt.show()


    print(f"The maximum value is {round(max(target),2)} located at {element1}={max_x} and the index is {max_index}")
    
    return cat, sub, tensor_eo


# In[ ]:


def single_dopants_new(base_composition = "(AlN)", element1="Mg", cat='B', point = 'hextetramm', order=[2,2]):
    import pandas as pd
    import numpy as np

    # x_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5])
    # y_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4])

    x_array = np.arange(0, 0.75, 0.025)
    y_array = np.arange(0, 0.75, 0.025)


    compositions = []
    for x, y in zip(x_array, y_array):
        composition = element1 + str(x) + base_composition # "Al"+str(1-x)+"N" #base_composition #
        compositions.append(composition)

    df_composition = pd.DataFrame(compositions, columns=["formula_pretty"])
    
    ######################################################################################################
    
    # Do the required predictions
    cat, sub, tensor_eo = prediction_model(df_composition, cat=cat, point = point)
    ##########################################################################################################
    ##########################################################################################################
    
#     target = []
#     for itm in range(len(tensor_eo)):
#         trgt = tensor_eo[itm][order[0]][order[1]]
#         target.append(trgt)
    
    target = []
    target_33 = []
    target_31 = []

    for itm in range(len(tensor_eo)):
        trgt = np.sqrt(np.square(tensor_eo[itm][2][0]) + np.square(tensor_eo[itm][2][1]) +np.square(tensor_eo[itm][2][2]))
        target.append(trgt)
        trgt_33 = tensor_eo[itm][2][2]
        target_33.append(trgt_33)
        trgt_31 = tensor_eo[itm][2][0]
        target_31.append(trgt_31)        

  
    # Locate the maximum value
    max_index = np.argmax(target)
    max_x = x_array[max_index]
    max_y = y_array[max_index]
    
    max_index_33 = np.argmax(target_33)
    max_x_33 = x_array[max_index_33]
    max_y_33 = y_array[max_index_33]

    print(f"The maximum value is {round(max(target),2)} and {round(target_33[max_index], 2)} {round(target_31[max_index], 2)} located at {element1}={max_x} and the index is {max_index}")
    print(f"The maximum 3x3 value is {round(max(target_33),2)} and {round(target[max_index_33], 2)} {round(target_31[max_index_33], 2)} located at {element1}={max_x_33} and the index is {max_index_33}")
    
    return tensor_eo, target, target_33, target_31


# In[ ]:



