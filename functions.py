#!/usr/bin/env python
# coding: utf-8
import math
import pandas as pd
import numpy as np
import streamlit as st

import pymatgen
import matminer
from pymatgen.core.composition import Composition
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
    
################################################################################################################


#####################################################################################################################
# Latex matrix
# Define the matrix size
def latex_36(my_tensor):
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

