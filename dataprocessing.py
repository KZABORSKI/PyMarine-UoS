# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Data processing module
"""
#%% IMPORT
import glob
import warnings
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
sns.set(color_codes=True)
#%% IMPORT LOCAL MODULES
import ropax as rpx
import navarch as na
import powering as pw
import utilities as util

#%% Merge the database
def merge_database(path='Ship_Database_Raw_Archives/',
                   ShipType='-',
                   ShipTypeTags=[]):
    """
    --------------------------------------------------------------------------
    Merge the datafiles from database directory tree
    --------------------------------------------------------------------------
    Input:
    path - str, specify the path for the unprocessed database
    ShipType - str, specifying the type of the ship
    ShipTypeTags - list of strings, specifying extra descriptive tag(s)
    --------------------------------------------------------------------------
    Output:
    0 - int, succesfull execution
    --------------------------------------------------------------------------
    NOTE: After resetting the index, the column 'Vessel Name' becomes the index 
    in dataframe, hence index must be enabled when saving to avoid data loss
    --------------------------------------------------------------------------
    """
    #Read the template file containing column names
    data = pd.read_csv('Manual\Ship_Template_Folder\data.csv',
                       dtype=None,
                       delimiter=',',
                       usecols=[0,1])
    #Browse through raw database path, find .csv files
    csv_files = [f for f in glob.glob(path + "**/*.csv", recursive=True)]
    for f in csv_files:
        #Extract tabulated data
        single_data = pd.read_csv('{}'.format(f),
                               dtype=None,
                               delimiter=',',
                               usecols=[2]) #That's where pure data is
        #Append single data to main data
        data = pd.concat([data, single_data.reindex(data.index)],
                          axis=1)
    #Reset the index to avoid bugs transponding
    data.reset_index()
    data = data.transpose()
    #Save datafile
    data.to_csv('Temp\data_temp.csv',
                index=True,
                index_label='Vessel Name') #Note refers to this bit
    #Remove the first row (residual indices) from the file
    util.csv_remove_first_row('Temp\data_temp.csv',
                               'Data_Input\data.csv',
                               delete=False)
    return 0

#%% Read the database
def read_data(filepath='Data_Input\data.csv',
              print_summary=True):
    """
    --------------------------------------------------------------------------
    Read the database from the file
    --------------------------------------------------------------------------
    Input:
    filepath  -string, path of the database .csv file
    print_stats - bool, specify whether to print statistical data (number of
    ships in the database, number of missing values)
    --------------------------------------------------------------------------
    Output:
    data - pandas.df, database
    --------------------------------------------------------------------------
    """
    #Define missing value types:
    missing_values = ["n/a","na", "-"]
    #Define boolean value types:
    bool_true_values = ['Y', 'y']
    bool_false_values = ['N', 'n']
    #Read Ship Database from .csv file:
    data = pd.read_csv('{}'.format(filepath),
                       dtype=None,
                       delimiter= ',',
                       skiprows = [1],
                                  na_values = missing_values,
                                  true_values = bool_true_values,
                                  false_values = bool_false_values)
    if print_summary:
        n_ships = data.shape[0]
        print('Number of ships in the loaded database:')
        print('N_Ship: %.0f' %(n_ships))
        print("------------------------------------------------------------")
        print_data_summary(data)
        
    return data

#%% Print dataset summary
def print_data_summary(data, extended=False):
    """
    --------------------------------------------------------------------------
    Print summary of the database, range of parameters and number of missing
    (NaN) values.
    --------------------------------------------------------------------------
    Input:
    data - pandas.df, database
    extended - return extended summary, default=False
    --------------------------------------------------------------------------
    Output:
    data - pandas.df, updated database
    --------------------------------------------------------------------------
    """
    L_min = data['LBP'].min()
    L_max = data['LBP'].max()
    B_min = data['B'].min()
    B_max = data['B'].max()
    T_min = data['T'].min()
    T_max = data['T'].max()
    D_min = data['D'].min()
    D_max = data['D'].max()
    V_min = data['V_Service'].min()
    V_max = data['V_Service'].max()
    P_min = data['P_I'].min()
    P_max = data['P_I'].max()
    Disp_min = data['Displacement'].min()
    Disp_max = data['Displacement'].max()
    DWT_min = data['DWT'].min()
    DWT_max = data['DWT'].max()
    NP_min = data['N_Pas'].min()
    NP_max = data['N_Pas'].max()
    print('Database summary:')
    print("------------------------------------------------------------")
    print('{}m < LBP < {}m'.format(L_min,L_max))
    print("------------------------------------------------------------")
    print('{}m < B < {}m'.format(B_min,B_max))
    print("------------------------------------------------------------")
    print('{}m < T < {}m'.format(T_min,T_max))
    print("------------------------------------------------------------")
    print('{}m < D < {}m'.format(D_min,D_max))
    print("------------------------------------------------------------")
    print('{}knts < V < {}knts'.format(V_min,V_max))
    print("------------------------------------------------------------")
    print('{}kW < P_I < {}kW'.format(P_min,P_max))
    print("------------------------------------------------------------")
    print('{}t < Displacement < {}t'.format(Disp_min,Disp_max))
    print("------------------------------------------------------------")
    print('{}t < DWT < {}t'.format(DWT_min,DWT_max))
    print("------------------------------------------------------------")
    print('{} < No Passengers < {}'.format(NP_min,NP_max))
    print("------------------------------------------------------------")
    if extended:
        LBr_min = data['L/B'].min()
        LBr_max = data['L/B'].max()
        BTr_min = data['B/T'].min()
        BTr_max = data['B/T'].max()
        TDr_min = data['T/D'].min()
        TDr_max = data['T/D'].max()
        Fn_min = data['Fn'].min()
        Fn_max = data['Fn'].max()
        LMt_min = data['Lane_Mt'].min()
        LMt_max = data['Lane_Mt'].max()
        print('{:.2f} < L/B < {:.2f}'.format(LBr_min,LBr_max))
        print("------------------------------------------------------------")
        print('{:.2f} < B/T < {:.2f}'.format(BTr_min,BTr_max))
        print("------------------------------------------------------------")
        print('{:.2f} < T/D < {:.2f}'.format(TDr_min,TDr_max))
        print("------------------------------------------------------------")
        print('{:.2f} < Fn < {:.2f}'.format(Fn_min,Fn_max))
        print("------------------------------------------------------------")
        print('{:.2f}m < Lane Meters < {:.2f}m'.format(LMt_min,LMt_max))
        print("------------------------------------------------------------")
    print("Missing data statistics:")
    print(data.isnull().sum())
    print("------------------------------------------------------------")
    #Print tags appearing in the database
    tags = []
    print("Tags:")
    print(tags)
    print("------------------------------------------------------------")
    return 0

#%% Set filters
def set_filters(data,
                tags=['tag'],
                notags=['tag'],
                min_V_Service=0,
                max_V_Service=100,
                min_NPas=0,
                max_NPas=10000,
                min_year=1990,
                max_year=2019,
                min_LOA=0,
                max_LOA=300,
                min_B=0,
                max_B=100,
                min_T=0,
                max_T=100,
                min_D=0,
                max_D=100,
                min_LaneMt=0,
                max_LaneMt=7000,
                shaft=True,
                azipod=True,
                print_summary=True):
    """
    --------------------------------------------------------------------------
    Filter database
    --------------------------------------------------------------------------
    Input:
    data - pandas.df, database
    tags - list, list of strings describing required tags
    notags - list, list of strings describing excluding tags
    --------------------------------------------------------------------------
    Output:
    ShipData - pandas.df, filtered database
    --------------------------------------------------------------------------
    """
    #Get number of ships at the input
    n_ships = data.shape[0]
    #Filter speed
    data = data[data['V_Service'] >= min_V_Service]
    data = data[data['V_Service'] <= max_V_Service]
    #Filter number of passengers
    data = data[data['N_Pas'] >= min_NPas]
    data = data[data['N_Pas'] <= max_NPas]
    #Filter year built
    data = data[data['Year'] >= min_year]
    data = data[data['Year'] <= max_year]
    #Filter length over all
    data = data[data['LOA'] >= min_LOA]
    data = data[data['LOA'] <= max_LOA]
    #Filter beam
    data = data[data['B'] >= min_B]
    data = data[data['B'] <= max_B]
    #Filter draft
    data = data[data['T'] >= min_T]
    data = data[data['T'] <= max_T]
    #Filter depth
    #NaNs are not excluded
    data = data[(data['D'] >= min_D) | (data['D'].isnull())]
    data = data[(data['D'] <= max_D) | (data['D'].isnull())]
    #Filter lane meters
    data = data[data['Lane_Mt'] >= min_LaneMt]
    data = data[data['Lane_Mt'] <= max_LaneMt]
    #Filter propulsion type
    if not shaft and not azipod:
        warnings.warn("At least one propulsion type must be selected")
    if not shaft or not azipod:
        if shaft:
            data = data[data['Azi?'] != shaft]
        else:
            data = data[data['Azi?'] == azipod]
    if print_summary:
        #Print number of ships after filtering and the summary
        n_ships_filtered = data.shape[0]
        print('Filtering')
        print("------------------------------------------------------------")
        print('Number of ships removed: %.0f' %(n_ships-n_ships_filtered))
        print("------------------------------------------------------------")
        print('Number of ships after filtering: %.0f' %(n_ships_filtered))
        print("------------------------------------------------------------")
        print_data_summary(data,extended=True)
    return data

#%% Fill missing data
def fill_missing_data(ShipData,
                      column_name,
                      value):
    """
    --------------------------------------------------------------------------
    Fill missing data with chosen value
    --------------------------------------------------------------------------
    Input:
    ShipData - pandas.df, database
    column_name - string, name of the data to fill NaN's
    value - value to replace the NaN entries
    --------------------------------------------------------------------------
    Output:
    ShipData - pandas.df, updated database
    --------------------------------------------------------------------------
    """
    ShipData['{}'.format(column_name)].fillna(value, inplace=True)
    return ShipData

#%% Batch fill missing data
def batch_fill_data(ShipData,
                    fill_sets):
    """
    --------------------------------------------------------------------------
    Batch fill missing data in the dataframe
    --------------------------------------------------------------------------
    Input:
    ShipData - pandas.df, database
    fill_sets - list, list of lists containing strings (column names) and values
    replacing NaN's e.g:
    fill_sets = [['L_Car',4.5],['L_Trailer',12]]
    fill_sets[0] = column_name, fill_sets[1] = value
    --------------------------------------------------------------------------
    Output:
    See fill_missing_data
    ShipData - pandas.df, updated database
    --------------------------------------------------------------------------
    """
    #Loop through plot_data function
    for group in fill_sets:
        fill_missing_data(ShipData,
                          group[0],
                               group[1])
    return ShipData

#%% Bar graph of ships by classification society
def class_stats(ShipData,
                save=True,
                viewplot=False):
    """
    --------------------------------------------------------------------------
    Sort the vessels according to class society
    --------------------------------------------------------------------------
    Input:
    ShipData - pandas.df, database
    viewplot - bool, view plot?
    --------------------------------------------------------------------------
    Output:
    Data_Derived\ClassFreq.csv - histogram
    --------------------------------------------------------------------------
    """   
    if save:
        Class_Hist = ShipData.groupby('Classification')\
                                     ['Vessel Name'].nunique()
        Class_Hist.to_csv('Data_Derived\ClassFreq.csv',
                          header=True,
                          index=True)
    if viewplot:
        ShipData.groupby('Classification')\
                        ['Vessel Name'].nunique().plot(kind='bar')
        plt.show()
    return 0

#%% Manipulate the data:
def data_process(dt):
    """
    --------------------------------------------------------------------------
    Create ratios and other derived data:
    L/B, LWL/LOA, LB, B/T, T/D, B/D, L/D, L/D_Upperdeck, Fn, endurance, range,
    projected transeverse area, total bowthruster power, number of crew
    (non-officers)
    --------------------------------------------------------------------------
    Input:
    dt - pandas.df, ship database
    --------------------------------------------------------------------------
    Output:
    dt - pandas.df,, updated ship database
    --------------------------------------------------------------------------
    """
    dt['L/B'] = dt['LBP']/dt['B']
    dt['LBP/LOA'] = dt['LBP']/dt['LOA']
    dt['LB'] = dt['LBP']*dt['B']
    dt['B/T'] = dt['B']/dt['T']
    dt['T/D'] = dt['T']/dt['D']
    dt['B/D'] = dt['B']/dt['D']
    dt['L/D'] = dt['LBP']/dt['D']
    dt['L/D_Up'] = dt['LBP']/dt['D_Upperdeck']
    dt['LBD'] = dt['LBP']*dt['B']*dt['D']
    dt['LBD_Up'] = dt['LBP']*dt['B']*dt['D_Upperdeck']
    dt['D_Prop/T'] = dt['D_Prop']/dt['T']
    dt['P_I_k'] = dt['P_I']/1000
    dt['Fn'] = pw.Froude_Number(dt['V_Service'],dt['LBP'],input_knots=True)
    dt['LDisprat'] = na.LDispl_ratio(dt['LBP'],dt['Displacement'])
    dt['W_LS'] = dt['Displacement']-dt['DWT']
    dt['Endurance'] = dt['HFO Tanks']/dt['Fuel consumption']
    dt['Range'] = dt['Endurance']*24*dt['V_Service']
    dt['A_Projected'] = dt['LBP']*1.2*(dt['D_Upperdeck']-dt['T'])
    dt['P_TB'] = dt['No. Bowtrusters']*dt['P_Bowthruster']
    dt['N_Crew'] = dt['N_Crew_Total']-dt['N_Officers']
    dt['L_Lane_max_T'], dt['L_Lane_add_C'], dt['L_Lane_max_C']\
    = zip(*dt.apply(lambda x: rpx.get_L_Lanes(x['L_LaneTrailer'],
    x['L_LaneTrailer(or)'],x['L_LaneCar'],x['L_LaneCar(or)'],x['N_Trailers'],
    x['N_Trailers(or)'],x['N_Cars'],x['N_Cars(or)'],x['L_Trailer'],
    x['L_Car']),1))
    #Total meters (arrangement of hoistable decks is not taken into account
    #in this scenario)
    dt['Lane_Mt'] = dt.apply(lambda x: rpx.Lane_Mt(x['L_Lane_max_T'],
    x['L_Lane_add_C'],x['L_Lane_max_C'],x['L_TrainTrack']),1)
    dt['LM/Pas'] = dt['Lane_Mt']/dt['N_Pas']
    dt['DWT/Pas'] = dt['DWT']/dt['N_Pas']
    dt['A_Pas'] = dt['LB']*dt['A_PasRelative']*(0.7) #assumed deduction
    dt['A_Veh'], dt['A_Trailer'], dt['A_Car'], dt['A_Car_max'], dt['A_Train']\
    = zip(*dt.apply(lambda x: rpx.A_Veh(x['L_Lane_max_T'],x['L_Lane_add_C'],
    x['L_Lane_max_C'],x['L_TrainTrack'],),1))
    return dt

#%% Plot data
def plot_data(ShipData,
              x_args,
              y_args,
              savename='dataplot',
              line_best_fit=True,
              viewplot=False,
              save=True,
              robust=True):
    """
    --------------------------------------------------------------------------
    Plot selected corelations (and save them in .csv)
    --------------------------------------------------------------------------
    Input:
    ShipData - pandas.df, database
    x_args - string, column name of x data
    y_args - string, column name of y data 
    viewplot - bool, plot or not?
    save - bool, save the data in .csv files or not
    robust - bool, robust regression model, default=True
    --------------------------------------------------------------------------
    Output:
    Plot of x_args vs y_args
    Data_Derived\{savename}.csv - saved plot data
    --------------------------------------------------------------------------
    """
    if line_best_fit:
        #Creater line of best fit data
        pass
    if viewplot:
        #Get coeffs of linear fit
        plt.figure() #Used to avoid figures stacking on to each other
        sns.regplot(x=x_args,
                    y=y_args,
                    data=ShipData,
                    color='b',
                    fit_reg=True,
                    robust=robust)
    if save: #Copy and save
        data = ShipData[[x_args, y_args]].copy()
        data.to_csv('Data_Derived\Graphs\{}.csv'.format(savename),
                      header=False,
                      index=False)
    return 0

#%% Plot selected correlations (and save them in .csv)
def batch_plot_data(data,
                    plots,
                    viewplot=False,
                    printdata=False,
                    save=True):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    ShipData - pandas.df, database
    plots - list, list of lists containing strings e.g:
    plots = [['LOA','B','LOAvsB'],['L/B','V_Service','LoverBvsV']]
    plots[0] - x_args, plots[1] - y_args, plots[2] - filename
    viewplot - bool, plot or not?
    save - bool, save the data in .csv files or not
    --------------------------------------------------------------------------
    Output:
    See plot_data
    --------------------------------------------------------------------------
    """
    #Loop through plot_data function
    for plot in plots:
        plot_data(data, 
                  plot[0],
                  plot[1],
                  savename='{}'.format(plot[2]),
                  viewplot=viewplot,
                  save=save,
                  robust=True)
        if printdata:
            linear_best_fit(data,plot[0],plot[1],printdata=printdata)
    return 0

#%% Create linear line of best fit and get coefficients
def linear_best_fit(data,
                    x_args,
                    y_args,
                    fillNaN=True,
                    robust=True,
                    printdata=False,
                    plot=False):
    
    """
    --------------------------------------------------------------------------
    Create linear line of best fit and get coefficients
    --------------------------------------------------------------------------
    Input:
    --------------------------------------------------------------------------
    Output:
    intercept - float, intercept of the linear equation (y=slope*x+intercept)
    slope - float, slope of the linear equation (y=slope*x+intercept)
    --------------------------------------------------------------------------
    WARNIGN:
    Input data cannot be negative - see first part of the code
    --------------------------------------------------------------------------
    """
    divider = '------------------------------------------------------------'
    #Filter data, get rid of NaN's
    data = data[(data[x_args] >= -1)]
    data = data[(data[y_args] >= -1)]
    #Set bounds
    x_min = data[x_args].min()
    x_max = data[x_args].max()
    #Use add_constants to get intercept
    x2_args = sm.add_constant(data[x_args])
    if robust:
        model = sm.RLM(data[y_args], x2_args, M=sm.robust.norms.LeastSquares())
    else:
        model = sm.OLS(data[y_args], x2_args)
    #Straight line equation coefficients
    parameters = model.fit().params
    intercept = parameters[0]
    slope = parameters[1]
    if printdata:
        #Get bounds of y-values
        y_min = data[y_args].min()
        y_max = data[y_args].max()
        print('Data for {} vs {}:'.format(y_args,x_args))
        print(divider)
        print('Range of x:{} - {}, y:{} - {}'.format(x_min,x_max,y_min,y_max))
        print(divider)
        if robust:
            #Calculate OLS as well in order to get R^2 value
            model2 = sm.OLS(data[y_args], x2_args)
            parameters2 = model2.fit().params
            intercept2 = parameters2[0]
            slope2 = parameters2[1]
            #Calculate R^2
            r2 = model2.fit().rsquared
            print('OLS: Slope: {}, Intercept: {}'.format(slope2, intercept2))
            print(divider)
            print('R^2={:.3f}'.format(r2))
            print(divider)
            print('RLM: Slope: {}, Intercept: {}'.format(slope, intercept))
            print(divider)
        else:
            print('OLS: Slope: {}, Intercept: {}'.format(slope, intercept))
            print(divider)
            print('R^2=')
            print(divider)
        print('Extreme points: ({},{:.2f})({},{:.2f})'.format(x_min,
              (slope*x_min+intercept),x_max,(slope*x_max+intercept)))
        print(divider)
    if plot:
        ax = data.plot(x=x_args, y=y_args, kind='scatter')
        #Plot regression line on the same axes, set values
        x = [x_min, x_max]
        ax.plot(x, [intercept+x_min*slope,intercept+x_max*slope])
        ax.set_xlim([x_min, x_max])
    return intercept, slope

#%% Get interpolated value based on linear regression
def get_value_reg_lin(data,
                        x_value,
                        x_data,
                        y_data,
                        robust=False):
    """
    --------------------------------------------------------------------------
    Get interpolated value based on linear regression
    --------------------------------------------------------------------------
    Input:
    --------------------------------------------------------------------------
    Output:
    y - float, predicted y value
    --------------------------------------------------------------------------
    """
    m,a = linear_best_fit(data,
                          x_data,
                          y_data,
                          robust=robust)
    y = a*x_value+m
    return y

#%% Create and plot statistical data
def data_stat_analysis(data, save=True):
    """
    --------------------------------------------------------------------------
    Create and plot statistical data
    --------------------------------------------------------------------------
    Input:
    data - pandas.df, database
    --------------------------------------------------------------------------
    Output:
    --------------------------------------------------------------------------
    """
    # Number of vehicle decks

    # Number of propellers

    # CPP vs fixed pitch propellers

    # Azipod vs shaft propulsion

    # Bulbous bow?

    #Bar graph of ships by Froude Number
    data[['Fn']].plot(kind='hist',
            bins=[0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42],
                 rwidth=0.8)
    plt.show()
    if save: #Copy and save
        Fn = data['Fn'].copy()
        Fn.to_csv('Data_Derived\Fn.csv',
                        header=False,
                        index=False)

    #Bar graph of number of engines
    data[['N_MainEngines']].plot(kind='hist',
            bins=[2,3,4],
                 rwidth=1)
    plt.show()
    if save: #Copy and save
        NME = data['N_MainEngines'].copy()
        NME.to_csv(r'Data_Derived\NME.csv',
                        header=False,
                        index=False)
    return 0

#%% Save the data
def save_derived_data(data):
    """
    --------------------------------------------------------------------------
    Save the extended data in a .csv file
    --------------------------------------------------------------------------
    Input:
    data - pandas.df, database
    --------------------------------------------------------------------------
    Output:
    data saved as Data_Derived\data_extended.csv
    --------------------------------------------------------------------------
    """
    data.to_csv('Data_Derived\data_extended.csv', index=False)
    print('Data has been saved')
    print ("------------------------------------------------------------")
    return 0
