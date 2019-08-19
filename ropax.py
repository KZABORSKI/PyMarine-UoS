# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Ropax calculation module
"""
#%% IMPORT
import math
import numpy as np
import pandas as pd
import warnings
import dataprocessing as dpro

#%% Calculate lane meters
def L_Lane(N_Trailers=0,
           L_Trailer=12,
           N_Cars=0,
           L_Car=4.5):
    """
    --------------------------------------------------------------------------
    Calculate lane meters
    --------------------------------------------------------------------------
    Input:
    N_Trailers - int, number of trailers [-]
    L_Trailer - float, typical trailer length [m], default=12
    N_Cars - int, number of cars [-]
    L_Car - float, typical car length [m], default=4.5
    --------------------------------------------------------------------------
    Output:
    LTL - int, length of trailer lane [m]
    LCL - int, length of car lane [m]
    --------------------------------------------------------------------------
    """
    LTL = math.ceil(N_Trailers*L_Trailer) #round up
    LCL = math.ceil(N_Cars*L_Car) #round up
    return LTL, LCL

#%% Calculate lane meters
def Lane_Mt(LTL,
           LCL,
           LCOL,
           LTT):
    """
    --------------------------------------------------------------------------
    Calculate lane meters
    --------------------------------------------------------------------------
    Input:
    LTL - int, length of trailer lane [m]
    LCL - int, length of car lanes [m]
    LCOL - int, length of car lanes (cars only) [m]
    LTT - int, train track length [m]
    --------------------------------------------------------------------------
    Output:
    LMt - int, normalised lane meters [m]
    --------------------------------------------------------------------------
    """
    if (LTL+LCL) > LCOL:
      LMt = LTL+LCL+LTT
    else:
      LMt = LCOL+LTT

    return LMt

#%% Calculate lane meters
def get_L_Lanes(LT1,LT2,LC1,LC2,NT1,NT2,NC1,NC2,L_Trailer,L_Car):
    """
    --------------------------------------------------------------------------
    Clean the data - calculate combination achieving maximum lane meters for
    each vehicle type and total normalised lane meters
    --------------------------------------------------------------------------
    Input:
    LT1 - int, length of trailer lane, 1st load case [m]
    LT2 - int, length of trailer lane, 2nd load case [m]
    LC1 - int, length of car lane, 1st load case [m]
    LC2 - int, length of car lane, 2nd load case [m]
    NT1 - int, number of trailers, 1st load case [-]
    NT2 - int, number of trailers, 2nd load case [-]
    NC1 - int, number of cars, 1st load case [-]
    NC2 - int, number of cars, 2nd load case [-]
    L_Trailer - float, trailer length [m]
    L_Car - float, car length [m]
    --------------------------------------------------------------------------
    Output:
    LTL - int, length of trailer lane [m]
    LCL - int, length of car lanes [m]
    LCOL - int, length of car lanes (cars only) [m]
    --------------------------------------------------------------------------
    NOTE: Lane lengths and number of vehicles input data do not have to be
    related to each other.
    --------------------------------------------------------------------------
    WARNING: The method may over-estimate the lane lengths. It can happen in a
    case when missing vehicle length data is filled with assumed length that
    may be larger than actual.
    --------------------------------------------------------------------------
    """
    #Calculate the case when maximum number of trailers is achieved
    #Check which combination of lane lengths favors trailer loading
    if LT1 > LT2:
      LT_temp = LT1
      LC_temp = LC1
    else:
      LT_temp = LT2
      LC_temp = LC2
    #Check which combination of vehicle numbers favors trailer loading
    if NT1*L_Trailer > NT2*L_Trailer:
      LT_temp2 = NT1*L_Trailer
      LC_temp2 = NC1*L_Car
    else:
      LT_temp2 = NT2*L_Trailer
      LC_temp2 = NC2*L_Car
    #Check which overall combination is larger
    if LT_temp > LT_temp2:
      LTL = LT_temp
      LCL = LC_temp
    else:
      LTL = LT_temp2
      LCL = LC_temp2

    #Calculate case when cars only are loaded
    #Check which combination of lane lengths favors car loading
    if LC1 > LC2:
      LCOL_temp = LC1
    else:
      LCOL_temp = LC2
    #Check which combination of lane lengths favors car loading
    if NC1*L_Car > NC2*L_Car:
      LCOL_temp2 = NC1*L_Car
    else:
      LCOL_temp2 = NC2*L_Car
    #Check which overall combination is larger
    if LCOL_temp > LCOL_temp2:
      LCOL = LCOL_temp
    else:
      LCOL = LCOL_temp2
      
    #Round the values
    LTL = math.ceil(LTL)
    LCL = math.ceil(LCL)
    LCOL = math.ceil(LCOL)

    return LTL, LCL, LCOL

#%% Calculate vehicle area
def A_Veh(L_Trailer_Lane=0,
          L_Car_Lane=0,
          L_Car_max_Lane=0,
          L_Train_Track=0,
          width_TL=2.9,
          width_CL=2.1,
          width_TrainL = 3.35,
          total=False):
    """
    --------------------------------------------------------------------------
    Calculate vehicle area
    --------------------------------------------------------------------------
    Input:
    L_Trailer_Lane - float, trailer lane length [m]
    width_TL - float, trailer lane width [m], default=2.9
    L_Car_Lane - float, car lane length [m]
    L_Car_max_Lane - float, car lane length (max cars load case) [m]
    width_CL - float, car lane width [m], default=2.1
    L_Train_Track - float, train track length [m]
    width_TrainL - train lane width [m], default=3.35
    --------------------------------------------------------------------------
    Output:
    Trailer_Lane_Meters - int, length of trailer lanes [m]
    Car_Lane_Meters - int, length of car lanes [m]
    --------------------------------------------------------------------------
    """
    A_Trailer = L_Trailer_Lane*width_TL
    A_Car = L_Car_Lane*width_CL
    A_Car_max = L_Car_max_Lane*width_CL
    A_Train = L_Train_Track*width_TrainL
    if A_Car_max > (A_Car + A_Trailer):
        A_Veh = A_Trailer+A_Car_max+A_Train
    else:
        A_Veh = A_Trailer+A_Car+A_Train
    if total:
        return A_Veh
    else:
        return A_Veh, A_Trailer, A_Car, A_Car_max, A_Train

#%% Calculate cabin areas
def A_Cabins(filepath='Data_Input\cabins.csv'):
    """
    --------------------------------------------------------------------------
    Calculate cabin areas
    --------------------------------------------------------------------------
    Input:
    N_Trailers - int, number of trailers [-]
    L_Trailer - float, typical trailer length [m], default=12
    N_Cars - int, number of cars [-]
    L_Car - float, typical car length [m], default=4.5
    --------------------------------------------------------------------------
    Output:
    Trailer_Lane_Meters - int, length of trailer lanes [m]
    Car_Lane_Meters - int, length of car lanes [m]
    --------------------------------------------------------------------------
    """
    #Read cabin data
    dt = dpro.read_data(filepath=filepath,
                        print_stats=False)
    #Manipulate the data
    dt['N_Berths'] = dt['N_Cabins']*dt['Berths']
    dt['A_Cabins'] = dt['N_Cabins']*dt['Area']
    A_Pas_Cabins = dt['A_Cabins'].sum()
    return A_Pas_Cabins

#%% Calculate seating area
def A_Seating(N_Pas,
              Pas_Area_pp=2):
    """
    --------------------------------------------------------------------------
    Calculate seating area
    --------------------------------------------------------------------------
    Input:
    N_Pas - int, number of seated passengers [-]
    Pas_Area - float, area per seat [m^2], default=2
    --------------------------------------------------------------------------
    Output:
    A_Seat - float, passenger seating area [m^2]
    --------------------------------------------------------------------------
    """
    A_Seat = N_Pas*Pas_Area_pp
    return A_Seat

#%% Calculate passenger area
def A_Pas(A_Seat=0,
          A_Pas_Cabins=0,
          A_Pas_Other=0):
    """
    --------------------------------------------------------------------------
    Calculate passenger area
    --------------------------------------------------------------------------
    Input:
    N_Pas - int, number of seated passengers [-]
    Pas_Area - float, area per seat [m^2], default=2
    --------------------------------------------------------------------------
    Output:
    A_Seat - float, passenger seating area [m^2]
    --------------------------------------------------------------------------
    """
    return 0

#%% Calculate range and endurance
def endurance(P_Engine,
              V_HFO_Tank,
              N_engines=1,
              SFC=0.2,
              HFO_SG=0.93,
              Permeability=0.96):
    """
    --------------------------------------------------------------------------
    Calculate range and endurance
    --------------------------------------------------------------------------
    Input:
    P_Engine - int, engine power [kW]
    V_HFO_Tank - float, volume of HFO tanks [m^3]
    N_Engines - int, number of engines [-]
    SFC - float, specific fuel consumption (kg kW^-1 hr^-1], default=0.2
    HFO_SG - float, specific gravity of heavy fuel oil [], default=0.93
    Permeability - float, permeability of fuel tanks [-], default=0.96
    --------------------------------------------------------------------------
    Output:
    endr - float, endurance (days)
    --------------------------------------------------------------------------
    """
    #Convert tank volume to mass and apply deduction
    W_HFO_Tank = V_HFO_Tank*HFO_SG*Permeability
    #Daily fuel consumption [t]
    W_HFO_Daily = SFC/(1000*1000)*P_Engine*24
    endr = W_HFO_Tank/W_HFO_Daily
    return endr

#%% Custom defined weight from .csv file
def W_custom(input_csv_file):
    """
    --------------------------------------------------------------------------
    Description
    --------------------------------------------------------------------------
    Input:
    --------------------------------------------------------------------------
    Output:
    --------------------------------------------------------------------------
    """
    # Define missing value types:
    missing_values = ["n/a","na", "-"]
    # Read weight data from .csv file:
    W = pd.read_csv('Data_Input\custom_weights.csv',
                       dtype=None,
                       delimiter= ',',
                       skiprows = [1],
                                  na_values = missing_values)
    #Sum the weight column
    W['W_Tot_Row'] = W['N_Items']*W['Weight']
    return W_custom

#%% Calculate weight and volume of heavy fuel oil
def fluid_HFO(endurance,
           P_I,
           SG=0.93,
           ded_struc=3):
    """
    --------------------------------------------------------------------------
    Calculate weight and volume of heavy fuel oil (HFO)
    --------------------------------------------------------------------------
    Input:
    endurance - int, endurance [days]
    P_I - int, installed power [kW]
    SG - float, specific gravity of heavy fuel oil [-]
    ded_struc - float, volume deduction accounting for structure [%],
    default=3
    --------------------------------------------------------------------------
    Output:
    W_HFO - float, weight of heavy fuel oil [t]
    V_HFO - float, volume of heavy fuel oil [t]
    --------------------------------------------------------------------------
    NOTE:
    A typical consumption figure is 0.2 kg/kW/hr for large diesels. Thus a
    large tanker with continuous power of 30,000 kW would use about 150 tonnes
    OF/day. [Ship Design and Economics, page 70]
    --------------------------------------------------------------------------
    """
    W_HFO = 24*(0.2/1000)*P_I*endurance
    V_HFO = W_HFO/SG
    return W_HFO, V_HFO

#%% Calculate weight and volume of diesel oil
def fluid_DO(endurance,
           P_Aux,
           SG=0.85,
           ded_struc=3):
    """
    --------------------------------------------------------------------------
    Calculate weight and volume of diesel oil (DO)
    --------------------------------------------------------------------------
    Input:
    endurance - int, endurance [days]
    P_Aux - int, auxilary power power [kW]
    SG - float, specific gravity of diesel oil [-]
    ded_struc - float, volume deduction accounting for structure [%],
    default=3
    --------------------------------------------------------------------------
    Output:
    W_DO - float, weight of diesel oil [t]
    V_DO - float, volume of diesel oil [t]
    --------------------------------------------------------------------------
    NOTE:

    --------------------------------------------------------------------------
    """
    W_DO = 24*(0.2/1000)*P_Aux*endurance
    V_DO = W_DO/SG
    return W_DO, V_DO

#%% Calculate weight and volume of lubricating oil
def fluid_LO(endurance,
           P_I,
           SG=0.9,
           ded_struc=3):
    """
    --------------------------------------------------------------------------
    Calculate weight and volume of lubricating oil (LO)
    --------------------------------------------------------------------------
    Input:
    endurance - int, endurance [days]
    P_I - int, installed power
    SG - float, specific gravity of lubricating oil [-]
    ded_struc - float, volume deduction accounting for structure [%],
    default=3
    --------------------------------------------------------------------------
    Output:
    W_FW - float, weight of lubricating oil [t]
    V_FW - float, volume of lubricating oil [t]
    --------------------------------------------------------------------------
    NOTE:
    Assumtions made based on Practical Ship Design, P 508
    "Lubricating oil usage for the main engine crankcase and cylinders plus
    that used in generators and other machinery can be approximated to 35
    litres per day per 1000 KW of main engine power."
    --------------------------------------------------------------------------
    """
    W_LO = endurance*(35/1000)*(math.ceil(P_I/1000))
    V_LO = W_LO/SG
    return W_LO, V_LO

#%% Calculate weight and volume of fresh water
def fluid_FW(endurance,
          N_Crew,
          N_Pas,
          P_Fw=50,
          SG=1,
          ded_struc=3):
    """
    --------------------------------------------------------------------------
    Calculate weight and volume of fresh water (FW)
    --------------------------------------------------------------------------
    Input:
    endurance - int, endurance [days]
    P_Fw - int, fresh water per crew per day [kg]
    SG - float, specific gravity of fresh water [-]
    ded_struc - float, volume deduction accounting for structure [%],
    default=3
    --------------------------------------------------------------------------
    Output:
    W_FW - float, weight of fresh water [t]
    V_FW - float, volume of fresh water [t]
    --------------------------------------------------------------------------
    """
    W_FW = ((N_Crew+N_Pas)*endurance*P_Fw) / 1000
    V_FW = W_FW/SG
    return W_FW, V_FW

#%% Calculate weight and volume of fluids
def fluids(endurance,
           P_I,
           N_Crew,
           N_Pas,
           ded_struc=3,
           save_summary=True):
    """
    --------------------------------------------------------------------------
    Calculate weight and volume of fluids
    --------------------------------------------------------------------------
    Input:
    endurance - int, endurance [days]
    ded_struc - float, volume deduction accounting for structure [%],
    default=3
    --------------------------------------------------------------------------
    Output:
    W_Fluids - float, weight of fluids [t]
    --------------------------------------------------------------------------
    """
    #Fresh Water [t]
    #Waste water
    #Lubricating Oil
    #HFO
    #DO
    #Ballast water [t]
    #Anti-roll tanks
    W_Fluids = W_HFO+W_DO+W_LO+W_WW+W_FW+W_BW
    if save_summary:
        fluid_summary = pd.DataFrame()
        #Save the summary in .csv
        #fluid_summary
        pass
    return W_Fluids

#%% Calculate optimum block coefficient:
def Cb_opt(LWL,V):
    """
    --------------------------------------------------------------------------
    Add reference
    --------------------------------------------------------------------------
    Input:
    LWL - float, length of waterline [m]
    V - float, cruising speed [knots]
    --------------------------------------------------------------------------
    Output:
    CB - float, block coefficient [-]
    --------------------------------------------------------------------------
    """
    Cb=1.23-0.395*(V/math.sqrt(LWL))
    return Cb

#%% Calculate the number of crew and officers, and weight:
def W_complement(N_Crew,
                 W_Person=0.085,
                 W_Luggage=0.08,
                 belongings=True):
    """
    --------------------------------------------------------------------------
    Description
    --------------------------------------------------------------------------
    Input:
    --------------------------------------------------------------------------
    Output:
    --------------------------------------------------------------------------
    """
    if belongings:
        W_Complement = N_Crew*(W_Person+W_Luggage)
    else:
        W_Complement = N_Crew*W_Person
    return W_Complement

#%% Calculate the number of crew and officers, and weight:
def W_passengers(N_Pas,
                 W_Person=0.085,
                 W_Luggage=0.05):
    """
    --------------------------------------------------------------------------
    Description
    --------------------------------------------------------------------------
    Input:
    --------------------------------------------------------------------------
    Output:
    --------------------------------------------------------------------------
    """
    W_Passengers = N_Pas*(W_Person+W_Luggage)
    return W_Passengers

#%% Calculate weight of provisions:
def W_provision(endurance,
                N_Persons,
                W_Provision_Day=10):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    endurance - float, endurance [days]
    N_Persons - int, number of people onboard (passengers+complement) [-]
    W_Provision_Day - float, provision per crew per day [kg]
    --------------------------------------------------------------------------
    Output:
    W_Provision - float, weight of provisions [t]
    --------------------------------------------------------------------------
    """
    W_Provisions = endurance*N_Persons*(W_Provision_Day/1000)
    return W_Provisions

#%% Calculate the deadweight
def deadweight(endurance,
               N_Crew,
               N_Pas,
               L_Trailer_Lane,
               L_Car_Lane,
               W_Fluids,
               W_Trailer=15,
               L_Trailer=12,
               W_Car=2,
               L_Car=4.5,
               W_Luggage=0.05,
               extra_weight=0,
               margin=0):
    """
    --------------------------------------------------------------------------
    Calculate the deadweight
    --------------------------------------------------------------------------
    CHECK DEFAULT WEIGHT OF CARS AND TRAILERS
    Input:
    N_Complement - int, number of crew and officers [-]
    N_Pas - int, number of passengers [-]
    L_Trailer_Lane - int, length of trailer lane [m]
    L_Car_Lane - int, length of car lane [m]
    W_Provisions - float, weight of provisions [t]
    W_Fluids - float, weight of all liquids in the tanks [t]
    W_Trailer - float, average trailer weight [t], default=15
    W_Car - float, average car weight [t], default=3
        Check: https://cars.lovetoknow.com/List_of_Car_Weights
    W_Luggage - float, average luggage weight per passenger [t]
    margin - float, margin [%], default=0
    --------------------------------------------------------------------------
    Output:
    DWT - float, deadweight [t]
    --------------------------------------------------------------------------
    """
    W_Crew = W_complement(N_Crew)
    W_Pas = W_passengers(N_Pas)
    W_Trailers = L_Trailer_Lane*(W_Trailer/L_Trailer)
    W_Cars = L_Car_Lane*(W_Car/L_Car)
    W_Provisions = W_provision(endurance,(N_Crew+N_Pas))
    DWT = np.sum([W_Crew,W_Pas,W_Trailers,W_Cars,W_Provisions,W_Fluids])
    DWT = DWT*(1+(margin/100)) #Apply percentage margin
    return DWT
