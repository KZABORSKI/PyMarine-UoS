# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Runfile
"""
#%% IMPORT
import time
import pandas as pd
import dataprocessing as dp
import generateropax as genrpx

#%% Start time
start_time = time.time()


#%% Set global parameters
setfilters = True #Set argument for running the filter
UpdateReport = False #Update the report once the code has executed
MergeDatabase = False #Update the database file from the database

#%% Read and analyze the data
#Merge the database files
if MergeDatabase:
    dp.merge_database(path='Ship_Database_Raw_Archives/',
                      ShipType='-',
                      ShipTypeTags='-')

#Read dataframe - database
ShipData = dp.read_data(print_summary=True) 

#Aplly missing data fill filters
fill_filters = [['MCR',85],
                ['L_LaneTrailer',0],
                ['L_LaneTrailer(or)',0],
                ['L_LaneCar',0],
                ['L_LaneCar(or)',0],
                ['L_TrainTrack',0],
                ['N_Trailers',0],
                ['N_Trailers(or)',0],
                ['N_Cars',0],
                ['N_Cars(or)',0],
                ['L_Car',4.5],
                ['L_Trailer',12],
                ['L_TrainTrack',0],
                ['LNG Tanks',0]]
ShipData = dp.batch_fill_data(ShipData, fill_filters)

#Process the data
#ADD: Analyse the data for pass and vehicle area vs no decks
ShipData = dp.data_process(ShipData)

#Filter the data
if setfilters:
    ShipData = dp.set_filters(ShipData,
                              min_V_Service=0,
                              max_V_Service=40,
                              min_NPas=0,
                              max_NPas=10000,
                              min_year=1990,
                              max_year=2019,
                              min_LOA=70,
                              max_LOA=260,
                              min_B=0,
                              max_B=60,
                              min_T=0,
                              max_T=20,
                              min_D=1,
                              max_D=20,
                              min_LaneMt=0,
                              max_LaneMt=10000,
                              shaft=True,
                              azipod=True)

#Derive statistics relating to classification societies
dp.class_stats(ShipData,viewplot=False)

#Enter data for plotting and saving
plots = [['LBP','LOA','LOAvsLBP'],
         ['LOA','LBP','LBPvsLOA'],
         ['LBP','L/B','LBrvsLBP'],
         ['Lane_Mt','LBP','LBPvsLMt'],
         ['Lane_Mt','LB','LBvsLMt'],
         ['Lane_Mt','DWT','DWTvsLMt'],
         ['B','B/T','BTrvsB'],
         ['N_Pas','LM/Pas','LMtPasrNPas'],
         ['LBD','W_LS','WLSvsLBD'],
         ['LBD_Up','W_LS','WLSvsLBDUp'],
         ['N_Pas','DWT/Pas','DWTPasrvsNPas'],
         ['N_Pas','N_Berths','NBerthvsNPas'],
         ['LBP','B','BvsLBP'],
         ['LBP','L/B','LBrvsLBP'],
         ['LBP','V_Service','VvsLBP'],
         ['LBP','T','TvsLBP'],
         ['LBP','DWT', 'DWTvsLBP'],
         ['LBP','L/D','LDrvsLBP'],
         ['LB','L/D','LDrvsLB'],
         ['LBP','L/D_Up','LDUprvsLBP'],
         ['LB','L/D_Up','LDUprvsLB'],
         ['D','D_Upperdeck','DUpvsD'],
         ['DWT','BW Tanks','BWvsDWT'],
         ['DWT','GT','GTvsDWT'],
         ['LBD_Up','GT','GTvsLBDUp'],
         ['A_Projected','P_TB','PBowvsApr'],
         ['P_I_k','D_Prop','DpropvsPi'],
         ['P_I','Fuel consumption','FCvsPi'],
         ['T','D_Prop','DpropvsT'],
         ['T','D_Prop/T','DPropTrvsT'],
         ['P_I','P_Aux','PAuxvsPi'],
         ['N_Pas','P_Aux','PAuxvsNPas'],
         ['LB','N_Pas','NPasvsLB'],
         ['N_Pas','N_Crew_Total','CrewvsNPas'],
         ['P_I','HFO Tanks','HFOvsPi']]
        #Add Lbp vs L/disp ratio

dp.batch_plot_data(ShipData,
                   plots,
                   printdata=False,
                   viewplot=True,
                   save=True)

#ShipData = dp.data_stat_analysis(ShipData,
#                                 save=True)

#save extended database
dp.save_derived_data(ShipData)

#Output time of execution
print("Database analysis finished in {:.2f} seconds"\
      .format(time.time() - start_time))
print ("------------------------------------------------------------")
print('The analysis code has executed successfully')
print ("------------------------------------------------------------")

#%%Generate the case studies

#Case study 1:
#F.A. Gauthier
genrpx.generate_ropax(ShipData,
                      20,
                      10,
                      B=22,
                      A_Pas=3000,
                      N_Pas=755,
                      L_Trailer_Lane=1240,
                      L_Car_Lane=0,
                      allowable_error=0.2,
                      allowable_iterations=1500,
                      debug=True)

#Case study 2:
#Golfo Dei Coralli
genrpx.generate_ropax(ShipData,
                      22.5,
                      7,
                      LBratio=7.168,
                      A_Pas=1800,
                      N_Pas=308,
                      L_Trailer_Lane=2510,
                      L_Car_Lane=0,
                      allowable_error=0.2,
                      allowable_iterations=1500,
                      debug=True)

#Case study 3:
#Sprit of Britain
genrpx.generate_ropax(ShipData,
                      22,
                      7,
                      #LOA=213,
                      B=30.8,
                      #LBratio=4.5,
                      #Cb=0.652,
                      A_Pas=8960,
                      N_Pas=2000,
                      #P_I=30400,
                      L_Trailer_Lane=2700,
                      L_Car_Lane=1000,
                      allowable_error=0.2,
                      allowable_iterations=1500,
                      debug=True)

print("Finished in {:.2f} seconds"\
      .format(time.time() - start_time))
print ("------------------------------------------------------------")
print('Code has executed')
print ("------------------------------------------------------------")
