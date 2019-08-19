# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Ropax generation algorithm module
"""
#%% IMPORT
import math
import warnings
import numpy as np
import pandas as pd
#%% IMPORT LOCAL MODULES
import navarch as na
import powering as pwr
import ropax as rpx
import technoeconomics as te
import dataprocessing as dp

#%% Run algorithm debugging
def debug_summary(data,magnify=False,save=True):
    """
    --------------------------------------------------------------------------
    Vizualize how the parameters change over iterations
    --------------------------------------------------------------------------
    Input:
    data - pandas.df, parameters over number of iterations
    --------------------------------------------------------------------------
    Output:
    -
    --------------------------------------------------------------------------
    Note:
    If detailed debug mode is selected, only last 100 iterations are displayed
    --------------------------------------------------------------------------
    """
    divider = '------------------------------------------------------------'

    #Select only last 100 iterations if indicated
    if magnify:
    	#Create lower bound number of iterations
    	min_it = data['Iteration'].max()-100
    	#Filter values only above lower bound
    	data = data[data['Iteration'] >= min_it]

    #Plot parameters versus number of iterations
    data.plot(x='Iteration',y=['LOA','LBP','LWL'],
              color=['red','green','yellow'])
    data.plot(x='Iteration',y=['B','T','D','D_Up'],
              color=['red','green','yellow','blue'])
    data.plot(x='Iteration',y=['L/B','B/T','T/D','T/D_Up'],
              color=['red','green','yellow','blue'])
    data.plot(x='Iteration',y=['Cb','Cm','Cp','Cw'],
              color=['red','green','yellow','blue'])
    data.plot(x='Iteration',y=['W_Displ_HS','W_Displ'],
              color=['red','green'])
    data.plot(x='Iteration',y=['W_LS','DWT'],
              color=['red','green','yellow','blue'])
    data.plot(x='Iteration',y='W_err',color='red')
    
    if save:
    	data.to_csv('Data_Derived\debug_data.csv', index=False)
    	print('Debug data has been saved')
    	print(divider)
    return 0

#%% Main Code: ropax generator, define input parameters
def generate_ropax(database,
                   V,
                   endurance,
                   LBP=0,
                   LOA=0,
                   LWL=0,
                   B=0,
                   LBratio=0,
                   T=0,
                   BTratio=0,
                   D=0,
                   D_Up=0,
                   Cb=0,
                   Cw=0,
                   Cm=0,
                   R_bilge=0,
                   lcb=100,
                   restrict_LOA=1e3,
                   restrict_B=1e3,
                   restrict_T=1e3,
                   P_I=0,
                   W_LS=0,
                   W_Steel=0,
                   DWT=0,
                   A_Pas=0,
                   N_Pas=0,
                   A_Seat=0,
                   N_Pas_Seat=0,
                   A_Pas_Other=0,
                   N_Crew=8,
                   N_Trailers=0,
                   L_Trailer=12,
                   L_Trailer_Lane=0,
                   width_TL=2.9,
                   N_Cars=0,
                   L_Car=4.5,
                   L_Car_Lane=0,
                   width_CL=2.1,
                   N_Full_Veh_Decks=2,
                   extra_DWT=0,
                   allowable_error=0.5,
                   allowable_iterations=500,
                   cabins=False,
                   te_analysis=False,
                   debug=False):
    """
    --------------------------------------------------------------------------
    Main algorithm generating a ropax ship, given requirements
    --------------------------------------------------------------------------
    Input:
    database - pandas.df, processed ship database
    V - float, service speed [knots]
    LBP - float, length between perpendiculars [m]
    LOA - float, length over all [t]
    LWL - float, length of waterline [t]
    B - float, beam [m]
    T - float, draft [m]
    Cb - float, block coefficient [-]

    restrict_LOA - float, maximum length [m]
    restrict_B - float, maximum beam [m]
    restrict_T - float, maximum draft [m]

    If *_Lane_Meters is provided, N_Trailers and N_Cars is disregarded
    N_Pas - int, number of passengers [-]
    N_Cabins - int, number of cabins [-]
    A_Cabin - float, average passenger cabin area [m^2]

    N_Trailers - int, number of trailers [-]
    L_Trailer - float, typical trailer length [m], default=12
    L_Trailer_Lane - int, length of trailer lanes [m], default=0
    width_TL - float, trailer lane width [m], default=2.9

    N_Cars - int, number of cars [-]
    L_Car - float, typical car length [m], default=4.5
    Car_Lane_Meters - int, length of car lanes [m], default=0
    width_CL - float, car lane width [m], default=2.1

    endurance - float, endurance (days)

    allowable_error - float, allowable error [%], default=0.5
    allowable_iterations - int, maximum number of iterations [-], default=500

    te_analysis - bool, run techno-economic analysis? default=True

    --------------------------------------------------------------------------
    Output:
    LCB - float, ???

    --------------------------------------------------------------------------
    Variables:
    W_Displ_HS - float, displacement calculated from Cb and main particulars
    W_Displ - float, displacement calculated from weights and deadweight
    --------------------------------------------------------------------------
    """
    divider = '------------------------------------------------------------'
    print('Starting ship generating algorithm')
    print(divider)

    #Preset data for debugging purposes
    #-------------------------------------------------------------------------
    if debug:
        #Initiate an empty debug dataframe
        ddat = pd.DataFrame()
        #Temporarly preset some values in order to avoid undefined values in
        #debugging (itdt) dataframe.
        GM = 0
        T_roll = 0
        T_roll_IMO = 0
        Fn = 0
        P_E = 0
        LBD_Up = 0
        W_LS = 0
        DWT = 0
        W_Fluids = 0
        W_Displ = 0
        displ_error = 100000
    #-------------------------------------------------------------------------

    #Set ship specific 'common sense' bounds
    #-------------------------------------------------------------------------
    Cb_min = 0.40
    Cb_max = 0.70
    #-------------------------------------------------------------------------

    #Set default (False) values of dimension controls
    #-------------------------------------------------------------------------
    LBP_Set = False
    LOA_Set = False
    LWL_Set = False
    B_Set = False
    LBratio_Set = False
    T_Set = False
    BTratio_Set = False
    D_Set = False
    D_Up_Set = False
    Cb_Set = False
    Cw_Set = False
    Cm_Set = False
    R_bilge_Set = False
    lcb_Set = False
    P_I_Set = False
    W_LS_Set = False
    DWT_Set = False
    #-------------------------------------------------------------------------

    #Pre-process input particulars
    #-------------------------------------------------------------------------
    if LBP !=0:
        LBP_Set = True
        #Check if LBP is within bounds of the database
        if LBP > database['LBP'].max() or LBP < database['LBP'].min():
            warnings.warn("LBP outside bounds, extrapolation may\
                          yield innacurate results")
    else:
        #Pre-set LBP (mid of LBP range in the database)
        LBP = (database['LBP'].min())\
              +((database['LBP'].max())-(database['LBP'].min()))/2
    if LOA !=0:
        LOA_Set = True 
    if LWL !=0:
        LWL_Set = True
        #Check if LBP is set (must be if LWL is defined)
        if not LBP_Set:
            warnings.warn('LBP must be defined alongside LWL')
    if B !=0:
        B_Set = True
    if LBratio !=0:
        LBratio_Set = True
    if not B_Set and not LBratio_Set and not (LBP_Set or LOA_Set):
        #Pre-set B (mid of B range in the database)
        B = (database['B'].min())\
              +((database['B'].max())-(database['B'].min()))/2
    if T !=0:
        T_Set = True
    if BTratio != 0:
        BTratio_Set = True
    if D !=0:
        D_Set = True
    if D_Up !=0:
        D_Up_Set = True
    if Cb !=0:
        Cb_Set = True
        #Check Cb bounds
        if Cb > Cb_max or Cb < Cb_min:
            warnings.warn('Input Cb outside bounds ({}<Cb<{})'\
                          .format(Cb_min,Cb_max))
    if Cw !=0:
        Cw_Set = True
    if Cm !=0:
        Cm_Set = True
    if R_bilge !=0:
        R_bilge_Set = True
    if Cm_Set and R_bilge_Set:
    	warnings.warn('Midship section overdefined')
    if lcb !=100:
        lcb_Set = True
    if P_I !=0:
    	P_I_Set = True
    if W_LS !=0:
    	W_LS_Set = True
    if DWT !=0:
    	DWT_Set = True
    #-------------------------------------------------------------------------

    #Handle vehicle data
    #-------------------------------------------------------------------------
    #Check vehicle data for input consistency
    if L_Trailer_Lane != 0 and N_Trailers != 0:
        warnings.warn("Input for both trailer lane length and number of\
                      trailers is not allowed")
    if L_Car_Lane != 0 and N_Cars != 0:
        warnings.warn("Input for both car lane length and number of\
                      cars is not allowed")

    #Calculate lane lengths unless provided in the input
    if L_Trailer_Lane == 0 and L_Car_Lane == 0:
        #Calculate lane lengths
        L_Trailer_Lane, L_Car_Lane = rpx.L_Lane(N_Trailers=0,
                                                L_Trailer=12,
                                                N_Cars=0,
                                                L_Car=4.5)
    
    #Calculate number of vehicles unless provided in the input
    else:
        pass
    #-------------------------------------------------------------------------
    
    #Calculate areas
    #-------------------------------------------------------------------------
    #Calculate required vehicle areas
    A_Veh, A_Trailer, A_Car, A_Car_max, A_Train\
    = rpx.A_Veh(L_Trailer_Lane=L_Trailer_Lane,
                width_TL=width_TL,
                L_Car_Lane=L_Car_Lane,
                width_CL=width_CL)
    #Calculate crew area
    A_Crew=25*N_Crew+60
    #Calculate passenger area if not prescribed
    if A_Pas == 0:
        #Check passenger data for input consistency
        if A_Seat != 0 and N_Pas_Seat != 0:
            warnings.warn("Input for both passenger area and number of\
                          passengers is not allowed")
        if A_Seat == 0 and N_Pas_Seat == 0:
            A_Seat = 0
        if N_Pas_Seat != 0:
            #Calculate seating area
            A_Seat = rpx.A_Seating(N_Pas_Seat,Pas_Area_pp=2)
        if cabins:
            #Read cabins from external file
            A_Pas_Cabins = rpx.A_Cabins()
        else:
            A_Pas_Cabins = 0
        #Multiply to account for layout obstructions/loses
        A_Pas = (A_Seat+A_Pas_Cabins+A_Pas_Other)*1.3
    A_Total = A_Veh+A_Pas+A_Crew
    #Add case when all of the areas are input from a .csv file
    print(A_Veh)
    #-------------------------------------------------------------------------

    #Import prescribed volumes
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    
    #Calculate volumes
    #-------------------------------------------------------------------------
    Vol_Veh = A_Veh*4.8
    Vol_Pas = A_Pas*2.8
    Vol_Crew = A_Crew*2.8
    Vol_Total = Vol_Veh+Vol_Pas+Vol_Crew
    #-------------------------------------------------------------------------
    
    #Initialize increment array
    #-------------------------------------------------------------------------
    #Change in: [LBP,Cb,LBratio,BTratio,D] respectively
    incr = np.zeros(5)
    #-------------------------------------------------------------------------

    #Initiate the loop, set counter of iterations
    #-------------------------------------------------------------------------
    it_counter = 0
    loop = True
    goal = False
    #Start design loop
    while loop:
        #Gather particulars into dataframe
        #---------------------------------------------------------------------
        if (debug and it_counter > 0) or goal:
            #Insert iteration data into dataframe
            itdt = pd.DataFrame({'Iteration':[it_counter],'LOA':[LOA],
            'LBP':[LBP],'LWL':[LWL],'B':[B],'T':[T],'D':[D],'D_Up':[D_Up],
            'Cw':[Cw],'Cm':[Cm],'Cb':[Cb],'Cp':[Cp],'LCB':[lcb],
            'L/B':[LBratio],'B/T':[BTratio],'L/D':[LDratio],
            'LD_Up':[LDupratio],'LB':[LB],'T/D':[TDratio],
            'T/D_Up':[TDupratio],'Veh decks':[N_d_Veh],'Pas decks':[N_d_Pas],
            'Crew decks':[N_d_Crew],'KG':[KG],'KB':[KB],'BM':[BM],'GM':[GM],
            'T_roll':[T_roll],
            #'V calc':[V_Calculated],
            'W_Fluids':[W_Fluids],
            'W_Displ_HS':[W_Displ_HS],'V_Displ_HS':[V_Displ_HS],
            'W_Displ':[W_Displ],'W_LS':[W_LS],'DWT':[DWT],'Failed':[failed],
            'W_err':[displ_error],'LBD_Up':[LBD_Up],
            'L/Disp_ratio':[LDisp_ratio],'P_E':[P_E],'P_I':[P_I],'Fn':[Fn],
            'dL':[incr[0]],'dCb':[incr[1]],'T_roll_IMO':[T_roll_IMO],
            'dL/B':[incr[2]],'dB/T':[incr[3]]})
            if goal:
                particulars = itdt
                break
            else:
                #Attach iteration dataframe to debug dataframe
                ddat = pd.concat([ddat, itdt])
        #---------------------------------------------------------------------

    	#Check for number of iterations
        #---------------------------------------------------------------------
        if it_counter > allowable_iterations:
            #raise RuntimeError('Computation failed: unable to create ship\
            #within the given number of iterations') from error
            print ('Ship generation failed')
            print (divider)
            if debug:
                #Run debug
                debug_summary(ddat)
            return 1
        #---------------------------------------------------------------------

        #Calculate particulars, ratios and coefficients
        #---------------------------------------------------------------------
        #Length between perpendiculars
        if not LBP_Set:
            if LOA_Set:
                LBP = dp.get_value_reg_lin(database,LOA,'LOA','LBP',
                                           robust=True)
            else:
                if incr[0] != 0:
                    #Adjust LBP
                    LBP += incr[0]
                else:
                    pass

        #Length over all
        if not LOA_Set:
            #Calculate LOA based on database data
            LOA = dp.get_value_reg_lin(database,LBP,'LBP','LOA',robust=True)
        #Check if LOA within restrictions
        if LOA > restrict_LOA:
            #Shorten LOA to maximum allowed, calculate LBP from data
            LOA = restrict_LOA
            LBP = dp.get_value_reg_lin(database,restrict_LOA,
                                       'LOA',
                                       'LBP',
                                       robust=True)
            #Adjust B/T to account for lack of increase in volume due to L
            if incr[0] != 0:
            		#Adjust 
            		BTratio += 0.1
        
        #Length of waterline
        if not LWL_Set:
            #Calculate LWL
            LWL = LBP*1.041
        
        #Beam
        if LBratio_Set:
            #Calculate B
            B = LBP/LBratio
        else:
            #Calculate L/B
            if not B_Set and not (LBP_Set or LOA_Set):
                if incr[2] != 0:
                	#Increment L/B ratio
                	LBratio += incr[2]
                else:
                	#Calculate L/B ratio from data
                	LBratio = dp.get_value_reg_lin(database,LBP,'LBP','L/B',
                                               robust=True)
            else:
                LBratio = LBP/B
            if not B_Set:
                #Calculate B
                B = LBratio/LBP
        #Check if B within restrictions
        if B > restrict_B:
            #Shorten B to maximum allowed, calculate L/B
            B = restrict_B
            LBratio = LBP/B

        #Calculate LB (LOA*B)
        LB = LOA*B
        
        #Draft
        if BTratio_Set:
            #Calculate T
            T = B/BTratio
        else:
            #Calculate B/T
            if not T_Set:
                if incr[3] != 0:
            		#Increment B/T ratio
                	BTratio += incr[3]
                #Calculate B/T based on database data
                BTratio = dp.get_value_reg_lin(database,B,'B','B/T',
                                               robust=True)
                T = B/BTratio
            else:
                BTratio = B/T
        if T > restrict_T:
            #Shorten T to maximum allowed, calculate B/T
            T = restrict_T
            BTratio = B/T
		
        if not D_Set:
            if incr[4] != 0:
                #Increment L/B ratio
                LBratio += incr[2]
            else:
            	#Calculate D based on database data
            	D = dp.get_value_reg_lin(database,LBP,'LBP','D',robust=True)
        #Calculate L/D ratio
        LDratio = LBP/D
        #Calculate T/D ratio
        TDratio = T/D

        if not D_Up_Set:
            #Calculate D upperdeck based on database data
            D_Up = dp.get_value_reg_lin(database,D,'D','D_Upperdeck')
        #Calculate L/D_Up ratio
        LDupratio = LBP/D_Up
        #Calculate T/D_Up ratio
        TDupratio = T/D_Up

        if not Cb_Set:
            if incr[1] != 0:
                #Adjust Cb
                Cb += incr[1]
            else:
                #Calculate optimum Cb
                Cb = rpx.Cb_opt(LWL,V)
            #Check if within bounds
            if Cb > Cb_max:
                Cb = Cb_max
            if Cb < Cb_min:
                Cb = Cb_min

        if not Cw_Set:
        	#Calculate Cw
        	#Cw = Cb+0.1 #Munro-Smith
        	#Reference: H. O. Kristensen Analysis of technical data of Ro-Ro
        	#ships
        	Cw = 0.7*Cb+0.38

        if not Cm_Set:
        	#Calculate Cm
        	if R_bilge_Set:
        		Cm = 1-((R_bilge**2)*(2-(math.pi/2)))/(B*T)
        	else:
        		#Reference: H. O. Kristensen Analysis of technical data of 
        		#Ro-Ro ships
        		Cm = 0.38-1.25*(Cb**2)+1.725*Cb

        #Calculate Cp
        Cp = Cb/Cm
        #---------------------------------------------------------------------

        #Calculate principle dimension-based displacement
        #---------------------------------------------------------------------
        W_Displ_HS = na.W_displ_princ(LBP,B,T,Cb,margin=0)
        V_Displ_HS = W_Displ_HS/1.025
        #---------------------------------------------------------------------

        #Calculate power/displacement ratio
        #---------------------------------------------------------------------
        LDisp_ratio = na.LDispl_ratio(LBP,(W_Displ_HS/1.025))
        #---------------------------------------------------------------------

        #Check areas (no decks per area group)
        #---------------------------------------------------------------------
        N_d_Veh = LB/(A_Veh/0.7)
        N_d_Pas = LB/(A_Pas/0.7)
        N_d_Crew = LB/(A_Crew/0.7)
        if N_d_Veh > N_Full_Veh_Decks:
            it_counter += 1
            #Reset other constraints
            incr = np.zeros(5)
            #Adjust length
            incr[4] = 1
            #Indicate the type of iteration failure
            failed = 'Volume (Decks)'
            continue
        #---------------------------------------------------------------------

        #Check volume
        #---------------------------------------------------------------------

        #---------------------------------------------------------------------
        """
        To be corrected
        V_Calculated = LBP*B*D_Up
        #Reference: ship design and economics
        Cbprime = Cb+(1-Cb)*(0.8*T-T)/3*T
        V_Hull = LBP*B*D*Cbprime
        V_Upperdecks = LBP*B*(D_Up-D)
        if V_Calculated < Vol_Total or V_Calculated > Vol_Total*1.1:
            if V_Calculated > Vol_Total*1.1:
            	#Decrease dimensions
                sign = -1
            else:
            	#Increase dimension
                sign = 1
            #Reset increments
            incr = np.zeros(5)
            if abs((Vol_Total-V_Calculated)/Vol_Total) < 0.95:
            	#Increment LBP
                incr[0] = 0.01*sign
                incr[1] = 0.0001*sign
                it_counter += 1
                if debug:
        			#Indicate the type of iteration failure
                	failed = 'Volume'
                continue
            if abs((Vol_Total-V_Calculated)/Vol_Total) < 0.9:
            	#Increment LBP
                incr[0] = 0.5*sign
                incr[1] = 0.000001*sign
                it_counter += 1
                if debug:
        			#Indicate the type of iteration failure
                	failed = 'Volume'
                continue
            if abs((Vol_Total-V_Calculated)/Vol_Total) < 0.75:
            	#Increment LBP
                incr[0] = 1*sign
                incr[1] = 0.000001*sign
                it_counter += 1
                if debug:
        			#Indicate the type of iteration failure
                	failed = 'Volume'
                continue
        """
        #---------------------------------------------------------------------

        #Calculate and check freeboard
        #---------------------------------------------------------------------
        frbrd = D-T
        #frbrd = na.freeboard()
        frbrd_req = 0.5
        if frbrd < frbrd_req:
            #Reset other constraints
            incr = np.zeros(5)
            #Adjust depth
            incr[4] = 0.1
            if debug:
        		#Indicate the type of iteration failure
            	failed = 'Volume'
            it_counter += 1
            continue
        #---------------------------------------------------------------------

        #Check stablity
        #---------------------------------------------------------------------
        GM_min = 1.0 #Set min GM value
        KG = D*0.7 #Assume KG
        KB = na.KB_estimate(LWL,B,T,Cw,V_Displ_HS)
        BM = na.BM_transverse_estimate(LWL,B,Cw,V_Displ_HS)
        GM = na.GM(KB,BM,KG)
        if GM < GM_min:
            #Reset increments
            incr = np.zeros(5)
            #Adjust B/T ratio
            incr[3] = 0.1
            incr[1] = 0.000001
            if debug:
        		#Indicate the type of iteration failure	
            	failed = 'GM'
            it_counter += 1
            continue
        #---------------------------------------------------------------------

        #Check roll period
        #---------------------------------------------------------------------
        target = 6 #Set minimum roll period target [s]
        T_roll, roll = na.roll_period_estimate(B, GM, target=target,
                                               check=True)
        T_roll_IMO, roll_IMO = na.roll_period_IMO(LWL, B, T, GM, 
        	target=target, check=True)

        if not roll or not roll_IMO:
            #Reset increments
            incr = np.zeros(5)
            #Adjust B/T ratio
            incr[3] = 0.1
            incr[1] = 0.000001
            if debug:
        		#Indicate the type of iteration failure
            	failed = 'Roll'
            it_counter += 1
            continue
        #---------------------------------------------------------------------
        
        #Calculate Froude number
        #---------------------------------------------------------------------
        Fn = pwr.Froude_Number(V,LWL,input_knots=False)
        #---------------------------------------------------------------------
        
        #Calculate longitudinal centre of buoyancy
        #---------------------------------------------------------------------
        if not lcb_Set:
            #Guldhammer HE, Harvald SA (1974) Ship resistance: Effect of form
            #and principal dimensions. Academisk Forlag, Copenhagen
            lcb = Fn*(-43.75)+9.375
        #---------------------------------------------------------------------

        #Calculate powering estimate
        #---------------------------------------------------------------------
        if not P_I_Set:
        	#Set parameters for holtrop analysis
        	#Vessel trimmed flat
        	Tf = T
        	Ta = T
        	#Assume transom is emerged
        	At = 0
        	#Assume a corellation between Abt and LBP
        	#To be revisited, Abt should be a function of B,T
        	Abt = LBP*0.0775+0.6
        	#Assume a corellation between hb and draft
        	hb = 0.604*T+0.03

        	P_E = pwr.powering_holtrop(V,LWL,B,T,V_Displ_HS,Cm,Cw,lcb,Tf,Ta,At,
                                    Abt,hb,Rapp=0,N_Screws=2,U_shaped=True)
        	P_E /= 1000 
        	P_I = P_E/0.68
        #---------------------------------------------------------------------

        #Estimate weights
        #---------------------------------------------------------------------
        #Calculate machinery weight
        #W_Mach = 0
        #Get ratio from regression based on enclosed volume
        LBD_Up = LBP*B*D_Up*0.7
        if not W_LS_Set:
        	#Calculate approximate lightship weight
        	W_LS = dp.get_value_reg_lin(database,LBD_Up,'LBD_Up','W_LS')
        #Calculate fluid weights and volumes
        HFO = rpx.fluid_HFO(endurance,P_I)
        DO = rpx.fluid_DO(endurance,(P_I/10))
        LO = rpx.fluid_LO(endurance,P_I)
        FW = rpx.fluid_FW(endurance,N_Crew,N_Pas)
        W_Fluids = HFO[0]+DO[0]+LO[0]+FW[0]
        DWT = rpx.deadweight(endurance,N_Crew,N_Pas,L_Trailer_Lane,L_Car_Lane,
        W_Fluids,extra_weight=0,margin=15)
        
        #Calculate displacement weight with margins
        W_Displ = na.W_displ_comp(W_LS,DWT,margin=5)
        #W_LS, DWT, W_Displ = rpx.weights()
        V_Displ = W_Displ/1.025
        #---------------------------------------------------------------------

        #Calculate displacement error, check if the goal is met
        #---------------------------------------------------------------------
        displ_error = ((W_Displ-W_Displ_HS)*100)/W_Displ
        if abs(displ_error) < allowable_error:
            #Got it!
            goal = True
            if debug:
        		#Indicate the type of iteration failure
            	failed = 'None'
            break
        #---------------------------------------------------------------------

        #Adjust dimensions
        #---------------------------------------------------------------------
        #Check in case ship is underdefined for its required volume
        if LOA == restrict_LOA and B == restrict_B and T == restrict_T\
        and Cb == Cb_max:
            print ('Ship generation failed')
            print (divider)
            print ('Ship too restricted in dimensions for required volume')
            print (divider)
            if debug:
                #Run debug
                debug_summary(ddat)
            return 1 
        #Reset increments
        incr = np.zeros(5)
        #Increment or decrement dimensions?
        if W_Displ >= W_Displ_HS:
            sign = 1
        else:
            sign = -1
        #Adjust particulars accordingly to the error
        if abs(displ_error) < 0.1:
        	if Cb_Set:
        		#Check for length constraints
        		if LBP_Set or LOA_Set or LOA == restrict_LOA:
        			#Increment B/T
        			incr[2] = 0.001*sign
        		else:
        			#Increment LBP
        			incr[0] = 0.001*sign
        	else:
        		#Increment Cb
        		incr[1] = 0.00001*sign
        elif abs(displ_error) < 1:
        	#Fix main dimensions, use ceil function
        	#
        	if Cb_Set:
        		if LBP_Set or LOA_Set or LOA == restrict_LOA:
        			incr[2] = 0.005*sign
        		else:
        			incr[0] = 0.005*sign
        	else:
        		if Cb == Cb_max or Cb == Cb_min:
        			incr[0] = 0.005*sign
        		else:
        			incr[1] = 0.0001*sign
        elif abs(displ_error) < 5:
        	if LBP_Set or LOA_Set or LOA == restrict_LOA:
        		#Increment B/T
        		incr[2] = 0.03*sign
        	else:
        		#Increment LBP
        		incr[0] = 0.1*sign
        elif abs(displ_error) < 10:
        	if LBP_Set or LOA_Set or LOA == restrict_LOA:
        		#Increment B/T
        		incr[2] = 0.05*sign
        	else:
        		#Increment LBP
        		incr[0] = 0.5*sign
        elif abs(displ_error) > 10:
        	if LBP_Set or LOA_Set or LOA == restrict_LOA:
        		#Increment B/T
        		incr[2] = 0.1*sign
        	else:
        		#Increment LBP
        		incr[0] = 1*sign
        if debug:
        	#Indicate the type of iteration failure
        	failed = 'Displacement'
        #---------------------------------------------------------------------

        #Increment iteration counter
        #---------------------------------------------------------------------
        it_counter += 1
        #---------------------------------------------------------------------

        #End of iteration
    #-------------------------------------------------------------------------

    #Calculate extra data
    #-------------------------------------------------------------------------
    #Calculate propeller diameter
    D_Prop = dp.get_value_reg_lin(database,T,'T','D_Prop')
    #-------------------------------------------------------------------------

    #Run debugging
    #-------------------------------------------------------------------------
    if debug:
    	debug_summary(ddat)
    #-------------------------------------------------------------------------

    #Estimate auxilary power
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    #Run techno-economic analysis
    #-------------------------------------------------------------------------
    if te_analysis:
        #te.steel_cost(W_S, PHTS=0)
        print ("Techno-economic analysis completed successfully")
        print (divider)
    #-------------------------------------------------------------------------

    #Save the ship data in a .csv
    #-------------------------------------------------------------------------
    particulars = itdt
    particulars.to_csv('Data_Output\Ship_Particulars.csv', index=False)
    print('Particulars have been saved')
    print(divider)
    #-------------------------------------------------------------------------
    
    #Print summary and particulars
    #-------------------------------------------------------------------------
    print('Ship completed within {no} iterations'.format(no=it_counter))
    print(divider)
    print('Ship particulars:')
    print('LOA={:.2f}, LBP={:.2f}, LWL={:.2f}'.format(LOA,LBP,LWL))
    print('B={:.2f}, T={:.2f}, D={:.2f}'.format(B,T,D))
    print('D_Up={:.2f}, Cb={:.2f}, V_S={:.2f}'.format(D_Up,Cb,V))
    print('W_LS={:.2f}, DWT={:.2f}, W_Displ={:.2f}'.format(W_LS,DWT,W_Displ))
    print('L/B={:.2f}, B/T={:.2f}, L/D={:.2f}'.format(LBratio,BTratio,
    	LDratio))
    print('L/D_Up={:.2f},T/D={:.2f}, T/D_Up={:.2f}'.format(LDupratio,TDratio,
    	TDupratio))
    print('P_I={:.2f},D_Prop={:.2f}'.format(P_I,D_Prop))
    print(divider)
    #-------------------------------------------------------------------------

    return particulars