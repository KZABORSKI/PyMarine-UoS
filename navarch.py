# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Naval architecture module
"""
#%% IMPORT
import math
import numpy as np

#%% Block coefficient
def Block_Coeff(L,B,T,V_displ):
    """
    --------------------------------------------------------------------------
    Calculate block coefficient
    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    B - float, beam [m]
    T - float, draft [m]
    V_displ - float, displacement volume [m^3]
    --------------------------------------------------------------------------
    Output:
    Cb - float, block coefficient [-]
    --------------------------------------------------------------------------
    """
    Cb=V_displ/(L*B*T)
    return Cb

#%% Prismatic coefficient
def Prism_Coeff(Am,L,V_displ):
    """
    --------------------------------------------------------------------------
    Calculate block coefficient
    --------------------------------------------------------------------------
    Input:
    Am - float, midship section area [m^2]
    L - float, length [m]
    V_displ - float, displacement volume [m^3]
    Output:
    --------------------------------------------------------------------------
    Cp - float, prismatic coefficient [-]
    [Also CP=CB/CM]
    --------------------------------------------------------------------------
    """
    Cp=V_displ/(Am*L)
    return Cp

#%% Waterplane area coefficient
def WaterPl_Coeff(Aw,L,B):
    """
    --------------------------------------------------------------------------
    Calculate waterplane area coefficient
    --------------------------------------------------------------------------
    Input:
    Aw - float, water plane area [m^2]
    L - float, length [m]
    B - float, beam [m]
    Output:
    --------------------------------------------------------------------------
    Cw - float, waterplane area coefficient [-]
    --------------------------------------------------------------------------
    """
    Cw=Aw/(L*B)
    return Cw

#%% Midship coefficient
def Midship_Coeff(Am,
                  B,
                  T):
    """
    --------------------------------------------------------------------------
    Calculate block coefficient
    --------------------------------------------------------------------------
    Input:
    Am - float, midship section area [m^2]
    B - float, beam [m]
    T - flotat, draft [m]
    Output:
    --------------------------------------------------------------------------
    Cm - float, midship coefficient [-]
    --------------------------------------------------------------------------
    """
    Cm=Am/(B*T)
    return Cm

#%% Length-displacement ratio
def LDispl_ratio(L,V_displ):
    """
    --------------------------------------------------------------------------
    Calculate block coefficient
    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    V_displ - float, displacement volume [m^3]
    --------------------------------------------------------------------------
    Output:
    LDrat - float, length-displacement ratio [-]
    --------------------------------------------------------------------------
    """
    LDrat = L/(V_displ)**(1/3)
    return LDrat

#%% Calculate displacement
def W_displ_comp(W_LS,
                 DWT,
                 margin=0):
    """
    --------------------------------------------------------------------------
    Calculate displacement [t] lighship and DWT breakdown
    --------------------------------------------------------------------------
    Input:
    W_LS - float, lightship weight [t]
    DWT - float, deadweight [t]
    margin - float, margin [%]
    --------------------------------------------------------------------------
    Output:
    W_displ - float, displacement weight [t]
    --------------------------------------------------------------------------
    """
    W_displ = W_LS + DWT
    W_displ=W_displ*(1+margin/100)
    return W_displ

#%% Calculate displacement
def W_displ_princ(L,
                   B,
                   T,
                   CB,
                   rho=1.025,
                   margin=0):
    """
    --------------------------------------------------------------------------
    Calculate displacement [t] from principal particulars
    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    B - float, beam [m]
    T - flotat, draft [m]
    CB - float, block coefficient [-]
    rho - float, water density [t m^-3]
    margin - float, margin [%], default=0
    --------------------------------------------------------------------------
    Output:
    W_displ - float, displacement [m^3]
    --------------------------------------------------------------------------
    """
    W_displ=rho*L*B*T*CB
    W_displ=W_displ*(1+margin/100)
    return W_displ

#%% Estimate KB
def KB_estimate(L,B,T,Cw,V_displ):
    """
    --------------------------------------------------------------------------
    Estimate vertical centre of buoyancy KB
    Based on Moorish's formula approximation
    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    B - float, beam [m]
    T - flotat, draft [m]
    Cw - float, waterplane area coefficient [-]
    V_displ - float, displacement volume [m^3]
    --------------------------------------------------------------------------
    Output:
    KB - float, freeboard of the ship [m]
    --------------------------------------------------------------------------
    """
    #Reverse engineer waterplane area from Cp
    Aw = Cw*(L*B)
    KB = (5/6)*T-(V_displ/(3*Aw))
    return KB

#%% Calculate the metacentric height
def GM(KB,BM,KG):
    """
    --------------------------------------------------------------------------
    Calculate the metacentric height
    --------------------------------------------------------------------------
    Input:
    KB - float, vertical centre of buoyancy [m]
    BM - float, 
    KG - float, vertical centre of gravity [m]
    --------------------------------------------------------------------------
    Output:
    GM - float, metacentric height [m]
    --------------------------------------------------------------------------
    """
    GM = KB+BM-KG
    return GM

#%% Calculate BM
def BM_transverse(I_t,V_displ):
    """
    --------------------------------------------------------------------------
    Calculate BM
    --------------------------------------------------------------------------
    Input:
    I_t - float, transverse second moment of area [m^4]
    V_displ - float, displacement volume [m^3]
    --------------------------------------------------------------------------
    Output:
    BM - float, distance between centre of buoyancy and metacentre [m]
    --------------------------------------------------------------------------
    """
    BM_t = I_t/V_displ
    return BM_t

#%% Estimate BM
def BM_transverse_estimate(L,B,Cw,V_displ):
    """
    --------------------------------------------------------------------------
    Estimate BM
    Ship Design and Economics Booklet 2017
    University of Southampton; Ship Design and Economics; Shenoi, Pomeroy,
    Keane, Taunton; page 25
    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    B - float, beam [m]
    Cw - float, waterplane area coefficient [-]
    V_displ - float, displacement volume [m^3]
    --------------------------------------------------------------------------
    Output:
    BM - float, distance between centre of buoyancy and metacentre [m]
    --------------------------------------------------------------------------
    """
    I_t = ((L*(B**3))/12)*(6*Cw**3)/((1+Cw)*(1+2*Cw))
    BM_t = I_t/V_displ
    return BM_t

#%% Estimate roll period
def roll_period_estimate(B, GM, K=0.43, target=0, check=False):
    """
    --------------------------------------------------------------------------
    Calculate approximate roll period
    Ship Design and Economics Booklet 2017
    University of Southampton; Ship Design and Economics; Shenoi, Pomeroy,
    Keane, Taunton; page 42
    --------------------------------------------------------------------------
    Input:
    B - float, beam [m]
    GM - float, metacentric height [m]
    K - float, constant ~(0.4<K<0.55) [-]
    --------------------------------------------------------------------------
    Output:
    T_roll - float, period of roll motion [s]
    --------------------------------------------------------------------------
    """
    T_roll = (K*B**2)/np.sqrt(GM)
    if check:
      if T_roll > target:
        return T_roll, True
      else:
        return T_roll, False
    return T_roll

#%% Calculate roll period
def roll_period_IMO(LWL, B, T, GM, target=0, check=False):
    """
    --------------------------------------------------------------------------
    Calculate roll period based on IMO guidelines
    Intact Stability Code, 2008, edition 2009
    --------------------------------------------------------------------------
    Rolling period:
    T = (2*C*B)(GM)^(-0.5) [s]
    Where:
    C = 0.0373+0.023(B/d)-0.043(Lwl/100)
    Lwl = length of the ship at the waterline [m]
    B = moulded breadth of the ship [m]
    d = mean moulded draught of the ship [m]
    --------------------------------------------------------------------------
    Input:
    LWL - float, length of the waterline [m]
    B - float, beam [m]
    GM - float, metacentric height [m]
    --------------------------------------------------------------------------
    Output:
    T_roll - float, period of roll motion [s] 
    """
    d=T
    C = 0.373+0.023*(B/d)-0.043*(LWL/100)
    T_roll = (2*C*B)/np.sqrt(GM)
    if check:
      if T_roll > target:
        return T_roll, True
      else:
        return T_roll, False
    return T_roll
