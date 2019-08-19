# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Powering analysis library module
"""
#%% IMPORT
import math
import warnings
import numpy as np
import navarch as na

#%% Froude number
def Froude_Number(V,
                  L,
                  input_knots=False):
    """
    --------------------------------------------------------------------------
    Calculate Froude number
    --------------------------------------------------------------------------
    Input:
    V - float, speed [knots]
    L - float, length [m]
    input_knots - bool, speed in knots? default=False
    --------------------------------------------------------------------------
    Output:
    FN - float, Froude number [-]
    --------------------------------------------------------------------------
    """
    if input_knots:
        V=V*0.5144
    else:
        pass
    g=9.81
    Fn = V/(np.sqrt(g*L))
    return Fn

#%% Reynolds number
def Reynolds_Number(V,
                    L,
                    kinematic_viscocity=1.19e-6,
                    input_knots=False):
    """
    --------------------------------------------------------------------------
    Calculate Reynolds number
    --------------------------------------------------------------------------
    Input:
    V - float, speed [knots]
    L - float, length [m]
    kinematic_viscocity - float, [m^2 s^-1], default=1.19e-6
    input_knots - bool, speed in knots? default=False
    --------------------------------------------------------------------------
    Output:
    Rn - float, Reynolds number [-]
    --------------------------------------------------------------------------
    Comments:
    kinematic viscocity of fresh water = 1.14 x 10^-6 m^2 s^-1
    kinematic viscocity of salt water =  1.19 x 10^-6 m^2 s^-1
    --------------------------------------------------------------------------
    """
    if input_knots:
        V=V*0.5144
    else:
        pass
    Rn = V*L/kinematic_viscocity
    return Rn

#%% Full scale speed
def full_scale_V(Vm,
                 scaleF):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Vm - model speed [ms^-1]
    scaleF - scale factor
    --------------------------------------------------------------------------
    Output:
    Vs - full scale ship speed [ms^-1]
    --------------------------------------------------------------------------
    """
    Vs=Vm*np.sqrt(scaleF)
    return Vs

#%% ITTC 57 Friction coefficient
def Cf_ITTC57(Rn):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Rn - Reynolds number [-]
    --------------------------------------------------------------------------
    Output:
    Cf - Friction coefficient basen on ITTC 57 regression line [-]
    --------------------------------------------------------------------------
    """
    Cf = 0.075/(((math.log10(Rn))-2)**2)
    return Cf

#%% Roughness allowance:
def roughness_allowance(L,
                        ks=150e-6):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    L - float, length [m]
    ks - float, roughness [m]
    --------------------------------------------------------------------------
    Output:
    delta_Cf - roughness allowance [-]
    --------------------------------------------------------------------------
    """
    delta_Cf = (105*(ks/L)**(1/3)-0.64)*10e-3
    return delta_Cf
#%% Coefficient of total resistance
def Ctm(Rtm,
        Sm,
        Vm,
        rho=1025,
        input_knots=True):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Rtm - model resistance [N]
    Sm - model wetted surface area [m^2]
    Vm - model speed [ms^-1]
    rho - float, water density [kg m^-3], default=1025
    --------------------------------------------------------------------------
    Output:
    Ctm - model total resistance coefficient [-]
    --------------------------------------------------------------------------
    """
    if input_knots:
        V *= 0.5144
    Ctm = Rtm/(0.5*rho*Sm*Vm**2)
    return Ctm

#%% Coefficient of residuary resistance
def Cr(Ctm,
       Cfm):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Ctm - model total resistance coefficient [-]
    Cfm - model frictional resistance coefficient [-]
    --------------------------------------------------------------------------
    Output:
    Cr - residuary resistance coefficient [-]
    --------------------------------------------------------------------------
    """
    Cr = Ctm-Cfm
    return Cr

#%% Coefficient of air resistance
def air_res_coeff(ATproj,
                    S):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    ATproj - float, transverse projected area of ship [m^2]
    S - float, wetted surface area [m^2]
    --------------------------------------------------------------------------
    Output:
    Cair - float, air resistance coefficient [-]
    --------------------------------------------------------------------------
    """
    Cair = ATproj/(1000*S)
    return Cair

#%% Sum resistance coefficients
#CHECK for air resistance
def sum_res_coeff(Cf,
                  Cr,
                  Cap=0,
                  Cair=0,
                  k=None):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Cf - float, frictional resistance coefficient [-]
    Cr - float, residuary resistance coefficient [-]
    Cap - float, appendage resistance coefficient [-]
    Cair - float, air resistance coefficient [-]
    --------------------------------------------------------------------------
    Output:
    Ct - float, total resistance coefficient [-]
    --------------------------------------------------------------------------
    """
    if k != None:
        Cf = Cf*(1+k)
    else:
        pass

    Ct = np.sum([Cf,Cr,Cap,Cair])

    return Ct

#%% Total resistance
def resistance_total(Ct,
                     S,
                     V,
                     rho=1025,
                     margin=0,
                     input_knots=True):
    """
    --------------------------------------------------------------------------
    Calculate total resistance based on total resistance coefficient
    --------------------------------------------------------------------------
    Input:
    Ct - float, coefficient of total resistance [-]
    S - float, wetted surface area [m^2]
    V - float, ship speed [knots]
    rho - float, water density ([m^-3], default=1.025
    margin - float, sea/fouling margin [%]
    input_knots - bool, if true, the input speed is in knots, otherwise it is
    in ms^-1, default=True
    --------------------------------------------------------------------------
    Output:
    Rt - float, rotal resistance [kN]
    --------------------------------------------------------------------------
    """
    #Convert speed to ms^1
    if input_knots:
        V *= 0.5144
    Rt = 0.5*Ct*rho*S*(V)**2
    Rt = Rt*(1+(margin/100)) #apply percentage margin
    return Rt

#%% Sum of resistance components
def resistance_sum(Rf,
                    one_k,
                    Rw,
                    Rtr,
                    Rapp=0,
                    Ra=0):
    """
    --------------------------------------------------------------------------
    Calculate the sum of resistance components
    --------------------------------------------------------------------------
    Input:
    Rf - float, frictional resistance according to the ITTC-57 friction 
    formula [N]
    one_k - float, form factor (1+k1) decribing the viscous resistance of
    the hull form in relation to Rf [-]
    Rapp - float, resistance of appendages [N]
    Rw - float, wave making and wave breaking resistance [N]
    Rb - float, additional pressure resistance of bulbous bow near the water
    surface [N]
    Rtr - float, additional pressure resistance of immersed transom stern [N]
    Ra - float, model-ship correlation resistance [N]
    --------------------------------------------------------------------------
    Output:
    Rt - rotal resistance [kN]
    --------------------------------------------------------------------------
    """
    Rt = np.sum([(Rf*one_k),Rapp,Rw,Rtr,Ra])
    return Rt

#%% Powering curve, effective power
def power_effective(V,
                   Rt,
                   margin=0,
                   input_knots=True,
                   plot=False,
                   output_HP=False):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    Vs - ship speed [knots]
    Rt - rotal resistance [kN]
    margin - margin [%]
    input_knots - bool, speed in knots? default=True
    plot - bool, display graph or not, default=False
    output_HP - bool, output power in HP? default=False
    --------------------------------------------------------------------------
    Output:
    PE - float, effective power [kW or HP]
    --------------------------------------------------------------------------
    """
    #Speed in knots or ms^-1 ?
    if input_knots:
        PE = V*Rt*0.5144
        PE = PE*(1+(margin/100)) #Apply percentage margin
    else:
        PE = V*Rt
    #Plot?
    if plot:
        pass
    #Convert to horse power if indicated
    if output_HP:
        PE = PE*1.34102
    else:
        pass
    return PE

#%% Delivered power:
def power_delivered(PE,
                    nu_D):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    PE - float, effective power [kW]
    nu_D - float, quasi propulsive efficiency [-]
    --------------------------------------------------------------------------
    Output:
    PD - float, delivered power [kW]
    --------------------------------------------------------------------------
    """
    PD = PE/nu_D
    return PD

#%% Installed power:
def power_installed(PE,
                    nu_D,
                    SCF=1,
                    nu_T=0.98,
                    margin=0):
    """
    --------------------------------------------------------------------------

    --------------------------------------------------------------------------
    Input:
    PE - float, effective power [kW]
    nu_D - float, quasi propulsive efficiency [-]
    SCF - float, model-ship correlation factor [-], default=1
    nu_T - float, transmission efficiency [-]
    margin - sea/fouling margin [%], usually 15%-30%
    --------------------------------------------------------------------------
    Output:
    PI - float, installed power [kW]
    --------------------------------------------------------------------------
    """
    PI = (PE/nu_D)*SCF*(1/nu_T)
    PI = PI*(1+(margin/100)) #Apply percentage margin
    return PI

#%% Resistance of bowthruster tunnel
def resistance_BTtunnel(d,
                     V,
                     Cbto,
                     rho=1025,
                     input_knots=True):
    """
    --------------------------------------------------------------------------
    Calculate resistance of bowthruster tunnel using method described by
    Holtrop and Mennen.
    J.Holtrop, G.G.J. Mennen, 'AN APPROXIMATE POWER PREDICTION METHOD',
    International Shipbuilding Progress, Vol29, July 1982
    --------------------------------------------------------------------------
    Input:
    d - float, coefficient of frictional resistance [-]
    V - float, ship speed [knots]
    Cbto - float, bow thruster tunnel opening coefficient [-]
    see notes for further explanation
    rho - float, water density ([m^-3], default=1.025
    input_knots - bool, if true, the input speed is in knots, otherwise it is
    in ms^-1, default=True
    --------------------------------------------------------------------------
    Output:
    Rbto - float, additional appendage resistance of bow thruster tunnel [N]
    --------------------------------------------------------------------------
    NOTES:
    The coefficient Cbto ranges from 0.003 to 0.012. For openings in the
    cylindrical part of a bulbous bow the lower figures should be used.
    --------------------------------------------------------------------------
    """
    #Convert speed to ms^1
    if input_knots:
        V *= 0.5144

    Rbto = rho*(V**(2))*math.pi*(d**2)*Cbto

    return Rbto

#%% Appendage resistance
def resistance_appendage(Cf,
                     Sapp,
                     V,
                     one_k2,
                     rho=1025,
                     input_knots=True):
    """
    --------------------------------------------------------------------------
    Calculate resistance of an appendage using method described by Holtrop
    and Mennen.
    J.Holtrop, G.G.J. Mennen, 'AN APPROXIMATE POWER PREDICTION METHOD',
    International Shipbuilding Progress, Vol29, July 1982
    --------------------------------------------------------------------------
    Input:
    Cf - float, coefficient of frictional resistance [-]
    Sapp - float, appendage wetted surface area [m^2]
    V - float, ship speed [knots]
    one_k2 - float, appendage resistance factor (1+k2) [-]
    see notes for further explanation
    rho - float, water density ([m^-3], default=1.025
    input_knots - bool, if true, the input speed is in knots, otherwise it is
    in ms^-1, default=True
    --------------------------------------------------------------------------
    Output:
    Rapp - float, resistance of appendage [N]
    --------------------------------------------------------------------------
    NOTES:
    To validate the software, use the example from the paper:
    results = resistance_appendage(0.001390,50,25,1.5,rho=1025,
                                   input_knots=True)
    Approximate values pf 'one_k2' (1+k2) for streamlined flow-oriented
    appendages are given below:
    -----------------------------------------
    rudder behind skeg              1.5 - 2.0
    rudder behind stern             1.3 - 1.5 
    twin-screw balance rudders      2.8
    shaft brackets                  3.0
    skeg                            1.5 - 2.0
    strut bossings                  3.0
    hull bossings                   2.0
    shafts                          2.0 - 4.0
    stabilizer fins                 2.8
    dome                            2.7
    bilge keels                     1.4
    -----------------------------------------
    Equivalent 'one_k2' (1+k2) value for a combination of appendages:
    (one_k2)eq = sum((one_k2)*Sapp)/sum(Sapp)
    --------------------------------------------------------------------------
    """
    #Convert speed to ms^1
    if input_knots:
        V *= 0.5144

    Rapp = 0.5*rho*(V**(2))*Sapp*one_k2*Cf

    return Rapp

def powering_holtrop(V,LWL,B,T,V_displ,Cm,Cwp,lcb,Tf,Ta,At,Abt,hb,S=0,ie=0,
                    Rapp=0,N_Screws=1,rho=1025,sea_margin=25,P=0,D=0,V_shaped=False,
                    U_shaped=False,propulsion=False,detailed_output=False):
    """
    --------------------------------------------------------------------------
    Calculate powering requirements using Holtrop and Mennen regression 
    analysis.
    J.Holtrop, G.G.J. Mennen, 'AN APPROXIMATE POWER PREDICTION METHOD',
    International Shipbuilding Progress, Vol29, July 1982
    --------------------------------------------------------------------------
    Input:
    V - float, ship speed [knots]
    LWL - float, length of the waterline [m]
    B - float, beam [m]
    T - float, draft [m]
    V_displ - float, displacement volume [m^3]
    Cm - float, midship coefficient [-]
    Cwp - float, waterplane area coefficient [-]
    lcb - float, longitudinal position of the centre of buoyancy forward
    of 0.5LWL as a percentage of LWL [%]
    Tf - float, draft at FP [m]
    Ta - float, draft at AP [m]
    At - float, immersed transom area at 0 speed, in this figure the
    transverse area o f wedges placed at the transom chine should be
    included [m^2]
    Abt - float, transverse sectional area of the bulb at the position where
    the still-water surface intersects the stem [m]
    hb - float, position of the centre of transverse area Abt above the B.L.
    S - float, wetted surface area of the hull [m^2]
    ie - float, half angle of entrance, angle of the waterline at the bow in
    degrees wit h reference to the centre plane but neglecting the local shape
    at the stem [deg]
    Rapp - float, resistance of appendages [N] default=0
    N_Screws - float, number of propellers [-] default=1
    rho - float, water density [kg m^-3] default=1025
    sea_margin - float, resistance sea margin [%] default=25
    P - float, propeller pitch [m] default=0
    D - float, propeller diameter [m] default=0
    V_shaped - bool, true if stern sections are V-shaped, default=False
    U_shaped - bool, true if (Hogner stern) stern sections are U-shaped, 
    default=False
    detailed_output - bool, return detailed output, default=False
    --------------------------------------------------------------------------
    Output:
    -----
    Default:
    PE - float, effective power [N]
    -----
    If detailed results chosen:
    PE, [Fn,Cp,Lr,lcb,c12,c13,one_k1,S,Cf,Rf,Rapp,c7,ie,c1,c3,c2,c5,m1,c15,m2,
    lmbd,Rw,Pb,Fni,Rb,Fnt,Rtr,c4,Ca,Ra,Rt]
    -----
    If propulsion calculation chosen:
    --------------------------------------------------------------------------
    Intermediate parameters:
    Cb - float, block coefficient based on LWL [-]
    Cp - float, prismatic coefficient based on LWL [-]
    c2 - parameter which accounts for the reduction of the wave resistance due
    to the accumulation of a bulbous bow
    c3 - coefficient that determines the influence of the bulbous bow on the
    wave resistance
    c5 - coefficient expressing influence of transom stern on the wave
    resistance
    Lr - float, parameter reflecting the length of the run
    V_displ - float, displacement volume [m^3]
    Pb - float, measure for the emergence of the bow [-]
    --------------------------------------------------------------------------
    NOTE:
    To validate the software, use the example from the paper:
    results = powering_holtrop(25,205,32,10,37500,0.980,0.750,-0.75,10,10,16,
                               20,4,S=0,ie=0,Rapp=8830,rho=1025,
                               V_shaped=False,U_shaped=True,
                               detailed_output=True)
    print(results)
    --------------------------------------------------------------------------
    """
    #Acceleration due to gravity
    g = 9.81
    #Ship speed in [m s^-1]
    V *= 0.5144
    #Coefficients
    Cb = V_displ/(LWL*B*T)
    Cp = Cb/Cm
    #Convert lcb position from relative to 1/2 LBP, to 1/2 LWL

    #Frictional resistance
    #-------------------------------------------------------------------------
    #Check S for input
    if S == 0:
        #Calculate approximate wetted area of the hull
        #Double check the signs later
        S = LWL*(2*T+B)*(math.sqrt(Cm))\
        *(0.453+0.4425*Cb-0.2862*Cm-0.003467*(B/T)+0.3696*Cwp)\
        +2.38*(Abt/Cb)
    Rn = Reynolds_Number(V,LWL)
    Cf = Cf_ITTC57(Rn)
    Rf = resistance_total(Cf,S,V,input_knots=False)
    #-------------------------------------------------------------------------

    #Form factor
    #-------------------------------------------------------------------------
    #Calculate Lr parameter
    Lr = (1-Cp+(0.06*Cp*lcb)/(4*Cp-1))*LWL

    #Calculate c12 coefficient
    if T/LWL > 0.05:
        c12 = (T/LWL)**(0.2228446)
    elif T/LWL > 0.02 and T/LWL < 0.05:
        c12 = 48.2*((T/LWL-0.02)**(2.078))+0.479948
    else:
        c12 = 0.479948

    #Calculate c13 coefficient
    #Check input consistency
    if V_shaped and U_shaped:
        warnings.warn('Inconsistent input (set either v_shaped OR u_shaped)')
    #Set Cstern
    Cstern = 0
    if V_shaped:
        #Adjust Cstern
        Cstern = -10
    if U_shaped:
        #Adjust Cstern
        Cstern = 10
    c13 = 1+0.003*Cstern

    one_k1 = c13*(0.93+(c12*((B/Lr)**(0.92497)))\
            *((0.95-Cp)**(-0.521448))\
            *((1-Cp+0.0225*lcb)**(0.6906)))
    #-------------------------------------------------------------------------

    #Wave resistance
    #-------------------------------------------------------------------------
    #Convert displacement [t] to [kg]
    V_displ = Cb*LWL*B*T
    #Calculate Froude number
    Fn = Froude_Number(V,LWL,input_knots=False)
    #Calculate c2,c3,c5
    c3 = 0.56*(Abt**(1.5))/(B*T*(0.31*(math.sqrt(Abt))+Tf-hb))
    c2 = math.exp(-1.89*math.sqrt(c3))
    c5 = 1-0.8*At/(B*T*Cm)
    #Calculate lambda
    if LWL/B < 12:
        lmbd = 1.446*Cp - 0.03*(LWL/B)
    else:
        lmbd = 1.446*Cp - 0.36
    #Calculate c7
    if B/LWL <= 0.11:
        c7 = 0.229577*((B/LWL)**(0.33333))
    elif B/LWL > 0.11 and B/LWL < 0.25:
        c7 = B/LWL
    else:
        c7 = 0.5-0.0625*(LWL/B)
    #Calculate c15
    if (LWL**(3))/V_displ < 512:
        c15 = -1.69385
    elif (LWL**(3))/V_displ > 1727:
        c15 = 0
    else:
        c15 = -1.69385+((LWL**(3))/V_displ-8.0)/2.36
    #Calculate c16
    if Cp < 0.8:
        c16 = 8.07981*Cp-13.8673*(Cp**(2))+6.984388*(Cp**(3))
    else:
        c16 = 1.73014-0.7067*Cp
    #Calculate m1
    m1 = 0.0140407*(LWL/T)-1.75254*(V_displ**(1/3))/LWL\
        -4.79323*(B/LWL)-c16
    #Calculate m2
    m2 = c15*(Cp**(2))*math.exp(-0.1*(Fn**(-2)))
    #Unless defined, calculate ie
    if ie == 0:
        ie = 1+89*(math.exp(-((LWL/B)**(0.80856))\
                *((1-Cwp)**(0.30484))\
                *((1-Cp-0.0225*lcb)**(0.6367))\
                *((Lr/B)**(0.34574))\
                *((100*(V_displ/((LWL)**(3))))**(0.16302))))
    #Calculate c1
    c1 = 2223105*(c7**(3.78613))*((T/B)**(1.07961))*((90-ie)**(-1.37565))
    Rw = c1*c2*c5*V_displ*rho*g\
        *math.exp(m1*(Fn**(-0.9))+m2*math.cos(lmbd*(Fn**(-2))))
    #-------------------------------------------------------------------------

    #Additional resistance due to presence of bulbous bow near the surface
    #-------------------------------------------------------------------------
    #Calculate measure for the emergence of the bow
    Pb = 0.56*(math.sqrt(Abt))/(Tf-1.5*hb)
    Fni = V/(math.sqrt(g*(Tf-hb-0.25*(math.sqrt(Abt)))+0.15*(V**2)))
    Rb = 0.11*(math.exp(-3*(Pb**(-2))))*(Fni**(3))*(Abt**(1.5))\
            *rho*g/(1+Fni**(2))
    #-------------------------------------------------------------------------

    #Additional pressure resistance of immersed transom stern
    #-------------------------------------------------------------------------
    #Check if transom is immersed
    if At != 0:
        Fnt = V/(math.sqrt(2*g*At/(B+B*Cwp)))
        if Fnt < 5:
            c6 = 0.2*(1-0.2*Fnt)
        else: 
            c6 = 0
        Rtr = 0.5*rho*(V**2)*At*c6
    else:
        Rtr = 0
    #-------------------------------------------------------------------------

    #Model-ship correlation resistance
    #-------------------------------------------------------------------------
    if Tf/LWL > 0.04:
        c4 = 0.04
    else:
        c4 = Tf/LWL
    Ca = 0.006*((LWL+100)**(-0.16))-0.00205+0.003*(math.sqrt(LWL/7.5))\
            *(Cb**(4))*c2*(0.04-c4)
    Ra = 0.5*rho*(V**2)*S*Ca
    #-------------------------------------------------------------------------

    #Calculate total resistance
    #-------------------------------------------------------------------------
    Rt = np.sum([(Rf*one_k1),Rapp,Rw,Rb,Rtr,Ra])
    #-------------------------------------------------------------------------

    #Calculate effective power
    #-------------------------------------------------------------------------
    PE = Rt*V
    PEservice = PE*(1+sea_margin/100)
    #-------------------------------------------------------------------------

    if propulsion:
        #Check input
        #---------------------------------------------------------------------
        if N_Screws > 3 or N_Screws < 1:
            warnings.warn('Can only determine powering for single or\
                        twin-screw ships')
        #---------------------------------------------------------------------

        #Calculate wake fraction
        #---------------------------------------------------------------------
        #Single screw
        if N_Screws == 1:
            #Calculate Cv
            Cv = one_k1*Cf+Ca
            #Calculate c8
            if B/Ta < 5:
                c8 = B*S/(LWL*D*Ta)
            else:
                c8 = S*((7*B/Ta)-25)/(L*D*((B/Ta)-3))
            #Calculate c9
            if c8 > 28:
                c9 = 32-16/(c8-24)
            else:
                c9 = c8
            #Calculate c11
            if Ta/D < 2:
                c11 = Ta/D
            else:
                c11 = 0.0833333*((Ta/D)**3)+1.33333
            #Calculate Cp1
            Cp1 = 1.45*Cp-0.315-0.0225*lcb

            w = c9*Cv*(LWL/Ta)*(0.0661875+1.21756*c11*(Cv/(1-Cp1)))\
                +0.24558*(math.sqrt(B/(LWL*(1-Cp1))))-(0.09726/(0.95-Cp))\
                +(0.11434/(0.95-Cb))+0.75*Cstern*Cv+0.002*Cstern
        #Twin screw
        if N_Screws == 2:
            w = 0.3095*Cb+10*Cv*Cb-0.23*D/(math.sqrt(B*T))
        #---------------------------------------------------------------------

        #Calculate thrust deduction
        #---------------------------------------------------------------------
        #Single screw
        if N_Screws == 1:
            #Calculate c10
            if LWL/B > 5.2:
                c10 = B/LWL
            else:
                c10 = 0.25-0.003328402/((B/LWL)-0.134615385)

            t = 0.001979*L/(B-B*Cp1)+1.0585*c10-0.00524-0.1418*(D**2)/(B*T)\
                +0.0015*Cstern
        #Twin screw
        if N_Screws == 2:
            t = 0.325*Cb-0.1885*D/(math.sqrt(B*T))
        #---------------------------------------------------------------------

        #Calculate relative-rotative efficiency
        #---------------------------------------------------------------------
        #Single screw
        if N_Screws == 1:
            nu_r = 0.992-0.05908*AeAo+0.07424*(Cp-0.0225*lcb)
        #Twin screw
        if N_Screws == 2:
            nu_r = 0.9737+0.111*(Cp-0.0225*lcb)-0.06325*(P/D)
        #---------------------------------------------------------------------

    if detailed_output:
        results = [Fn,Cp,Lr,lcb,c12,c13,one_k1,S,Cf,Rf,Rapp,c7,ie,c1,c3,c2,c5,
        m1,c15,m2,lmbd,Rw,Pb,Fni,Rb,Fnt,Rtr,c4,Ca,Ra,Rt]
        return PE, results
    else:
        return PE
