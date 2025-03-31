#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import sys


import flopy
from flopy.modflow import ModflowBas, ModflowChd,ModflowDis
from flopy.mfusg import (MfUsg, MfUsgDisU, MfUsgLpf, MfUsgSms, 
MfUsgBct, MfUsgRch, MfUsgOc)
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen
from flopy.plot import PlotCrossSection,PlotMapView

import flopy.utils.binaryfile as bf

from tempfile import TemporaryDirectory

#%%

# Define model workspace where files will be written to
model_ws = "Brusseau_2021"

# temp_dir = TemporaryDirectory()
# model_ws = temp_dir.name

mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws= model_ws,
    modelname="Brusseau_2021",
    exe_name="mfusg_gsi",
)

# The setup simulated here consists of a 1-dimensional vertical soil column of 0.35 mm sand, 15 cm long, with a steady-state recharge from the top that gives a pore velocity of 37 cm/hr. The soil column was discretized uniformly into 30 numerical layers of 0.5 cm thickness each with a bottom elevation of zero and a top elevation of 15 cm.

#%% Parameter definitions

############################ Column parameters ################################

# Inner radius of column [cm]
r = 1
# Column recharge [cm/h]
q_column = 10 #3.08
# PFOS recharge concentration per time period [mg/l]
rch_con = 0.1


####################### General transport parameters ##########################

# Porosity
porosity = 0.33
# Horizontal hydraulic conductivity [cm/h]
hk = 80
# Vertical hydraulic conductivity [cm/h]
vka = 80
# Diffusion coeffcient [cm2/h]
diffnc = 0.01944
# Longitudinal dispersivity [cm]
dl = 2
# Transverse dispersivity [cm]
dt = 0
# Solid phase adsopriton coefficient
Kd = 0.25
# Bulk density of soil [g/cm3]
rho_bulk = 1.5
# Soil adsorption type; 0 = No adsorption, 1 = Linear adsorption, 2 = Freundlich adsorption, 3 = Langmuir adsorption
iadsorb = 2
# Freundlich (if iadsorb = 2) or Langmuir (if iadsorb = 3) adsorption isotherm 
flich = 0.81


####################### Air-water transport parameters ########################

# Air-water interfacial area
Aaw = 216
# Air-water interfacial partition coefficient function type, see documentation for BCT package for what each means.
ikawi_fn=1
# Air-water distribution coefficient a, input depends on ikawi_fn, see documentation for BCT package
a_Kaw = 0.027
# Air-water distribution coefficient b, input depends on ikawi_fn, see documentation for BCT package
b_Kaw = 0


###################### Model discretization parameters ########################

# Note : Multiple hard coded dependencies in code on the discretization, be aware before changing

# Number of model rows 
nrow = 2
# Number of model columns
ncol = 2
# Length of a single row [cm]
delr = 0.5
# Length of a single column [cm]
delc = 0.5
# Number of model layers
nlay = 30
# Top of the model simulation
top = 15
# Thickness of each model layer [cm]
delv = 0.5
# Time period lengths [h]
perlen=[7, 18.025]
# Number of time periods
nper = len(perlen)


########################### Saturated flow parameters #########################

# van Genuchten alpha parameter for saturated simulation
alpha_sat = 0.08 
# van Genuchten beta parameter for saturated simulation
beta_sat = 3.2 
# Residual water saturation for saturated simulation
sr_sat = 0.2364
# Brooks-Correy exponent for relative permeability for saturated simulation 
brook_sat = 4
# Hydraulic head of the bottom of the column for saturated simulation [cm]
h_bottom_sat = 16.0
# Initial hydraulic head in the column for saturated simulation [cm]
h_start_sat = 16.0

########################## unsaturated flow parameters ########################

# van Genuchten alpha parameter for unsaturated simulation
alpha_unsat = 12.6  
# van Genuchten beta parameter for unsaturated simulation
beta_unsat = 1.16
# Residual water saturation
sr_unsat = 0.2364
# Brooks-Correy exponent for relative permeability for unsaturated simulation
brook_unsat = 4
# Hydraulic head of the bottom of the column for unsaturated simulation [cm]
h_bottom_unsat = -3
# Initial hydraulic head in the column for unsaturated simulation [cm]
h_start_unsat = -3

############################# Other flow parameters ###########################

# Air entry head 
hb = 0 
# Relative elevation of the bottom of the column [cm]
z_bottom = 0 


#%% Calculation of other parameters

# Inner area of soil column cm^2
area_column = np.pi * r**2
# Recharge volume [cm3/h]
Q_input = q_column * area_column
# Saturated pore volume
sat_pv = porosity * np.pi * r**2 * top
# Multiplication factor to convert hours to pore volumes
hours_to_pv = sat_pv / Q_input

# Capillary head for unsaturated simulation [cm]
hc_unsat = z_bottom - h_bottom_unsat 
# van Genuchten gamma parameter for unsaturated simulation
gamma_unsat = 1 - (1 / beta_unsat) 
# Effective soil saturation for unsaturated simulation, equation as per information document for USGT by Panday, S. page 87
se_unsat = (1 + alpha_unsat * (abs(hc_unsat - hb))**beta_unsat)**-gamma_unsat
# Soil water saturation for unsaturated simulation
sw_unsat = se_unsat * (1 - sr_unsat) + sr_unsat

# Retardation factor (according to formula used in Brusseau 2021)
R = 1 + Kd * (rho_bulk / (porosity * sw_unsat)) + a_Kaw * (Aaw / (porosity * sw_unsat))

# PFOS recharge concentration per time period [mg/l]

stp = {
       0 : rch_con,
       1 : 0
       }

# Array with the bottom of each model cell
botm = np.linspace(top - delv, z_bottom, nlay)


#%% Input for both saturated and unsaturated
    
###############################################################################
############################ GENERAL INPUT FILES ##############################
###############################################################################

ms = flopy.modflow.Modflow()

dis = flopy.modflow.ModflowDis(ms,nlay,nrow,ncol, delr=delr, delc=delc, laycbd=0, top=top, botm=botm)

# Unstructured grid generation for both simulations

gridgen_ws = os.path.join(model_ws, 'gridgen')
if not os.path.exists(gridgen_ws):
    os.mkdir(gridgen_ws)    
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
g.build()


disu = g.get_disu(mf, itmuni=3, lenuni= 3, nper=nper, perlen=perlen)
disu.ivsd=-1
anglex = g.get_anglex()
disu.iac.fmtin = "(10I4)"
disu.ja.fmtin = "(10I4)"
disu.cl12.fmtin = "(10F6.2)"
disu.fahl.fmtin = "(10F6.2)"

gridprops_ug = g.get_gridprops_unstructuredgrid()
ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug)
mf.modelgrid = ugrid

# Visualizing model grid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
pmv = flopy.plot.PlotMapView(modelgrid=ugrid, ax=ax)
pmv.plot_grid(alpha=0.5)
plt.plot()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
pmv = flopy.plot.PlotCrossSection(modelgrid=ms.modelgrid, ax=ax, line = {"column" : 1})
pmv.plot_grid(alpha=0.5)
plt.plot()


# Solving preferences for both simulations

sms = MfUsgSms(mf, 
        hclose=1.0e-3,
        hiclose=1.0e-5,
        mxiter=250,
        iter1=600,
        iprsms=1,
        nonlinmeth=1,
        linmeth=1,
        theta=0.7,
        akappa=0.07,
        gamma=0.1,
        amomentum=0.0,
        numtrack=200,
        btol=1.1,
        breduc=0.2,
        reslim=1.0,
        iacl=1,
        norder=0,
        level=7,
        north=14,
        iredsys=0,
        rrctol=0.0,
        idroptol=1,
        epsrn=1.0e-3,
        options2=["SOLVEACTIVE","DAMPBOT"],
)

# Output control for both simulations

lrcsc = {(0,0): ["DELTAT 0.0205", "TMINAT 0.1", "TMAXAT 200.0", "TADJAT 1.0", "TCUTAT 2.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"],
         (1,0): ["DELTAT 0.0205", "TMINAT 0.1", "TMAXAT 200.0", "TADJAT 1.0", "TCUTAT 2.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"]}

oc = MfUsgOc(mf, atsa=1, npsteps=1, unitnumber= [14,30,31,0,0,132], stress_period_data = lrcsc,compact=False)

#%% Specific inputs for saturated simulation

###############################################################################
########################### SATURATED SIMULATION ##############################
###############################################################################

bas = ModflowBas(mf,ibound=1, strt=h_start_sat, richards=True, unstructured=True)

# Flow for saturated simulation

ipakcb = 50

lpf = MfUsgLpf(mf,ipakcb = ipakcb, constantcv=1, novfc=1, laytyp=4, 
               hk = hk,vka = vka, 
               alpha = alpha_sat, beta = beta_sat, sr = sr_sat, brook = brook_sat)


# Saturated transport conditions

bct = MfUsgBct(mf, itvd = 9, cinact=-999.9, diffnc= diffnc, prsity = porosity, timeweight=1.0,
               anglex=anglex, dl = dl, dt = dt,
              iadsorb=iadsorb, flich = flich, bulkd=rho_bulk, adsorb=0)


# Recharge for saturated simulation

rch = MfUsgRch(mf,ipakcb=ipakcb, iconc=1, rech=q_column, rchconc=stp)

# a prescribed head boundary condition at the bottom that can control the degree of saturation of the soil column. For the saturated case, the prescribed head condition was above the top of the soil column at 20 cm. For the case of Sw = 0.68, The bottom head was set to -1.8 cm with van Genuchten parameters  = 12.6 cm-1,= 1.16, Sr = 0.22, and the Brooks Corey exponent = 4. The steady-state flow-fields thus generated were used for the transport simulations. 

# Saturated flow conditions

dtype = np.dtype([
    ("node", int),
    ("shead", np.float32),
    ("ehead", np.float32),
    ("c01", np.float32)])

bottom_cells = range(nlay * 4 - 4, nlay * 4)

chead = h_bottom_sat
lrcsc = {0:[[bottom_cells[0],chead,chead,0.0],
            [bottom_cells[1],chead,chead,0.0],
            [bottom_cells[2],chead,chead,0.0],
            [bottom_cells[3],chead,chead,0.0]]}

chd = ModflowChd(mf,ipakcb = ipakcb, options=[], dtype=dtype, stress_period_data=lrcsc)

#%% Write input to file and run model

mf.write_input()
success, buff = mf.run_model()

# Extracting the head and concentration data for the saturated flow run

concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc1 = concobj.get_ts((119))

hdsobj = HeadUFile(f"{mf.model_ws}/{mf.name}.hds")
head1 = np.array(hdsobj.get_data())
#times = hdsobj.get_times()


#%% Input specific for unsaturated simulation

###############################################################################
########################## UNSATURATED SIMULATION #############################
###############################################################################

bas = ModflowBas(mf,ibound=1, strt=h_start_unsat, richards=True, unstructured=True)


# Unsaturated transport condtions

mf.remove_package("BCT")

bct = MfUsgBct(mf, itvd = 9, cinact=-999.9, diffnc= diffnc, prsity = porosity, timeweight=1.0,
               anglex=anglex, dl =dl, dt=dl,
               iadsorb=iadsorb, flich = flich, bulkd=rho_bulk, adsorb=Kd,
               aw_adsorb=1, iarea_fn=1, ikawi_fn=ikawi_fn, awamax=Aaw, alangaw=a_Kaw, blangaw=b_Kaw,
              )

# Constand head for unsaturated simulation

mf.remove_package("CHD")
bottom_cells = range(nlay * 4 - 4, nlay * 4)

chead=h_bottom_unsat

lrcsc = {0:[[bottom_cells[0],chead,chead,0.0],
            [bottom_cells[1],chead,chead,0.0],
            [bottom_cells[2],chead,chead,0.0],
            [bottom_cells[3],chead,chead,0.0]],
         }

chd = ModflowChd(mf,ipakcb = ipakcb, options=[], dtype=dtype, stress_period_data=lrcsc)

# Recharge for unsaturated simulation
    
rch = MfUsgRch(mf,ipakcb=ipakcb,iconc=1, rech=q_column, rchconc=stp)

mf.remove_package("LPF")
ipakcb = 50
hk  = hk
vka = vka
lpf = MfUsgLpf(mf,ipakcb = ipakcb, constantcv=1, novfc=1, laytyp=5, 
               hk = hk,vka = vka, 
               alpha = alpha_unsat, beta = beta_unsat, sr = sr_unsat, brook = brook_unsat)

#%% Write and run unsaturated simulation

mf.write_input()
success, buff = mf.run_model()

# Extracting the head and concentration data for the unsaturated flow run

concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc2 = concobj.get_ts((119))

hdsobj = HeadUFile(f"{mf.model_ws}/{mf.name}.hds")
head2 = np.array(hdsobj.get_data())

#%%

###############################################################################
############################## VISUALIZATION ##################################
###############################################################################

# Convert time to pore volume
simconc1_pf, simconc2_pf = simconc1, simconc2
simconc1_pf[:,0] = simconc1[:,0] / hours_to_pv
simconc2_pf[:,0] = simconc2[:,0] / hours_to_pv / 0.69

# Convert concentration to relative concentration
simconc1_pf[:,1] = simconc1[:,1] / rch_con
simconc2_pf[:,1] = simconc2[:,1] / rch_con

# Plot PFOS concentrations
fig = plt.figure(figsize=(8, 5), dpi=150)
ax = fig.add_subplot(111)

ax.plot(simconc1_pf[:,0], simconc1_pf[:,1], color = "darkred", label="saturated")
ax.plot(simconc2_pf[:,0], simconc2_pf[:,1], color = "darkorange", label="unsaturated")

ax.set_xlim((0,50))
ax.set_xlabel("Pore Volumes")
ax.set_ylabel("Normalized concentration")
ax.set_title("Adsorption of PFAS Adsorption on Air-Water Interface in the Unsaturated Zone")
ax.legend()
plt.grid(True)
plt.show()

# Calculate and plot water saturation

hc_unsat = botm - head2[:,0] # Capillary head for unsaturated simulation [cm]
gamma_unsat = 1 - (1 / beta_unsat) # van Genuchten gamma parameter for unsaturated simulation
# Equation as per information document for USGT by Panday, S. page 87
se_unsat_array = (1 + alpha_unsat * (abs(hc_unsat - hb))**beta_unsat)**-gamma_unsat # Effective soil saturation for unsaturated simulation
sw_unsat_array = se_unsat_array * (1 - sr_unsat) + sr_unsat # Soil water saturation for unsaturated simulation

fig = plt.figure(figsize=(4, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(sw_unsat_array, botm, color = "blue")
ax.set_xlabel("Water saturation")
ax.set_ylabel("Elevation [cm]")
plt.show()
# Define output directory and filename
output_npz = r"C:\Users\6346650\OneDrive - Universiteit Utrecht\modelling\PFAS-LEACH-Screening-Python-04122022\results\data_GSI-USGT_Brusseau_PFOS_01.npz"
 
# Ensure the directory exists
os.makedirs(os.path.dirname(output_npz), exist_ok=True)

# # Create data dictionary
data = {
    'T_d': simconc2_pf[:,0] ,  #dimensionless time
    'C_relative': simconc2_pf[:, 1] #Extract concentration values (assuming second column contains concentration)
 }

# Save to .npz file
np.savez(output_npz, **data)
