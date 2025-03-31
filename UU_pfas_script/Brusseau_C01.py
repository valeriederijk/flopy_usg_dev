#!/usr/bin/env python
# coding: utf-8

# ## MF-USG Example Problems for Adsorption of PFAS Adsorption on Air-Water Interface in the Unsaturated Zone 

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other Enhancements to MODFLOW-USG, GSI Environmental, July 2024 http://www.gsi-net.com/en/software/free-software/USG-Transport.html
# 
# This example problem discusses one dimensional advective transport of PFAS in the vadose zone. The experiments conducted by Lyu (2018) are simulated here to evaluate the accuracy of the model in their validation. These experiments were also simulated by Silva et al. (2020) using a modified Hydrus model. 
# 
# Once the flow field was stabilized, PFOA solution was injected with the recharge water. The experiment was conducted with different soil water saturations, and different injection concentrations of PFOA. The simulations conducted here are for saturated conditions, and for a water saturation (Sw) of 0.68, with PFOA concentrations of 1, 0.1, and 0.01 mg/L. 
# 
# For the saturated case, there is no air-water interface, so the only adsorption that occurs is on the soil. Lyu (2018) provides a linear sorption coefficient kd = 0.08 cm3g-1 for the 0.35 mm sand. Simulations were conducted with and without adsorption on soil to evaluate the retardation that occurs due to soil. For the unsaturated case, adsorption also occurs on the air-water interface and is related to the specific area of the air-water interface (Aawi) as well as an air-water interface partition coefficient (kawi). Aawi is computed as Aawi = Amax (1-Sw) where Amax is given as 216 cm2/cm3 (Lyu, 2018 and Silva et al. 2020). The air-water interface adsorption coefficient was provided as linearized for the three source concentrations by Lyu (2018) as 0.0021, 0.0027, and 0.004 for PFOA concentrations of 1, 0.1, and 0.01 mg/L respectively. Silva et al. (2020) used the thermodynamic relationship with surface tension to estimate the air-water interface adsorption as a function of concentration; however, there is no change in kawi for concentration values less than 1 mg/L and thus they did not see a relationship of PFAS transport with respect to source concentration. Since the Langmuir isotherm is used for kawi in USG-Transport, the air-water adsorption isotherm A and B parameters can be provided as A = 0.0021, 0.0027, and 0.004 for PFOA concentrations of 1, 0.1, and 0.01 mg/L respectively, and B = 0. 
# 
# For the unsaturated case, Silva et al. (2020) noted that the simulated breakthrough curves for the unsaturated cases match the laboratory study better with a dispersion coefficient of 0.7 cm. The same behavior was noted with simulations conducted with USG-Transport. Also, a diffusion coefficient of 0.01944 cm2/hr (5.4 x 10-6 cm2/s) was provided.  
# 
# A comparison of simulated results versus the laboratory observations is shown for the different source concentrations in Figure AW1. The simulated results are superposed on the results of Figure 4 of Lyu (2018). 
# 
# For all cases, it is noted that the simulations compare very well with the experimental results. Figure AW2 shows a comparison of simulated results versus the laboratory observations for the different saturation conditions. Note that the case of saturation = 0.86 was not simulated, however, the simulation cases for both with and without dispersion are showin for the unsaturated case. Again, it is noted that the simulations compare very well with the experimental results. The simulation case without soil adsorption is also shown on Figure AW2, to note the impact of the various adsorption mechanisms on the movement of PFOA. Retardation due to soil adsorption slows down the breakthrough of PFOA by less than half a pore volume, however, adsorption on the air-water interface slows it down further by more than one pore volume. 

# In[ ]:


import os, shutil
import numpy as np
import matplotlib.pyplot as plt

import flopy
from flopy.modflow import ModflowBas, ModflowChd,ModflowDis
from flopy.mfusg import (MfUsg, MfUsgDisU, MfUsgLpf, MfUsgSms,
MfUsgBct, MfUsgRch, MfUsgOc)
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen
from flopy.plot import PlotCrossSection,PlotMapView

import flopy.utils.binaryfile as bf

from tempfile import TemporaryDirectory


# In[ ]:


model_ws = "Brusseau_C01"

# temp_dir = TemporaryDirectory()
# model_ws = temp_dir.name


# In[ ]:


mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws= model_ws,
    modelname="Brusseau",
    exe_name="mfusg_gsi",
)


# The setup simulated here consists of a 1-dimensional vertical soil column of 0.35 mm sand, 15 cm long, with a steady-state recharge from the top that gives a pore velocity of 37 cm/hr. The soil column was discretized uniformly into 30 numerical layers of 0.5 cm thickness each with a bottom elevation of zero and a top elevation of 15 cm.

# In[ ]:


ms = flopy.modflow.Modflow()

nrow = 2
ncol = 2
delc = 0.5
delr = 0.5

nlay = 30
top = 15
delv = 0.5
botm = np.linspace(top - delv, 0.0, nlay)

dis = flopy.modflow.ModflowDis(ms,nlay,nrow,ncol, delr=delr, delc=delc, laycbd=0, top=top, botm=botm)


# In[ ]:


gridgen_ws = os.path.join(model_ws, 'gridgen')
if not os.path.exists(gridgen_ws):
    os.mkdir(gridgen_ws)    
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
g.build()


# In[ ]:

#time discretization; divided up into two periods. see mfusg docs for itmuni and lenuni defs
disu = g.get_disu(mf, itmuni=3, lenuni= 3, nper=2, perlen=[9,40.025])
disu.ivsd=-1
anglex = g.get_anglex()
disu.iac.fmtin = "(10I4)"
disu.ja.fmtin = "(10I4)"
disu.cl12.fmtin = "(10F6.2)"
disu.fahl.fmtin = "(10F6.2)"


# In[ ]:


# MODFLOW-USG does not have vertices, so we need to create
# and unstructured grid and then assign it to the model. This
# will allow plotting and other features to work properly.
gridprops_ug = g.get_gridprops_unstructuredgrid()
ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug)
mf.modelgrid = ugrid


# In[ ]:


bas = ModflowBas(mf,ibound=1,strt=-3,richards=True,unstructured=True)


# In[ ]:
# van Genuchten parameters
alpha = 0.08 # 0.08  # cm^-1
n = 3.2 #3.2
theta_r = 0.078 # 0.078
theta_s = 0.33
m = 1 - 1/n

ipakcb = 50
hk  = 2.2e-2*3600
vka = 2.2e-2*3600
lpf = MfUsgLpf(mf,ipakcb = ipakcb, constantcv=1, novfc=1, laytyp=4, 
               hk = hk,vka = vka, 
               alpha = alpha, beta = n, sr = theta_r)
# In[ ]:


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



# In[ ]:
porosity = 0.33
bct = MfUsgBct(mf, itvd = 9, cinact=-999.9, diffnc= 0.01944, prsity = porosity, timeweight=1.0,
               anglex=anglex, dl =0.7, dt=0.00,
               iadsorb = 3, flich = 0.81, bulkd=1.5, adsorb=0.15,
               aw_adsorb=3, iarea_fn=1, ikawi_fn= 1, awamax=350.0, alangaw=0.027, blangaw= 0,
              )


# Recharge was simulated from the top end at a rate of 12.21 cm/hr. For a porosity value of 0.33, the recharge rate is 12.21 cm/hr, and one pore volume (PV) is equal to 0.405 hours. 

# In[ ]:
recharge_val = 6.5 # roughly 10 *0.68
input_conc = 0.1

rch = MfUsgRch(
    mf,
    ipakcb=ipakcb,
    iconc=1,
    rech={0: recharge_val, 1: recharge_val},   # Recharge values for stress periods 0 and 1
    rchconc={0: input_conc, 1: 0.00}    # Recharge concentration for stress periods 0 and 1
)


# a prescribed head boundary condition at the bottom that can control the degree of saturation of the soil column. For the saturated case, the prescribed head condition was above the top of the soil column at 20 cm. For the case of Sw = 0.68, The bottom head was set to -1.8 cm with van Genuchten parameters  = 12.6 cm-1,= 1.16, Sr = 0.22, and the Brooks Corey exponent = 4. The steady-state flow-fields thus generated were used for the transport simulations. 

# In[ ]:


dtype = np.dtype([
    ("node", int),
    ("shead", np.float32),
    ("ehead", np.float32),
    ("c01", np.float32)])

chead= -3
lrcsc = {0:[[116,chead,chead,0.0],
           [117,chead,chead,0.0],
           [118,chead,chead,0.0],
           [119,chead,chead,0.0]],
          1:[[116,chead,chead,0.0],
            [117,chead,chead,0.0],
            [118,chead,chead,0.0],
            [119,chead,chead,0.0]]}
chd = ModflowChd(mf,ipakcb = ipakcb, options=[], dtype=dtype, stress_period_data=lrcsc)


# In[ ]:


lrcsc = {(0,0): ["DELTAT 0.0205", "TMINAT 0.1", "TMAXAT 200.0", "TADJAT 1.0", "TCUTAT 2.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"], 
         (1,0): ["DELTAT 0.0205", "TMINAT 0.1", "TMAXAT 200.0", "TADJAT 1.0", "TCUTAT 2.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"] }

oc = MfUsgOc(mf, atsa=1, npsteps=1, unitnumber= [14,30,31,0,0,132], stress_period_data = lrcsc,compact=False)


# In[ ]:


mf.write_input()
success, buff = mf.run_model()


# In[ ]:


concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc1 = concobj.get_ts((119))
headobj = HeadUFile(f"{mf.model_ws}/{mf.name}.hds")
times = headobj.get_times()
datahead = headobj.get_data()
print(datahead)

values = [arr[0] for arr in datahead]

# Create the discretization index
indices = np.arange(len(datahead))

# Plot
plt.figure(figsize=(6, 8))
plt.plot(values, indices, marker='o', linestyle='-')
plt.xlabel("Head 'cm'")
plt.ylabel("Discretization Index")
plt.title("Head")
plt.gca().invert_yaxis()
plt.grid(True)

plt.show()

heads = headobj.get_alldata()  # Extract all heads
selected_column = 2  # Change this based on your grid setup
head_profile = heads[-1, :, selected_column]
elevations = np.linspace(15, 0, heads.shape[1])  # Adjust based on discretization

# Convert head to pressure head
pressure_head = head_profile - elevations  # Element-wise operation


# Compute water saturation using van Genuchten equation
saturation = theta_r + (theta_s - theta_r) * (1 / ((1 + (alpha * np.abs(pressure_head))**n)**m))

plt.figure(figsize=(6, 8))
plt.plot(saturation, elevations, marker='o', linestyle='-')
plt.xlabel("Saturation")
plt.ylabel("Depth (cm)")
plt.title("Soil Water Saturation Profile")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


#%%

#calc pore volume
saturation = 0.68 # or should this be theta (e.g. 0.24)
theta = 0.24
velocity = abs(recharge_val/(porosity*saturation) ) # Pore water velocity cm/h
print(velocity)
PV = velocity/top  # Dimensionless time 
print(PV)

fig = plt.figure(figsize=(8, 5), dpi=150)
ax = fig.add_subplot(111)
ax.plot(simconc1[:,0]*PV, simconc1[:,1]/input_conc , label="S=0.68, Kd=0.15")
ax.set_xlabel("Pore Volumes")
ax.set_ylabel("Normalized concentration")
ax.set_title("Adsorption of PFAS Adsorption on Air-Water Interface in the Unsaturated Zone")
ax.legend()
plt.show()


#Define experiment parameters
name_exp = "Brusseau"
conc = "01"
pfas_name = "PFOS"

# Define output directory and filename
output_npz = r"C:\Users\6346650\OneDrive - Universiteit Utrecht\modelling\PFAS-LEACH-Screening-Python-04122022\results\data_GSI-USGT_Brusseau_PFOS_01.npz"
 
# Ensure the directory exists
os.makedirs(os.path.dirname(output_npz), exist_ok=True)

# # Create data dictionary
data = {
    'T_d': simconc1[:,0]*PV ,  #dimensionless time
    'C_relative': simconc1[:, 1]/input_conc #Extract concentration values (assuming second column contains concentration)
 }

# Save to .npz file
np.savez(output_npz, **data)
print(f"Data saved to {output_npz}")
