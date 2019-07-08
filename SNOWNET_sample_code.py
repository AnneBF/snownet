 # -*- coding: utf-8 -*-
"""
Updated version including uncertainty estimate
June 2019

This code demonstartes how to use the AMSR2-only ('snownet.h5') 
and the AMSR2+SMOS ('snownet_smos.h5') neural networks for snow depth estimation 
including an uncertainty estimation for each predicted value from a Monte Carlo 
simulation.

As input the brightness temperatures at 6.9V, 18.7V and 18.7H, 36.5V and 36.5H
are required (18.7H is only needed for the Nasa Team algorithm to do an open water 
correction). The AMSR2+SMOS neural network additionally relies on brightness 
temperatures at 1.4V and 1.4H at 55deg incidence angle. All the input brightness 
temperatures need to be collocated.

The neural network is loaded with model=load_model('snownet.h5')
and snow depth is calculated with snow_depth = model.predict(inputs).
inputs has to be a samples x 3 matrix for the AMSR2-only and a samples x 4 matrix
for the AMSR2+SMOS neural network. The first column is always the 37/19V gradient 
ratio, the second column has to be the 19/7V gradient ratio, the third one the 
polarization ratio at 37GHz. If the AMSR2+SMOS neural network is used the fourth
column stores the gradient ratio at 1.4GHz.

The snow depth output is always in meters.

For the uncertainty estimate a standard deviation can be provided for each input
brightness temperature or it can be set to a constant value like 0.5K. Additional 
uncertainties can be set for the tie points and sea ice concentration (SIC).
These uncertainties are then used to generate N=50 samples for a Monte Carlo 
Simulation to estimate the uncertainty [m] of each snow depth estimate.

To run the code you need a few packages installed including Tensorflow and Keras.
To make sure the model can be loaded and to reproduce the same results as presented 
in the paper, please install Tensorflow version 1.10.0 and Keras version 2.2.2.

When using the neural networks or this code, please cite:
'A. Braakmann-Folgmann and C. Donlon: Estimating Snow Depth on Arctic Sea Ice 
using Satellite Microwave Radiometry and a Neural Network, The Cryosphere, (2019)'

@author: Anne Braakmann-Folgmann

"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
# =============================================================================
# you should install Tensorflow 1.10.0 and Keras 2.2.2 to reproduce the same results
# =============================================================================
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization


#==============================================================================
# generate AMSR2-like sample data [K]
#==============================================================================
Tb6V=np.array([248.3, 245.44, 247.02, 245.02, 245.97, 245.02, 246.62, 245.86, 245.64, 246.64])
Tb18H=np.array([209.45, 206.67, 208.47, 205.19, 206.92, 205.19, 208.86, 207.19, 206.61, 208.78])
Tb18V=np.array([232.27, 230.65, 232.52, 229.04, 232.64, 229.04, 229.73, 231.26, 229.06, 229.53])
Tb36H=np.array([191.38,192.49, 192.81, 189.02, 193.87, 189.02, 191.59, 192.86, 189.5, 191.2])
Tb36V=np.array([207.07, 209.16, 209.51, 205.56, 212.2, 205.56, 206.3, 209.96, 205.22, 205.84])


# =============================================================================
# generate SMOS-like sample data [K]
# =============================================================================
Tb1H=np.array([231.77, 233.74, 234.88, 228.06, 229.03, 222.89, 222.84, 223.26, 219.59, 218.55])
Tb1V=np.array([261.13, 256.50, 257.23, 255.97, 254.17, 252.88, 252.88, 254.19, 251.27, 252.06])


# =============================================================================
# Calculate the gradient and polarization ratios
# =============================================================================

# SIC (can be calculated from the Nasa Team algorithm)
SIC=np.array([0.93307081, 0.90422626, 0.91017262, 0.8984863, 0.88024428, 0.9106783, 
     0.95138806,  0.90307585, 0.930586, 0.95366877])

# Open water tie points from Ivanova et al. (2014)
OW_6V=161.35
OW_18H=108.46
OW_18V=183.72
OW_36H=145.29
OW_36V=209.81

# calculate brightness Temperature of ice part within footprint
Tbice6V= np.divide((Tb6V - np.multiply((1-SIC), OW_6V) ), SIC)
Tbice18V= np.divide((Tb18V - np.multiply((1-SIC), OW_18V) ), SIC)
Tbice36H= np.divide((Tb36H - np.multiply((1-SIC), OW_36H) ), SIC)
Tbice36V= np.divide((Tb36V - np.multiply((1-SIC), OW_36V) ), SIC)

# gradient ratios
GR3618V = np.divide((Tbice36V - Tbice18V), (Tbice36V + Tbice18V))
GR186V = np.divide((Tbice18V- Tbice6V), (Tbice18V + Tbice6V))

# polarisation ratios
PR36=np.divide((Tbice36V-Tbice36H),(Tbice36V+Tbice36H)) 
PR1=np.divide((Tb1V-Tb1H),(Tb1V+Tb1H)) 

#==============================================================================
# prepare the input data 
#==============================================================================
input_data=np.concatenate((np.expand_dims(GR3618V, axis=1), np.expand_dims(GR186V, axis=1), np.expand_dims(PR36, axis=1)), axis=1)

input_data_smos=np.concatenate((input_data, np.expand_dims(PR1, axis=1)), axis=1)

# =============================================================================
#  load the neural network
# =============================================================================
# try to load the neural networks like this: 
try: 
    model=load_model('snownet.h5')
    model_smos=load_model('snownet_smos.h5')

# if an error occurs, you have a different version of Keras or Tensorflow installed. 
# Try to set up the design yourself and only load the weights
# this may however lead to slightly different results
except:
    print('You have a different version of Tensorflow or Keras installed. This may lead to slightly different results.')
    model = Sequential()
    model.add(Dense(15, input_shape=(3,), activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    model.load_weights('snownet_weights.h5')
    
    model_smos = Sequential()
    model_smos.add(Dense(15, input_shape=(4,), activation='sigmoid'))
    model_smos.add(BatchNormalization())
    model_smos.add(Dense(15, activation='relu'))
    model_smos.add(Dense(15, activation='relu'))
    model_smos.add(Dense(20, activation='relu'))
    model_smos.add(Dense(1, activation='tanh'))
    
    model_smos.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    model_smos.load_weights('snownet_smos_weights.h5')


#==============================================================================
# make predictions
#==============================================================================
snow_depth_AMSR2_only = model.predict(input_data)
snow_depth_AMSR2_SMOS = model_smos.predict(input_data_smos)


# -------- Monte Carlo simulation to estimate the uncertainties --------------#

# =============================================================================
# set the input uncertainties 
# =============================================================================
std_amsr=0.5 # uncertainty of the AMSR2 brightness temperatures [K]
std_smos=0.7 # uncertainty of the SMOS brightness temperatures [K]
std_sic=0.04 # uncertainty of SIC
std_tp =3 # uncertainty of the tie points [K]

# =============================================================================
#  Generate N=50 samples for each input
# =============================================================================
N=50

# AMSR2
err=np.random.randn(N, np.size(Tb6V))*std_amsr
Tb6V_ = Tb6V+err
err=np.random.randn(N, np.size(Tb6V))*std_amsr
Tb18H_ = Tb18H+err
err=np.random.randn(N, np.size(Tb6V))*std_amsr
Tb18V_ = Tb18V+err
err=np.random.randn(N, np.size(Tb6V))*std_amsr
Tb36H_ = Tb36H+err
err=np.random.randn(N, np.size(Tb6V))*std_amsr
Tb36V_ = Tb36V+err

# SMOS
err=np.random.randn(N, np.size(Tb1H))*std_smos
Tb1H_ = Tb1H+err
err=np.random.randn(N, np.size(Tb1H))*std_smos
Tb1V_ = Tb1V+err

# SIC
err=np.random.randn(N, np.size(SIC))*std_sic
SIC_ = SIC+err

# tie points
err=np.random.randn(N , np.size(Tb6V))*std_tp
OW_6V_=OW_6V+err
err=np.random.randn(N , np.size(Tb6V))*std_tp
OW_18H_=OW_18H+err
err=np.random.randn(N , np.size(Tb6V))*std_tp
OW_18V_=OW_18V+err
err=np.random.randn(N , np.size(Tb6V))*std_tp
OW_36H_=OW_36H+err
err=np.random.randn(N , np.size(Tb6V))*std_tp
OW_36V_=OW_36V+err

# =============================================================================
# do the calculations for each sample
# =============================================================================
# calculate brightness Temperature of ice part within footprint
Tbice6V_= np.divide((Tb6V_ - np.multiply((1-SIC_), OW_6V_) ), SIC_)
Tbice18V_= np.divide((Tb18V_ - np.multiply((1-SIC_), OW_18V_) ), SIC_)
Tbice36H_= np.divide((Tb36H_ - np.multiply((1-SIC_), OW_36H_) ), SIC_)
Tbice36V_= np.divide((Tb36V_ - np.multiply((1-SIC_), OW_36V_) ), SIC_)

# gradient ratios
GR3618V_ = np.divide((Tbice36V_ - Tbice18V_), (Tbice36V_ + Tbice18V_))
GR186V_ = np.divide((Tbice18V_- Tbice6V_), (Tbice18V_ + Tbice6V_))

# polarisation ratios
PR36_=np.divide((Tbice36V_-Tbice36H_),(Tbice36V_+Tbice36H_)) 
PR1_=np.divide((Tb1V_-Tb1H_),(Tb1V_+Tb1H_)) 

input_data_=np.concatenate((np.expand_dims(GR3618V_.flatten(), axis=1), np.expand_dims(GR186V_.flatten(), axis=1), np.expand_dims(PR36_.flatten(), axis=1)), axis=1)
input_data_smos_=np.concatenate((input_data_, np.expand_dims(PR1_.flatten(), axis=1)), axis=1)

# prediction for each sample
snow_depth_AMSR2_only_ = model.predict(input_data_)
snow_depth_AMSR2_SMOS_ = model_smos.predict(input_data_smos_)

snow_depth_AMSR2_only_=np.reshape(snow_depth_AMSR2_only_, (np.size(GR186V_, 0), np.size(GR186V_, 1)))
snow_depth_AMSR2_SMOS_=np.reshape(snow_depth_AMSR2_SMOS_, (np.size(GR186V_, 0), np.size(GR186V_, 1)))

# =============================================================================
#  Uncertainty of each snow depth prediction
# =============================================================================
std_snow_depth_AMSR2_only=np.nanstd(snow_depth_AMSR2_only_, axis=0)
std_snow_depth_AMSR2_smos=np.nanstd(snow_depth_AMSR2_SMOS_, axis=0)