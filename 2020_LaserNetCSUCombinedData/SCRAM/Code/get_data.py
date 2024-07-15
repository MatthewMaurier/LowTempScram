import glob
import pandas as pd
import tables
import numpy as np
import scipy as sp

#Function to import experimental spectra and returns interpolators for VH and HR spectra and error
def get_intrp(exp_path, shot_nums):

    # Change to directory of data and import all files (Pick folder for a single date e.g. all Dec 14 shots)
    shot_list = sorted(glob.glob(exp_path+"/*"))

    #Extract spectra from shot_nums with "best focus" and laser conditions 
    spec_VH,spec_HR = [],[]
    shot = []
    print('Current Shots: \n')
    for i in shot_nums:
        shot = pd.read_hdf(shot_list[i])
        spec_VH.append(shot["Spectrum_VonHamos"]*1000)
        spec_HR.append(shot["Spectrum_Kalpha"]*1000/2)
        # spec_Kbeta = shot["Spectrum_Kbeta"]*1000/2 #was not used?
        print('Number: {x}\tTarget: {y}\t\tFocus: {z}\n'.format(x=shot['Shot_Number']\
        , y=shot['Target_Type'], z = shot['Focus']))

    #Obtain energy axes, error in intensity, and average intensity
    enrg_ax_VH, enrg_ax_HR = shot["enAxis_VonHamos"], shot["enAxis_Kalpha"]
    VH_err, HR_err = np.std(spec_VH, axis=0), np.std(spec_HR, axis=0)
    spec_VH, spec_HR = np.average(spec_VH, axis = 0), np.average(spec_HR, axis = 0)

    #interpolate spectra and errors
    spec_VH, spec_HR = sp.interpolate.interp1d(enrg_ax_VH,spec_VH), sp.interpolate.interp1d(enrg_ax_HR,spec_HR)
    VH_err, HR_err = sp.interpolate.interp1d(enrg_ax_VH, VH_err), sp.interpolate.interp1d(enrg_ax_HR, HR_err)

    return spec_VH, spec_HR, VH_err, HR_err