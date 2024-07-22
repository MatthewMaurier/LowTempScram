import scipy as sp
import numpy as np
import pandas as pd

'''
Class holding all relevant information from the text-based SCRAM table. A SCRAM 
object has support to change the optical depth by calling the get_scram_intrp(dx)
function where dx is the target thickness given in cm. To see which properties 
are available in a SCRAM object, call SCRAM.attributes
'''
class SCRAM:
    
    def __init__(self, file_path, dx = 0.5e-4): #automatically initialize with dx = um if no arg given
        
        self.df = pd.read_csv(file_path,delimiter = '\t')
        self.get_interp()
        self.dx = dx
    
    '''
    This function extracts extra information from the scram table such as kP,
    mdet, and other parameters from the SCRAM table
    '''
    def get_interp(self):
        
        densities = self.df.iloc[[0], :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float)
        densities.columns = range(densities.shape[1])
        temperatures = self.df.iloc[[1], :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float) / 1000
        temperatures.columns = range(temperatures.shape[1])
        mdet = self.df.iloc[[2], :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float)
        mdet.columns = range(mdet.shape[1])
        KPfac1 = self.df.iloc[[21], :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float)
        KPfac1.columns = range(KPfac1.shape[1])

        info = pd.concat([densities,temperatures,mdet,KPfac1],ignore_index=1)
        
        
        j_values = self.df.iloc[23:3356, :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float)
        j_values.columns = range(j_values.shape[1])
        k_values = self.df.iloc[3357:, :].reset_index(drop=True).drop(columns=self.df.columns[0]).astype(float)
        k_values.columns = range(k_values.shape[1])
        denuinque = np.unique(info.iloc[0,:])
        tempunique = np.unique(info.iloc[1,:])
        j_fluor = np.zeros((len(denuinque), len(tempunique), len(j_values))) 
        k_fluor = np.zeros((len(denuinque), len(tempunique), len(k_values)))
        hot_electron_fraction = np.zeros((len(denuinque), len(tempunique))) 
        factor_fluor = np.zeros((len(denuinque), len(tempunique))) 
    
        for ix,d in enumerate(denuinque):
            for jx,t  in enumerate(tempunique):
                for mx in range(len(densities.to_numpy()[0])):
                    if d == densities.to_numpy()[0][mx] and t == temperatures.to_numpy()[0][mx]:
                        j_fluor[ix][jx] = j_values.to_numpy()[:,mx]
                        k_fluor[ix][jx] = k_values.to_numpy()[:,mx]
                        hot_electron_fraction[ix][jx] = mdet.to_numpy()[0][mx]
                        factor_fluor[ix][jx] = KPfac1.to_numpy()[0][mx]

        en_j = self.df.iloc[23:3356, :].reset_index(drop=True).iloc[:,0].astype(float) / 1000
        en_k = self.df.iloc[3357:, :].reset_index(drop=True).iloc[:,0].astype(float) / 1000

        l_dens = np.log(denuinque)
        l_Te = np.log(tempunique)
        l_en_j = np.array(np.log(en_j))
        log_j_fluor = np.log(j_fluor)
        log_k_fluor = np.log(k_fluor)
        l_en_k = np.array(np.log(en_k))


        j_fluor_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_j),log_j_fluor) 
        k_fluor_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_k),log_k_fluor) 
        hot_electron_interp =  sp.interpolate.RegularGridInterpolator((denuinque,tempunique),hot_electron_fraction,bounds_error = False) 
        factor_fluor_interp =  sp.interpolate.RegularGridInterpolator((l_dens,l_Te),np.log(factor_fluor)) 
        return j_fluor_interp, k_fluor_interp, hot_electron_interp,factor_fluor_interp





    def __str__(self):
      outStr = \
      """Temperature, Density, and tauR points:
          Te (eV): {Te}
          ne (g/cc): {ne}
          tauR: {tau}
          Units for j and k:
          j(W/g/eV)
          k(cm2/g)
          
          Current Target Thickness:
          {dx} cm
          """\
          .format(Te=np.unique(self.Te),\
      ne = np.unique(self.D),tau=np.unique(self.tauR),dx = self.dx)

      return outStr