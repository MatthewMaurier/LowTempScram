import scipy as sp
import numpy as np

'''
Class holding all relevant information from the text-based SCRAM table. A SCRAM 
object has support to change the optical depth by calling the get_scram_intrp(dx)
function where dx is the target thickness given in cm. To see which properties 
are available in a SCRAM object, call SCRAM.attributes
'''
class SCRAM:
    
    def __init__(self, file_path, dx = 10e-4): #automatically initialize with dx = 10um if no arg given
        
        self.table = [line for line in open(file_path,"r")]
        self.meta_data = []
        for i in range(0,15): #metadata held in rows 0-15
            self.meta_data.append(self.table[i].strip().replace("\t"," "))

        self.extract_attributes()
        self.get_spectra()
        self.construct_mat()
        # self.get_scram_intrp(dx)
        self.dx = dx
    
    '''
    This function extracts extra information from the scram table such as kP,
    mdet, and other parameters from the SCRAM table
    '''
    def extract_attributes(self):
        #Initialize SCRAM with attributes from txt file (Not including j and k)
        self.attributes = []
        self.meta_data.append("UNITS:")
        dict_SCRAM = {}
        for row in self.table[15:46]: #18th to 46th row holds non-essential data, 15-17 holds density, temp, energy
          row = row.split("\t")
          dict_SCRAM[row[0]] =  np.asarray(row[1:]).astype(float)

        #Assigning each key in dictionary as an attribute of the class - this enables easy access for later use
        for key in dict_SCRAM:
          self.meta_data.append(key)
          attribute_name = key.split("(")[0].replace("/","_") #Attributes cannot have brackets or slashes, 
          self.attributes.append(attribute_name)              #refer to meta data for units (above line removes them)
          setattr(self,attribute_name,dict_SCRAM[key])
    
    '''
    Function extracts the j, k, and energy values from the text file and stores 
    them in numpy arrays.
    '''
    def get_spectra(self):
        #Extract absorbtion and emissivity
        self.j_table, self.en_j = [], [] 
        for row in self.table[48:2045]:
          row = row.split("\t")
          self.en_j.append(row[0])
          self.j_table.append(row[1:])
        self.j_table, self.en_j = np.asarray(self.j_table).astype(float), np.asarray(self.en_j).astype(float)
        self.en_j = self.en_j/1000 #converting to keV to be consistent with experimental data

        self.k_table,self.en_k = [],[]
        for row in self.table[2047:]:
          row = row.split("\t")
          self.en_k.append(row[0])
          self.k_table.append(row[1:])
        self.k_table,self.en_k = np.asarray(self.k_table).astype(float),np.asarray(self.en_k).astype(float)
        self.en_k = self.en_k/1000
    
    '''
    Function creates a 4-dimensional matrix holding 1997 j/k values for each 
    combination of ne, Te, and tauR (one j/k value for each of the 1997 photon 
    energies).
    '''
    def construct_mat(self):
        dens = np.unique(self.D) #4 density points --> view meta_data for ordering
        Te = np.unique(self.Te) #14 temp points --> view meta_data for ordering
        tauR = np.unique(self.tauR[self.tauR != 0]) # 3 tauR points excluding fluorescense --> [0.1, 1, 1000]
        
        #initialize blank matrices for j/k
        self.j = np.zeros((len(dens), len(Te), len(tauR), len(self.en_j))) 
        self.k = np.zeros((len(dens), len(Te), len(tauR), len(self.en_k))) 
        
        '''
        The following nested loops put each combination of ne, Te, and tauR into
        a 4-dimensional matrix. Hence, the matrix has shape 4x14x3x1997
        '''
        for ix,d in enumerate(dens): 
            for jx,t in enumerate(Te): 
                for kx,tR in enumerate(tauR):                    
                    for mx in range(len(self.D)):                
                        if d == self.D[mx] and t == self.Te[mx] and tR == self.tauR[mx]:
                            self.j[ix][jx][kx] = self.j_table[:,mx]
                            self.k[ix][jx][kx] = self.k_table[:,mx]


        #fluoresence kept in separate matrix from regular emissions
        self.j_fluor = np.zeros((len(dens), len(Te), len(self.en_j)))
        self.k_fluor = np.zeros((len(dens), len(Te), len(self.en_k)))
        self.hot_electron_fraction = np.zeros((len(dens), len(Te))) 
        self.factor_fluor = np.zeros((len(dens), len(Te))) 
        for ix,d in enumerate(dens):
          for jx,t  in enumerate(Te):
              for mx in range(len(self.D)):
                  if d == self.D[mx] and t == self.Te[mx]:
                      self.j_fluor[ix][jx] = self.j_table[:,mx]
                      self.k_fluor[ix][jx] = self.k_table[:,mx]
                      self.hot_electron_fraction[ix][jx] = self.mdet[mx]
                      self.factor_fluor[ix][jx] = self.kPfac1[mx]
                                                                                             
    #Function dynamically sets target thickness and performs tauR interpolation
    def get_scram_intrp(self,dx):
        #Set thickness based on argument
        self.dx = dx
        
        #set up density, temp, and tauR axes for interpolation
        dens = np.unique(self.D)
        Te = np.unique(self.Te)/1000 #converting to keV to match experimental units
        tauR = np.unique(self.tauR[self.tauR != 0]) #excluding fluoresence

        #take log of everything for log-log interp
        l_dens = np.log(dens); l_Te = np.log(Te); l_tauR = np.log(tauR); log_j = np.log(self.j); l_en_j = np.log(self.en_j)
        log_k = np.log(self.k);  log_j_fluor = np.log(self.j_fluor); log_k_fluor = np.log(self.k_fluor)
        l_en_k = np.log(self.en_k)


        #print(l_dens)
        # print(l_Te)
        # print(l_tauR)
        # print(log_j)
        # print(l_en_j)
        # print(log_k)
        # print(log_j_fluor)
        # print(log_k_fluor)
        # print(l_en_k)





        #interpolators for j/k BEFORE finding correct tauR for the given dx
        f_kMatInterp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_tauR, l_en_k),log_k)
        f_jMatInterp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_tauR, l_en_j),log_j) 

        #interpolators for fluoresence (i.e., where tauR = 0 is constant)
        j_fluor_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_j),log_j_fluor) 
        k_fluor_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_k),log_k_fluor) 

        hot_electron_interp =  sp.interpolate.RegularGridInterpolator((dens,Te),self.hot_electron_fraction,bounds_error = False) 
        factor_fluor_interp =  sp.interpolate.RegularGridInterpolator((l_dens,l_Te),np.log(self.factor_fluor)) 

        # FIND PEAKS (Max(k) * dx = tauR)
        kMat = self.k
        tau_error = 0.01
        taus = np.zeros((len(dens), len(Te))) 
        for ix in range(len(dens)):
          for jx in range(len(Te)):
              temp1 = np.max(kMat[ix][jx][0])*dens[ix]*dx # START BY USING TAU = 0.1 ([0] corresponds to tau = 0.1) can also try TAU = 10, 1000, results in same convergence
              temp2 = 0 #temp meaning temporary variable not temperature
              while (np.abs(temp1 - temp2)/temp1 > tau_error): # CHECK IF ERROR > 1%
                  if temp1 < 0.1:
                      temp1 = 0.1 #If tauR<0.1, set it to 0.1 and break the loop 
                      break
                  
                  temp2 = temp1
                  temp1 = f_kMatInterp((l_dens[ix], l_Te[jx], np.log(temp2), l_en_k))
                  temp1 = np.max(np.exp(temp1))*dens[ix]*dx
              
              taus[ix][jx] = temp1

        #Initialize new j/k matrices with the interpolated tauR values
        kfin = np.zeros((len(l_dens), len(l_Te), len(l_en_k)))
        jfin = np.zeros((len(l_dens), len(l_Te), len(l_en_j)))
        for ix in range(len(l_dens)):
          for jx in range(len(l_Te)):
            kfin[ix][jx] = dens[ix]*(np.exp(f_kMatInterp((l_dens[ix], l_Te[jx], np.log(taus[ix][jx]), l_en_k))))
            jfin[ix][jx] = dens[ix]*(np.exp(f_jMatInterp((l_dens[ix], l_Te[jx], np.log(taus[ix][jx]), l_en_j))))
        
        #Return interpolator objects with correct target thickness/optical depth (tauR) 
        k_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_k),np.log(kfin))
        j_interp = sp.interpolate.RegularGridInterpolator((l_dens,l_Te,l_en_j),np.log(jfin))

        return j_interp,k_interp, j_fluor_interp, k_fluor_interp, hot_electron_interp, factor_fluor_interp

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