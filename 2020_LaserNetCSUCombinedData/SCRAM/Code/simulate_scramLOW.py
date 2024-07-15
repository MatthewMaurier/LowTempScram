# from matplotlib.patches import draw_bbox
import numpy as np
import scipy as sp

'''
Class SCRAMTarget converts temperatures and densities into emission spectra

Variables:
tempLayers - the temperatures of a cross section with thickness dx (one layer)
densLayers - the density of a cross section with thickness dx (one layer)
j and k - scipy.interpolator objects that take tuples of the form (ne, Te, enrg)
**note that the interpolators were built using log-log interpolation**

How to use it:
-instantiate an object using the arguments delineated above
-use SCRAMTarget.tempLayers = [array of temps] if you need to set a new temp profile,
similarly, use SCRAMTarget.densLayers to set a new density profile
-call SCRAMTarget.model() to obtain simulated VH/HR spectra
'''

class SCRAMTarget:
    def __init__(self,densLayers,tempLayers,j_f,k_f,fh,fj,en_VH,en_HR,dx=0.5e-4):
        self.densLayers = densLayers
        self.tempLayers = tempLayers
        self.k_f = k_f
        self.j_f = j_f
        self.fh = fh
        self.fj = fj
        self.en_VH = en_VH
        self.en_HR = en_HR
        self.dx = dx
        self.layers = []
        self.generateLayers()

    '''
    Calculating j and k for each layer with density 'dens' and temperature 'temp'
    '''
    def generateLayers(self):
        self.layers = []
        for dens, temp in zip(self.densLayers, self.tempLayers):
            
            #generate front layers using VH energy axis
            k_layer_fluor_f = (np.log(dens),np.log(temp),np.log(self.en_VH))
            k_layer_fluor_f = self.k_f(k_layer_fluor_f)
            k_layer_fluor_f = np.exp(k_layer_fluor_f)
            j_layer_fluor_f = (np.log(dens),np.log(temp),np.log(self.en_VH))
            j_layer_fluor_f = self.j_f(j_layer_fluor_f)
            j_layer_fluor_f = np.exp(j_layer_fluor_f)
            
            
            #generate rear layers using HR energy axis
            k_layer_fluor_r = (np.log(dens),np.log(temp),np.log(self.en_HR))
            k_layer_fluor_r = self.k_f(k_layer_fluor_r)
            k_layer_fluor_r = np.exp(k_layer_fluor_r)
            j_layer_fluor_r = (np.log(dens),np.log(temp),np.log(self.en_HR))
            j_layer_fluor_r = self.j_f(j_layer_fluor_r)
            j_layer_fluor_r = np.exp(j_layer_fluor_r)

            k_layer_f = k_layer_fluor_f
            j_layer_f = j_layer_fluor_f

            k_layer_r = k_layer_fluor_r
            j_layer_r = j_layer_fluor_r
            
            self.layers.append([dens, temp, k_layer_f, j_layer_f,k_layer_r,j_layer_r,j_layer_fluor_f,j_layer_fluor_r])
    
    '''Determines the fraction of hot electrons with energy greater than 80% of the K-shell'''
    def f_greater(self,Th):
        return min(1,2/np.sqrt(np.pi)*np.exp(-0.72*10.285/Th)) 
   
    '''
    Calculates the scaling of the fluorescence emission for a particular hot electron
    fraction and electron temperature
    '''
    def scale_fluor(self,D,Te,hotfrac,Th):
        fh_tab = self.fh((D,Te))
        fj_bin = np.exp(self.fj((np.log(D),np.log(Te))))
        scaling = fj_bin*self.f_greater(Th)*hotfrac/(0.8*fh_tab)
        return scaling 
   
    '''
    Transmission through layer defined by opitcal depth T = exp(-k*dx) where k 
    is the absorbtion coefficient in 1/cm and dx is the plasma depth in cm

    The projection is required if non-normal path lengths are used, such as the
    von Hamos spectra acquired at 45 deg from normal
    '''
    def getTransmission(self,layer,projection,rear):
        #exp(-k_rear*dx) - using indices from self.layers
        if rear: 
          return np.exp(-layer[4]*self.dx*projection)
        #exp(-k_front*dx) - using indices from self.layers
        else:
          return np.exp(-layer[2]*self.dx*projection)

    def transportEmission(self,fh,Th,viewingAngle = 0,rear = False):
        '''
        Performs radiation transport through individual layers to a detector at specified
        viewing angle and on the front or rear side of the target.
        '''
        projection = 1./np.cos(np.radians(viewingAngle))
        intensityTotal = 0
        trans_layers = [self.getTransmission(layer,projection,rear) for layer in self.layers]
        '''
        Self-emission intensity is defined by I = j/k(1-T)
        Total intensity is calculated by starting at either end of the target 
        (if the detector is on the rear side it begins with the front and vice verse)
        and sequentially stepping through individual layers calculating self-emission 
        intensity and transmission

        Stephanie Hansen recommends that we don't want to double count the flourescence contribution
        from a hot plasma region. Therefore, we should ignore the flourescence emission as the K-alpha line
        begins to shift from the cold plasma. Looking at the j emission, significant shifting appears to 
        occur by 500 eV, so I'll cut it there.
        '''
        for i,layer in enumerate(self.layers):
            #tranmission to High Resolution Crystal
            if rear:
                intensity_layer = (layer[5])/layer[4]*(1-trans_layers[i])
                transmission = np.prod(trans_layers[i+1:],axis = 0)

            #transmission to Von Hamos crystal
            else:
                intensity_layer = (layer[3]+layer[6]*self.scale_fluor(layer[0],layer[1],fh,Th)*100)/layer[2]*(1-trans_layers[i])
                transmission = np.prod(trans_layers[:i],axis = 0) 
            intensityTotal += intensity_layer*transmission

        return intensityTotal


    def model(self,d,t,fh,Th):
        #initialize with new temps
        self.densLayers = d
        self.tempLayers = t
        self.generateLayers()
        
        #Assume 5um Spot Size, 1ps Emission Time
        area = np.pi*(2.5e-4)**2
        tau = 1e-12
        scaling = area*tau/(np.pi/4000) #pi/4000 = 1e-3/4*pi converts to mj/keV/sr
        sigma = 8/(2*np.sqrt(2*np.log(2))) 

        #Front Von Hamos Spectra
        front_emission = self.transportEmission(fh,Th,45,rear=False)
        SimulatedVH = sp.ndimage.gaussian_filter1d(front_emission*scaling*1000,sigma)                                            

        #Rear High Res. Spectra
        rear_emission = self.transportEmission(fh,Th,0,rear=True)
        SimulatedHR = sp.ndimage.gaussian_filter1d(rear_emission*scaling*1000,sigma/8)                   

        return SimulatedVH, SimulatedHR

