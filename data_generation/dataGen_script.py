''' This script to generate gravitational wave signals using pycbc's numeric relativity algorithms '''

# Imports
import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

### Parameters of generated data

apx = 'SEOBNRv4' # waveform approximant
delta_t = 1/4096.
f_lower = 10

det_h1 = Detector("H1") # GW detector

# Wave projection parameters

end_time = 0 #1192529720 # GPS time -- What is this????
declination = 0 
right_ascension = 0
polarization = 0

### Generating data functions

def gen_pol(ms, apx, d_t, f_low):
    ''' Generates the plus and cross polarizations of a gravitational wave

        INPUT:    ms: two-dimentional tuple. Masses of the merger
                  apx: Waveform approximant
                  d_t: sampling rate of the signal
                  f_low: starting frequency of the signal
    '''

    hp, hc = get_td_waveform(mass1=ms[0], mass2=ms[1], approximant=apx, delta_t=d_t, f_lower=f_low)

    return hp, hc

def p_wave(pol, det,**params):
    ''' Projects the wave's polarization to form the signal strain
       
    INPUT:    pol: two-dimentional tuple. Polarizations of the wave
              det: gravitational wave detector where the signal is simulated
              **params: params for the  projection method.
    '''

    pol[0].start_time += params["end_time"]
    pol[1].start_time += params["end_time"]

    signal_h1 = det.project_wave(pol[0], pol[1], params["right_ascension"], params["declination"], params["polarization"])

    return signal_h1


if __name__ == "__main__":

    # data generation

    masses_1 = np.arange(10,76)
    masses_2 = np.array([mass+0.5 for mass in masses_1])

    for mass1 in masses_1:
        for mass2 in masses_1:
            if mass1 > mass2 and mass1/mass2 < 5:
                pols = gen_pol((mass1,mass2),apx,delta_t,f_lower)
                signal = p_wave(pols,det_h1,end_time=end_time, declination=declination, right_ascension=right_ascension, polarization=polarization)

                pre_tSignal = np.array(signal).reshape(1,-1)
                mass_array = np.array([mass1,mass2]).reshape(1,-1)
                
                total_signal = np.concatenate((pre_tSignal,mass_array), axis=1)
                
                np.save("data/clean_data_m{}_{}.npy".format(mass1,mass2), total_signal)

                print("Signal of merger signals {} {} creted".format(mass1,mass2))
            
    for mass1 in masses_2:
        for mass2 in masses_2:
            if mass1 > mass2 and mass1/mass2 < 5:
                pols = gen_pol((mass1,mass2),apx,delta_t,f_lower)
                signal = p_wave(pols,det_h1,end_time=end_time, declination=declination, right_ascension=right_ascension, polarization=polarization)

                pre_tSignal = np.array(signal).reshape(1,-1)
                mass_array = np.array([mass1,mass2]).reshape(1,-1)
                
                total_signal = np.concatenate((pre_tSignal,mass_array), axis=1)
                
                np.save("data/clean_data_m{}_{}.npy".format(mass1,mass2), total_signal)

                print("Signal of merger signals {} {} creted".format(mass1,mass2))
                
    # pols = gen_pol((10,20),apx,delta_t,f_lower)
    # signal = p_wave(pols,det_h1,end_time=end_time, declination=declination, right_ascension=right_ascension, polarization=polarization)

    # import matplotlib.pyplot as plt

    # print(np.array(signal).shape)

    # plt.plot(signal.sample_times, signal)

    # plt.show()
