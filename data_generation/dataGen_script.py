''' This script to generate gravitational wave signals using pycbc's numeric relativity algorithms '''

# Imports
import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import pycbc.psd
import pycbc.noise
import os

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


### Color noise signal.

def CNS(SNR,signal):
	'''This function was modified in order to resize the signal instead of the noise
	to achieve the desired SNR

        INPUT: SNR: desired signal to noise ratio of the new signal
               signal: clean signal to which the noise will be added'''

	psd_name = "ZERO_DET_high_P.txt"

	delta_f = 1.0/16
	flen = int(2048/delta_f)+1
	flow = 10
	psd = pycbc.psd.from_txt(psd_name,flen,delta_f,flow,is_asd_file=False)

	# generate 1sec of data

	delta_t = 1.0/4096
	t_samples = int(4096 * len(signal) * delta_t) ##################
	psd_noise = pycbc.noise.noise_from_psd(t_samples,delta_t,psd)#,seed=666)

	# re-scale noise to match SNR, then add it to the signal

	signal_MP = np.mean(np.array(signal)**2)
	noise_MP = np.mean(np.array(psd_noise)**2)

	# target_MP = signal_MP/SNR
	target_MP = noise_MP * SNR

	new_signal = signal * np.sqrt(target_MP / signal_MP)

	'''the following code re-scales the strain (noise+signal) so its mean power MP matches that
	of the noise alone'''

	new_strain = psd_noise + new_signal
	new_strain_MP = np.mean(np.array(new_strain)**2)

	new_strain = new_strain * np.sqrt(noise_MP / new_strain_MP)

        res_signal_MP = np.mean(np.array(new_signal)**2)
        res_signal = new_signal * np.sqrt(noise_MP / res_signal_MP) # rescaled signal

	return new_strain, res_signal


def genCS(file_n, snr):

    clean_data = np.load("data/"+file_n)
    
    strain, masses = clean_data[0,:-2], clean_data[0,-2:]

    # random shift of 0.2 seconds at the end of the signal
    
    sec = 4096
    d_sec = sec*0.2

    random_shift = int(np.random.uniform(low=0.1,high=1)*d_sec)

    strain_s = (strain[:-random_shift])

    print(strain_s.shape)

    color_signal = CNS(snr, strain)[0].numpy().reshape(1,-1)[:,-sec:]
    color_signal_s = CNS(snr, strain_s)[0].numpy().reshape(1,-1)[:,-sec:]

    color_signal = np.concatenate((color_signal,masses.reshape(1,-1)), axis=1)
    color_signal_s = np.concatenate((color_signal_s,masses.reshape(1,-1)), axis=1)

    t_signal = np.concatenate((color_signal,color_signal_s), axis=0)
    
    np.save("noi_data/cSignals_snr{}_masses{}_{}.npy".format(snr,masses[0],masses[1]), t_signal)

    print("Created colored signals of snr {} and masses {} and {}".format(snr,masses[0],masses[1]))

if __name__ == "__main__":

    # # data generation

    # masses_1 = np.arange(10,76)
    # masses_2 = np.array([mass+0.5 for mass in masses_1])

    # for mass1 in masses_1:
    #     for mass2 in masses_1:
    #         if mass1 > mass2 and mass1/mass2 < 5:
    #             pols = gen_pol((mass1,mass2),apx,delta_t,f_lower)
    #             signal = p_wave(pols,det_h1,end_time=end_time, declination=declination, right_ascension=right_ascension, polarization=polarization)

    #             pre_tSignal = np.array(signal).reshape(1,-1)
    #             mass_array = np.array([mass1,mass2]).reshape(1,-1)
                
    #             total_signal = np.concatenate((pre_tSignal,mass_array), axis=1)
                
    #             np.save("data/clean_data_m{}_{}.npy".format(mass1,mass2), total_signal)

    #             print("Signal of merger signals {} {} creted".format(mass1,mass2))
            
    # for mass1 in masses_2:
    #     for mass2 in masses_2:
    #         if mass1 > mass2 and mass1/mass2 < 5:
    #             pols = gen_pol((mass1,mass2),apx,delta_t,f_lower)
    #             signal = p_wave(pols,det_h1,end_time=end_time, declination=declination, right_ascension=right_ascension, polarization=polarization)

    #             pre_tSignal = np.array(signal).reshape(1,-1)
    #             mass_array = np.array([mass1,mass2]).reshape(1,-1)
                
    #             total_signal = np.concatenate((pre_tSignal,mass_array), axis=1)
                
    #             np.save("data/clean_data_m{}_{}.npy".format(mass1,mass2), total_signal)

    #             print("Signal of merger signals {} {} creted".format(mass1,mass2))
                

    file_names = os.listdir("data/")
    snr = 0.1
    
    for file_name in file_names:
        genCS(file_name, snr)
