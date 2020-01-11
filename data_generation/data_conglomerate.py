import numpy as np
import os

snrs = [90]
save_file = "dat_snr"

for snr in snrs:

    fi_names = os.listdir("data_snrs/"+save_file+str(snr))

    all_data = [np.load("data_snrs/"+save_file+str(snr)+"/"+str(name)) for name in fi_names[2:]]

    d1 = np.load("data_snrs/"+save_file+str(snr)+"/"+fi_names[0])
    d2 = np.load("data_snrs/"+save_file+str(snr)+"/"+fi_names[1])

    data = np.concatenate((d1,d2),axis=0)

    for dat in all_data:
        data = np.concatenate((data,dat),axis=0)
        print("{} of snr {}".format(data.shape,snr))

    np.save("train_data/data_snrs{}.npy".format(snr),data)
