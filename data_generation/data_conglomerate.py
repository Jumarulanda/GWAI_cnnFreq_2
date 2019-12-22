import numpy as np
import os

fi_names = os.listdir("noi_data/")

all_data = [np.load("noi_data/"+name) for name in fi_names[2:]]

d1 = np.load("noi_data/"+fi_names[0])
d2 = np.load("noi_data/"+fi_names[1])

data = np.concatenate((d1,d2),axis=0)

for dat in all_data:
    data = np.concatenate((data,dat),axis=0)
    print(data.shape)

np.save("train_data/tdata.npy",data)
