import numpy as np
from scipy.signal import spectrogram
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class base_model():
    '''Model is just CNN regressor for the moment'''
    def __init__(self, name):
        self.name = name
        self.path = "save_"+name+"/"

        try:
            os.mkdir(self.path)
            print("> model directory created")
            self.make_model()
            print("> model compiled")

        except FileExistsError:
            print("> model directory detected")

            ''' load model '''
            try:
                self.model = tf.keras.models.load_model(self.path+"save.model")
            except OSError:
                self.make_model()
                
            print("> model loaded")

        try:
            self.train_mape = np.load(self.path+"train_mape.npy")
            self.test_mape = np.load(self.path+"test_mape.npy")
            print("> test and train mape loaded")
            self.hist_exts = True

        except FileNotFoundError:
            print("> test and train mape not yet created")
            self.hist_exts = False
            
    def make_model(self, activation="selu", dp=0.0):
        data_shape = (31,63)
        channels = 1

        self.model = tf.keras.models.Sequential()

        ''' model build '''
        
        self.model.add(tf.compat.v2.keras.layers.Convolution2D(filters=16, kernel_size=16, activation=activation, padding="same", input_shape=(data_shape[0], data_shape[1], channels)))
        self.model.add(tf.compat.v2.keras.layers.Convolution2D(filters=8, kernel_size=8, activation=activation, padding="same"))
        self.model.add(tf.compat.v2.keras.layers.Convolution2D(filters=4, kernel_size=4, activation=activation, padding="same"))
        self.model.add(tf.compat.v2.keras.layers.MaxPooling2D(pool_size=4, padding="same"))

        self.model.add(tf.compat.v2.keras.layers.Dropout(dp))
        self.model.add(tf.compat.v2.keras.layers.Flatten())

        self.model.add(tf.compat.v2.keras.layers.Dropout(rate=0.1))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=32, activation=activation))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=16, activation=activation))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=2, activation=activation))

        ''' model compile '''

        self.model.compile(optimizer="adam", loss="mean_absolute_percentage_error")


        
    def gen_spect(self, data, plot=False):

        fs = 1./2.**12 # sampling freq: 4096 Hz
        nperseg = 2**7
        nfft = nperseg # data points in each DFFT
        n_overlap = 2**6 # overlap between blocks

        f, t, ssx = spectrogram(data, fs=fs, noverlap=n_overlap, nfft=nfft, nperseg=nperseg)

        ssx = ssx[:int(ssx.shape[1]/2),:]

        ssx = 10*np.log10(np.abs(ssx))

        if plot:
            plt.imshow(ssx)
            plt.show()

        return ssx

    
    def train_model(self, data, epochs):
        dfreq = 4096

        data_shape = (31,63)
        channels = 1

        ''' randomly shuffle data '''
        
        np.random.shuffle(data)

        ''' separate strains from masses '''

        strains, masses = data[:,:-2], data[:,-2:]

        ''' generate spectrograms for every strain '''

        strain_specs = np.zeros((data.shape[0], data_shape[0], data_shape[1], channels))

        for i in range(len(strain_specs)):
            
            strain_specs[i] = self.gen_spect(strains[i]).reshape(data_shape[0], data_shape[1], channels)

        ''' training '''

        batch_size = 54 # divisor of 5022*2
        
        history = self.model.fit(strain_specs, masses, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=2)

        if self.hist_exts:
            self.train_mape = np.concatenate((self.train_mape, np.array(history.history["val_loss"])))
            self.test_mape = np.concatenate((self.test_mape, np.array(history.history["val_loss"])))

        else:
            self.train_mape = np.array(history.history["loss"])
            self.test_mape = np.array(history.history["val_loss"])

        np.save(self.path+"train_mape.npy", self.train_mape)
        np.save(self.path+"test_mape.npy", self.test_mape)

        self.model.save(self.path+"save.model")

        self.plot_train_mape()


    def predict(self, signal):
        data_shape = (31,63)
        channels = 1
        spec = self.gen_spect(signal).reshape(1, data_shape[0], data_shape[1], channels)
        prediction = self.model.predict(x)

        return prediction
        

    def eval_model(self, signal):
        data_shape = (31,63)
        channels = 1
        strain, masses = signal[:,:-2], signal[:,-2:]
        spec = np.zeros((signal.shape[0], data_shape[0], data_shape[1], channels))

        for i in range(len(spec)): 
            spec[i] = self.gen_spect(strain[i]).reshape(data_shape[0], data_shape[1], channels)
            
        loss = self.model.evaluate(spec, masses)

        return loss

    def plot_train_mape(self):

        max_train = np.max(self.train_mape)

        mean_overfit = np.mean(np.abs((self.test_mape - self.train_mape)/self.train_mape))
        
        fig, ax = plt.subplots()

        ax.plot(range(len(self.train_mape)), self.train_mape, "-k", label="training mape")
        ax.plot(range(len(self.test_mape)), self.test_mape, "-r", label="validation mape")

        ax.set_ylim(0,max_train)
        ax.set_xlim(-0.05,1000)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAPE")

        ax.text(100, max_train/2, "mean relative overfit: {0:.2f}".format(mean_overfit))

        ax.legend()

        fig.savefig(self.path+"train_plot.png", format="png", bbox_inches="tight")

# cnn_article model ----------------------------------------------------------



class base_cnnarticle():
    '''Model is just CNN regressor for the moment'''
    def __init__(self, name):
        self.name = name
        self.path = "save_"+name+"/"

        try:
            os.mkdir(self.path)
            print("> model directory created")
            self.make_model()
            print("> model compiled")

        except FileExistsError:
            print("> model directory detected")

            ''' load model '''
            try:
                self.model = tf.keras.models.load_model(self.path+"save.model")
            except OSError:
                self.make_model()
                
            print("> model loaded")

        try:
            self.train_mape = np.load(self.path+"train_mape.npy")
            self.test_mape = np.load(self.path+"test_mape.npy")
            print("> test and train mape loaded")
            self.hist_exts = True
            
        except FileNotFoundError:
            print("> test and train mape not yet created")
            self.hist_exts = False

    def norm(self, arr):
        max_val = np.abs(np.max(arr,axis=1))
        batch_len = arr.shape[0]
        
        for i in range(batch_len):
                arr[i] = arr[i]/max_val[i]
  
        return(arr)
            
            
    def make_model(self, activation="relu", dp=0.5):

        self.model = tf.keras.models.Sequential()

        ''' model build '''
        
        '''Convolutional block 1'''
        # self.model.add(tf.compat.v2.keras.layers.Dropout(rate=dp))
        self.model.add(tf.compat.v2.keras.layers.Conv1D(filters=16, input_shape=(4096,1) ,kernel_size=4, strides=1, dilation_rate=1, activation=activation, padding='SAME'))
        self.model.add(tf.compat.v2.keras.layers.MaxPool1D(pool_size=16, strides=1, padding='SAME'))
            
        '''Convolutional block 2'''
        # self.model.add(tf.compat.v2.keras.layers.Dropout(rate=dp))
        self.model.add(tf.compat.v2.keras.layers.Conv1D(filters=16, kernel_size=4, strides=1, dilation_rate=4, activation=activation, padding='SAME'))
        self.model.add(tf.compat.v2.keras.layers.MaxPool1D(pool_size=16, strides=1, padding='SAME'))

        '''Convolutional block 3'''
        # self.model.add(tf.compat.v2.keras.layers.Dropout(rate=dp))
        self.model.add(tf.compat.v2.keras.layers.Conv1D(filters=16, kernel_size=2, strides=1, dilation_rate=4, activation=activation, padding='SAME'))
        self.model.add(tf.compat.v2.keras.layers.MaxPool1D(pool_size=8, strides=1, padding='SAME'))
        self.model.add(tf.compat.v2.keras.layers.Flatten())
        

        '''Dense block'''
        # self.model.add(tf.compat.v2.keras.layers.Dropout(rate=0.8))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=8, activation=activation))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=4, activation=activation))
        self.model.add(tf.compat.v2.keras.layers.Dense(units=2))
        

        ''' model compile '''

        self.model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    
    def train_model(self, data, epochs):
        dfreq = 4096

        ''' randomly shuffle data '''
        
        np.random.shuffle(data)

        ''' separate strains from masses '''

        strains, masses = data[:,:-2], data[:,-2:]

        ''' data normalization '''

        strains = self.norm(strains).reshape(strains.shape[0], strains.shape[1], 1)
        
        ''' training '''

        batch_size = 54 # divisor of 5022*2

        history = self.model.fit(strains, masses, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=2)


        if self.hist_exts:
            self.train_mape = np.concatenate((self.train_mape, np.array(history.history["val_loss"])))
            self.test_mape = np.concatenate((self.test_mape, np.array(history.history["val_loss"])))

        else:
            self.train_mape = np.array(history.history["loss"])
            self.test_mape = np.array(history.history["val_loss"])
        
        np.save(self.path+"train_mape.npy", self.train_mape)
        np.save(self.path+"test_mape.npy", self.test_mape)

        self.model.save(self.path+"save.model")

        self.plot_train_mape()


    def predict(self, signal):
        freq = 4096
        signal = self.norm(signal).reshape(-1,freq,1)
        
        prediction = self.model.predict(signal)

        return prediction


    def eval_model(self, signal):
        freq = 4096
        strain, masses = signal[:,:-2], signal[:,-2:]
        strain = self.norm(strain).reshape(-1,freq,1)
        
        loss = self.model.evaluate(strain, masses)

        return loss

    def plot_train_mape(self):

        max_train = np.max(self.train_mape)

        mean_overfit = np.mean(np.abs((self.test_mape - self.train_mape)/self.train_mape))
        
        fig, ax = plt.subplots()

        ax.plot(range(len(self.train_mape)), self.train_mape, "-k", label="training mape")
        ax.plot(range(len(self.test_mape)), self.test_mape, "-r", label="validation mape")

        ax.set_ylim(0,max_train)
        ax.set_xlim(-0.05,1000)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAPE")

        ax.text(100, max_train/2, "mean relative overfit: {0:.2f}".format(mean_overfit))

        ax.legend()

        fig.savefig(self.path+"train_plot.png", format="png", bbox_inches="tight")
