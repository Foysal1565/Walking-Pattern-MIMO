#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load "/home/foysal/onedrive/Northeastern University/Spring 2021/FIU ML/Data Generator/RAND-Lab-DataGenerator2.py"
#!/usr/bin/python3
# (C) Written by Udara De Silva (udesi001@fiu.edu)
import os
import numpy as np
import keras

from scipy.io import loadmat
from IPython.core.debugger import set_trace

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(
        self,
        indexes,
        batch_size,
        data_path = 'Noise_data',
        num_rx_beams = 32,
        num_blocks_per_frame = 1,
        num_samples_per_block = 1024,
        shuffle=False,
        is_2d=False
    ):
        
        'Initialization'
        self.indexes = indexes
        self.batch_size = 100
        self.data_path =  'Noise_data'
        self.shuffle =shuffle
        self.cache = {}
        # A frame is a matlab file
        # Each frame has Y number of blocks with length X
        # X * Y < number of samples per frame
        self.num_frames_per_pattern = 1000
        self.num_samples_per_frame = 128000 # number of samples in one file
        self.num_blocks_per_frame = 1 # Y
        self.num_samples_per_block = num_samples_per_block # X
        self.num_rx_beams = num_rx_beams = 32 # channel count
        self.is_2d = is_2d
        self.cache_rate = 10
        self.walk_patterns = ['P11', 'P12', 'P21', 'P22', 'P31', 'P41']
        self.num_patterns=len(self.walk_patterns)
        if not self.num_samples_per_block * self.num_blocks_per_frame <= self.num_samples_per_frame:
            print('[ERROR  ]: number of samples per block * number of blocks per frame > number of samples in a frame')


    def __len__(self):
        'Denotes the number of batches per epoch.'
        return int(self.num_patterns*(self.num_frames_per_pattern//np.ceil(self.batch_size/self.num_blocks_per_frame)))

    
    def build_cache(self):
        '''Add indexes to cache.'''
        size = self.__len__()
        for i in range(size):
            print("Adding to cache " + str(i) + "...")
            self.cache[i] = self.__getitem__(i)
    
    def read_from_cache(self, index):
        return None if index not in self.cache else self.cache[index]

    def __getitem__(self, index):
        'Generate one batch of data.'


        f_pattern = index % self.num_patterns
        # if batch size > num_blocks_per_frame we need to use several frames(files)
        f_trial   = int( (index//self.num_patterns) * np.ceil(self.batch_size // self.num_blocks_per_frame) + 1)

        file = os.path.join( self.data_path,  self.walk_patterns[f_pattern] + '_trial' + str(f_trial))

        block_index = 0
        out = np.zeros((self.batch_size, self.num_rx_beams, self.num_samples_per_block, 2)) # out(1000, 32, block_size, 2)
        for i in range(self.batch_size):
            data = self.next_block(file, block_index)
            if data is None:
                block_index = 0
                f_trial += 1
                file = os.path.join(self.data_path, self.walk_patterns[f_pattern] + '_trial' + str(f_trial))
                data = self.next_block(file, block_index)

            out[i, :, :, :] = np.stack((data, f_pattern * np.ones((self.num_rx_beams, self.num_samples_per_block))), axis=2)
            block_index += 1

        x = out[:,:,:,0]
        y = out[:,:,:,1]
        
        if not self.is_2d:  # this is done just to use 2D models and add a new dimension
            x = np.squeeze(x)

        return x, y
    
    def __fetch_index(self, index):
        self.x_out = self.x[index]
        self.y_out = self.y[index]

    def next_block(self, matfile, block_index):
        if block_index + 1 > self.num_blocks_per_frame:
            return None

        mat = loadmat(matfile)
        data = mat['export_noise_mat']
        data = data[:,(block_index*self.num_samples_per_block):(block_index+1)*self.num_samples_per_block]
        return data


if __name__ == '__main__':
    dg = DataGenerator(10,'./Noise_data')

    print('Size of datagen : {}'.format(len(dg)))

    for (i, data) in enumerate(dg):
        print('Batch Index {}\n\n'.format(i))
        print(data[0][0, 0:5, 0:5])
    exit()
