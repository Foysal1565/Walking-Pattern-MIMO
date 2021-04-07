import os
import numpy as np
import keras

from scipy.io import loadmat
from IPython.core.debugger import set_trace
from IPython.display import display

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(
        self,
        batch_size,
        data_path,
        num_samples_per_block=1024,
    ):
        'Initialization'
        self.batch_size = 100
        self.data_path =  '/home/foysal/ML/Noise_data'
        # A frame is a matlab file
        # Each frame has Y number of blocks with length X
        # X * Y < number of samples per frame
        self.num_frames_per_pattern=1000
        self.num_samples_per_frame=128000 # number of samples in one file
        self.num_blocks_per_frame = 1 # Y
        self.num_samples_per_block = num_samples_per_block # X
        self.num_rx_beams = 32 # channel count
        self.walk_patterns = ['P11', 'P12', 'P21', 'P22', 'P31', 'P41']
        self.num_patterns=len(self.walk_patterns)
        if not self.num_samples_per_block * self.num_blocks_per_frame <= self.num_samples_per_frame:
            print('[ERROR  ]: number of samples per block * number of blocks per frame > number of samples in a frame')


    def __len__(self):
        'Denotes the number of batches per epoch.'
        return int(self.num_patterns*(self.num_frames_per_pattern//np.ceil(self.batch_size/self.num_blocks_per_frame)))

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

        return x, y

    def next_block(self, matfile, block_index):
        if block_index + 1 > self.num_blocks_per_frame:
            return None

        mat = loadmat(matfile)
        data = mat['export_noise_mat']
        data = data[:,(block_index*self.num_samples_per_block):(block_index+1)*self.num_samples_per_block]
        return data
if __name__ == '__main__':
    dg = DataGenerator(10,'./Noise_data')
    display('Size of datagen : {}'.format(len(dg)))
    for (i, data) in enumerate(dg):
        display('Batch Index {}\n\n'.format(i))
        display(data[0][0, 0:5, 0:5])
        exit()
