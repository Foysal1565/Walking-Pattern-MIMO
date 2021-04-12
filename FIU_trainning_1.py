#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:59:24 2021

@author: foysal
"""

import argparse
import h5py
import json
import keras
import numpy as np
import pickle as pkl
import time
import os
import tensorflow 

from argparse import Namespace
from CustomModelCheckpoint import CustomModelCheckpoint
#from DataGenerator import DataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import CSVLogger

from keras.utils import plot_model
#from keras.utils.io_utils import HDF5Matrix
#from tensorflow.keras.utils import HDF5Matrix
from numba import njit, prange
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Utils import *
from sklearn.metrics import confusion_matrix

class BeamLearn(object):

    def __init__(self):
        '''Initialize class variables.'''
        self.args = self.parse_arguments()
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.args.id_gpu

        self.is_2d = self.args.is_2d_model
        self.num_beams = self.args.num_beams
        self.num_blocks_per_frame = self.args.num_blocks_per_frame
        self.how_many_blocks_per_frame = self.args.how_many_blocks_per_frame
        self.num_samples_per_block = self.args.num_samples_per_block
        self.num_samples_tot_gain_tx_beam = self.args.num_samples_tot_gain_tx_beam
        self.num_gains = self.args.num_gains
        self.train_perc = self.args.train_perc
        self.valid_perc = self.args.valid_perc
        self.test_perc = self.args.test_perc
        self.kernel_size = self.args.kernel_size
        self.save_best_only = True if self.args.save_best_only else False

        print("Dir path: " + self.args.save_path)
        if self.args.test_only:
            self.test_chain()
        else:
            self.train_chain()

    def train_chain(self):

        if not os.path.exists(
            os.path.join(
                self.args.save_path,
                self.args.bl_model_name
            )
        ):
            if self.is_2d:
                print('--------- Building 2D model from scratch -----------')
                self.model = build_model(
                    self.args.num_of_conv_layers,
                    self.args.num_of_kernels,
                    self.args.kernel_size,
                    self.args.num_of_dense_layers,
                    self.args.size_of_dense_layers,
                    self.args.input_size,
                    self.args.num_samples_per_block,
                    self.args.num_beams
                )
            else:
                print('--------- Building 1D model from scratch -----------')
                if not self.args.num_of_conv_layers:
                    self.model = build_model_1d_dense(
                        self.args.num_of_dense_layers,
                        self.args.size_of_dense_layers,
                        self.args.input_size,
                        self.args.num_beams
                    )
                else:
                    self.model = build_model_1d(
                        self.args.num_of_conv_layers,
                        self.args.num_of_kernels,
                        self.args.kernel_size,
                        self.args.num_of_dense_layers,
                        self.args.size_of_dense_layers,
                        self.args.input_size,
                        self.args.num_beams
                    )

            self.save_to_json()
        else:
            print('--------- Loading model from file -----------')
            self.load_from_json()
        self.load_data()
        self.train()
        self.test()

    def test_chain(self):
        self.load_from_json()
        self.load_data()
        self.test()

# Loading the data from FIU_datagen


    def load_data(
        self,
        num_samples_per_block=1024,):
        'Initialization'
        
        self.batch_size = 100
        self.data_path = '/home/foysal/ML/Walking-Pattern-MIMO/Noise_data'
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



        print('********************* Generating data for Baseline *********************')
        self.train_generator_BL = DataGenerator(indexes=self.train_indexes_BL,
                                             batch_size=self.args.batch_size,
                                             data_path=self.args.data_path,
                                             num_tx_beams=self.args.num_beams,
                                             num_blocks_per_frame=self.num_blocks_per_frame,
                                             input_size=self.args.input_size,
                                             num_samples_per_block=self.num_samples_per_block,
                                             how_many_blocks_per_frame=self.how_many_blocks_per_frame,
                                             shuffle=False,
                                             is_2d=self.is_2d)
        self.valid_generator_BL = DataGenerator(indexes=self.valid_indexes_BL,
                                             batch_size=self.args.batch_size,
                                             data_path=self.args.data_path,
                                             num_tx_beams=self.args.num_beams,
                                             num_blocks_per_frame=self.num_blocks_per_frame,
                                             input_size=self.args.input_size,
                                             num_samples_per_block=self.num_samples_per_block,
                                             how_many_blocks_per_frame=self.how_many_blocks_per_frame,
                                             shuffle=False,
                                             is_2d=self.is_2d)

        print('*********************  Generating testing data *********************')
        self.test_generator = DataGenerator(indexes=self.test_indexes,
                                            batch_size=self.args.batch_size,
                                            data_path=self.args.data_path,
                                            num_tx_beams=self.args.num_beams,
                                            num_blocks_per_frame=self.num_blocks_per_frame,
                                            input_size=self.args.input_size,
                                            num_samples_per_block=self.num_samples_per_block,
                                            how_many_blocks_per_frame=self.how_many_blocks_per_frame,
                                            is_2d = self.is_2d)


    def train(self):
        '''Train model through Keras framework.'''
        print('*************** Training Model ***************')
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])


        ''' Set up callbacks '''

        call_backs = []
        checkpoint = CustomModelCheckpoint(
                os.path.join(self.args.save_path, self.args.bl_model_name),
                monitor=self.args.stop_param, verbose=1, save_best_only=self.save_best_only)
        call_backs.append(checkpoint)
        earlystop_callback = EarlyStopping(
                    monitor=self.args.stop_param, min_delta=0, patience=self.args.patience,
                    verbose=1, mode='auto')
        call_backs.append(earlystop_callback)
        csv_logger = CSVLogger(self.args.save_path + "/train_history_log.csv", append=True)
        call_backs.append(csv_logger)

        start_time = time.time()
        self.model.fit_generator(generator=self.train_generator_BL,
                                 steps_per_epoch = self.args.max_steps if self.args.max_steps>0 else None,
                                 epochs=self.args.epochs,
                                 validation_steps=len(self.valid_generator_BL)//self.args.batch_size,
                                 validation_data=self.valid_generator_BL,
                                 shuffle=True,
                                 callbacks=call_backs,
                                 use_multiprocessing=False,
                                 max_queue_size=100)
        train_time = time.time() - start_time
        print('Time to train model %0.3f s' % train_time)
        self.best_model_path = checkpoint.best_path

    def test(self):
        #self.model.load_weights('/home/bruno/deepsig3/weights.hdf5')

        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        score = self.model.evaluate_generator(self.test_generator, verbose=1,
                                              use_multiprocessing = False)

        print('********************* Testing score ******************')
        print(score)
        f = open(self.args.save_path + "/accuracy.txt", "w")
        f.write(str(score))
        f.close()



    # def test(self):
    #     #self.model.load_weights('/home/bruno/deepsig3/weights.hdf5')
    #
    #     optimizer = Adam(lr=0.0001)
    #     self.model.compile(loss='categorical_crossentropy',
    #                        optimizer=optimizer,
    #                        metrics=['accuracy'])
    #     score = self.model.evaluate_generator(self.test_generator, verbose=1,
    #                                           use_multiprocessing = False)
    #
    #     print('********************* Testing score ******************')
    #     print(score)
    #     f = open(self.args.save_path + "/accuracy.txt", "w")
    #     f.write(str(score))
    #     f.close()


    def save_to_json(self):
        self.model.summary()
        model_json = self.model.to_json()
        json_path = self.args.save_path + "/model_arch.json"
        with open(json_path, "w") as json_file:
            json_file.write(model_json)
        print("Written model arch to " + json_path)

    def load_from_json(self):
        # load json and create model
        json_file = open(self.args.save_path + "/model_arch.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.args.save_path + "/beamlearn_model.hdf5")
        print("Loaded model from disk")


    def parse_arguments(self):
        '''Parse input user arguments.'''

        parser = argparse.ArgumentParser(description = 'Train and Validation pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--max_steps', type=int, default=0,
                            help='Max number of batches. If 0, it uses the whole dataset')

        parser.add_argument('--id_gpu', type=str, default=2,
                            help='GPU to use.')

        parser.add_argument('--is_2d_model', type=int, default=0,
                            help='Train a 1D model.')

        parser.add_argument('--bl_model_name', type=str, default='beamlearn_model.hdf5',
                            help='Name of baseline model.')

        parser.add_argument('--snr', type=str, default='all',
                            help='SNR level (low, mid, high, all).')

        parser.add_argument('--load_indexes', action='store_true',
                            help='Load indexes from external file. If False, you create and save them in "indexes.pkl".')

        parser.add_argument('--train_cnn', action='store_true',
                            help='Train CNN.')

        parser.add_argument('--save_best_only', type=int, default=1,
                            help='Save only best model during training.')

        parser.add_argument('--num_beams', type=int, default=24,
                            help='Number of beams.')

        parser.add_argument('--test_only', type=int, default=0,
                            help='Perform only testing.')

        parser.add_argument('--stop_param', type=str, default="val_acc",
                            help='Stop parameter to save model.')

        parser.add_argument('--num_samples_per_block', type=int, default=2048,
                            help='Number of samples per block.')

        parser.add_argument('--input_size', type=int, default=2048,
                            help='Number of I/Q samples per input.')

        parser.add_argument('--num_blocks_per_frame', type=int, default=15,
                            help='Number of blocks per frame.')

        parser.add_argument('--how_many_blocks_per_frame', type=int, default=1,
                            help='Number of blocks per frame I take.')

        parser.add_argument('--num_samples_tot_gain_tx_beam', type=int, default=10000,
                            help='How many frames we collected for each beam/gain pair.')

        parser.add_argument('--num_gains', type=int, default=3,
                            help='Number of different gains.')

        parser.add_argument('--kernel_size', type=int, default=6,
                            help='Kernel size in the convolutional layers.')

        parser.add_argument('--num_of_kernels', type=int, default=64,
                            help='Num of kernels in the convolutional layers.')

        parser.add_argument('--num_of_conv_layers', type=int, default=6,
                            help='Num of conv layers.')

        parser.add_argument('--num_of_dense_layers', type=int, default=2,
                            help='Num of dense layers.')

        parser.add_argument('--size_of_dense_layers', type=int, default=128,
                            help='Size of dense layers.')

        parser.add_argument('--train_perc', type=float, default=0.60,
                            help='Number of different gains.')

        parser.add_argument('--valid_perc', type=float, default=0.15,
                            help='Number of different gains.')

        parser.add_argument('--test_perc', type=float, default=0.25,
                            help='Number of different gains.')

        parser.add_argument('--save_path', type=str, default='./home/salvo/deepsig_res',
                            help='Path to save weights, model architecture, and logs.')

        parser.add_argument('--data_path', type=str,
                            default='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to data.')

        parser.add_argument('--patience', type=int, default=3,
                            help='Early stopping patience.')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for model optimization.')

        parser.add_argument('--epochs', type=int, default=25,
                            help='Number of epochs to train model.')

        return parser.parse_args()


if __name__ == '__main__':
    BeamLearn()
