from DataGenerator import DataGenerator
import numpy as np


def main():
    gains = [40, 50, 60]
    tx_beams = np.arange(0, 24)
    num_samples_tot_gain_tx_beam = 10000

    # Order is gain *

    indexes = np.arange(
        0,
        num_samples_tot_gain_tx_beam * len(tx_beams) * len(gains)
    )
    batch_size = 32
    data_path = '/media/michele/rx-12-tx-tm-0-rx-tm-1.h5'

    num_blocks_per_frame = 15
    how_many_blocks_per_frame = 1
    num_samples_per_block = 2048
    num_tx_beams = len(tx_beams)
    input_size = 1024
    
    dg = DataGenerator(
        indexes,
        batch_size,
        data_path,
        num_tx_beams,
        num_blocks_per_frame,
        input_size,
        num_samples_per_block,
        how_many_blocks_per_frame,
        shuffle=False,
        is_2d=False
    )

    batch_gain_tx_beam = num_samples_tot_gain_tx_beam / batch_size


    # for [i_g, val_g] in enumerate(gains):
    #     print("Gain: " + str(val_g))
    #     for [i_t, val_t] in enumerate(tx_beams):
    #         print("Beam idx: " + str(val_t))
    #         batch_index = (i_g * len(tx_beams) * batch_gain_tx_beam) + i_t * batch_gain_tx_beam
    #         print("Batch idx: " + str(batch_index))
    #         [batch, batch_y] = dg.__getitem__(batch_index)
    #         print("tx_beam %d y % s" % (val_t, batch_y[0]))
    #         # print(batch_y[0])


    for i in range(dg.__len__()):
        print("Batch idx: " + str(i))
        [batch, batch_y] = dg.__getitem__(i)
        print("tx_beam %s %s y %s %s" % (batch[0][0], batch[-1][0], batch_y[0], batch_y[-1]))
        print("batch_x_size: %s, batch_y_size: %s" % (str(batch.shape), str(batch_y.shape)))




if __name__ == '__main__':
    main()
