#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

id_gpu=1
test_only=0

# Dataset parameters

num_samples_tot_gain_tx_beam=10000
num_samples_per_block=2048
how_many_blocks_per_frame=1

input_size=512

# Piradios

# 5beam
# num_gains=3
# num_beams=5
# num_blocks_per_frame=5

# 3beam
# num_gains=1
# num_beams=3
# num_blocks_per_frame=5

num_gains=3
num_beams=24
num_blocks_per_frame=15

is_2d_beam=1
is_2d_model=0
snr="high"


# Training parameters

epochs=20
batch_size=100
train_perc=0.60
valid_perc=0.00
test_perc=0.40
save_best_only=1
stop_param="acc"

# Model parameters

kernel_size=7
num_of_kernels=16
num_of_conv_layers=1
num_of_dense_layers=0
size_of_dense_layers=128
patience=100

root=/home/frestuc/projects/modrec_thz/saved_models/ #customize!
data_path=/media/michele/rx-12-tx-tm-0-rx-tm-1.h5 #customize

save_path=$root
save_path+="training_code_cl_$num_of_conv_layers"
save_path+="_nk_$num_of_kernels"
save_path+="_nsp_$input_size"
save_path+="_ks_$kernel_size"
save_path+="_dl_$num_of_dense_layers"
save_path+="_sd_$size_of_dense_layers"
save_path+="_bf_$how_many_blocks_per_frame"
save_path+="_srn_$snr"
save_path+="_2dbeam_$is_2d_beam"
save_path+="_2dmodel_$is_2d_model"
save_path+="_ne_$epochs"
save_path+="_bs_$batch_size"

python2 ./TrainingCode.py \
    --data_path $data_path \
    --batch_size $batch_size \
    --train_cnn \
    --test_only $test_only \
    --epochs $epochs \
    --save_best_only $save_best_only \
    --stop_param $stop_param \
    --snr $snr \
    --num_blocks_per_frame $num_blocks_per_frame \
    --how_many_blocks_per_frame $how_many_blocks_per_frame \
    --num_samples_per_block $num_samples_per_block \
    --num_samples_tot_gain_tx_beam $num_samples_tot_gain_tx_beam \
    --num_gains $num_gains \
    --num_beams $num_beams \
    --train_perc $train_perc \
    --valid_perc $valid_perc \
    --input_size $input_size \
    --test_perc $test_perc \
    --kernel_size $kernel_size \
    --num_of_kernels $num_of_kernels \
    --num_of_conv_layers $num_of_conv_layers \
    --num_of_dense_layers $num_of_dense_layers \
    --size_of_dense_layers $size_of_dense_layers \
    --id_gpu $id_gpu \
    --is_2d_model $is_2d_model \
    --patience $patience \
	--save_path $save_path

