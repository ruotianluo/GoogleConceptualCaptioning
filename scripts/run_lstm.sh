#! /bin/sh

id="lstm"
ckpt_path="log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

python train.py --id $id --caption_model att2in2 --label_smoothing 0.1 --input_json data/googletalk.json --input_label_h5 data/googletalk_label.h5 --input_fc_dir data/googlebu_fc --input_att_dir /scratch/rluo/googlebu_att.lmdb --seq_per_img 1 --batch_size 250 --input_encoding_size 512 --rnn_size 2400 --beam_size 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start -1 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 6000 --language_eval 1 --val_images_use 14000 --max_epochs 35 

#if [ ! -d xe/$ckpt_path ]; then
#cp -r $ckpt_path xe/
#fi

#python train.py --id $id --caption_model att2in2 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --seq_per_img 5 --batch_size 1000 --input_encoding_size 512 --rnn_size 2400 --beam_size 1 --learning_rate 4.294967296000003e-05 --checkpoint_path $ckpt_path --start_from $ckpt_path --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 35
