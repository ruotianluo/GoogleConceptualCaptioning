#! /bin/sh

id="lstm_rl"
ckpt_path="log_"$id

if [ ! -d $ckpt_path ]; then
  bash scripts/copy_model.sh lstm $id
fi

start_from="--start_from "$ckpt_path

python train.py --id $id --caption_model att2in2 --label_smoothing 0.1 --input_json data/googletalk.json --input_label_h5 data/googletalk_label.h5 --input_fc_dir data/googlebu_fc --input_att_dir /scratch/rluo/googlebu_att.lmdb --seq_per_img 1 --batch_size 50 --input_encoding_size 512 --rnn_size 2400 --beam_size 1 --learning_rate 1e-5 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 6000 --language_eval 1 --val_images_use 14000 --max_epochs 50 --structure_after 10 --structure_sample_n 6 --structure_loss_weight 1 --structure_loss_type policy_gradient --cached_token google-train-idxs

#python train.py --id $id --caption_model transformer --input_json data/googletalk.json --input_label_h5 data/googletalk_label.h5 --input_fc_dir data/googlebu_fc.lmdb --input_att_dir /scratch/rluo/googlebu_att.lmdb --seq_per_img 1 --batch_size 15 --beam_size 1 --learning_rate 1e-5 --learning_rate_decay_start -1 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 10000 --max_epochs 50 --structure_after 10 --structure_sample_n 6 --structure_loss_weight $1 --structure_loss_type policy_gradient --cached_token google-train-idxs 

