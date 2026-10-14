#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
seq_len=96
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.001
batch_size=32
train_epochs=10
patience=3

root_path=./dataset/solar/
data_path=solar_AL.txt

python -u run.py --task_name long_term_forecast --is_training 1 --root_path './dataset/solar/' --data_path 'solar_AL.txt' --model_id solar_96_96 --model TimeMixer --data Solar --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --itr 1 --use_norm 0 --d_model 512 --d_ff 2048 --channel_independence 0 --batch_size 32  --learning_rate 0.001 --train_epochs 10 --patience 3 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2
python -u run.py --task_name long_term_forecast --is_training 0 --root_path './dataset/solar/' --data_path 'solar_AL.txt' --model_id solar_96_96 --model TimeMixer --data Solar --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --itr 1 --use_norm 0 --d_model 512 --d_ff 2048 --channel_independence 0 --batch_size 32  --learning_rate 0.001 --train_epochs 10 --patience 3 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/solar/' \
  --data_path 'solar_AL.txt' \
  --model_id solar_96_192 \
  --model TimeMixer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/solar/' \
  --data_path 'solar_AL.txt' \
  --model_id solar_96_336 \
  --model TimeMixer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/solar/' \
  --data_path 'solar_AL.txt' \
  --model_id solar_96_720 \
  --model TimeMixer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2