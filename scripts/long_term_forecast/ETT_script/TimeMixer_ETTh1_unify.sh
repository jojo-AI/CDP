export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

# train and test
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/  --data_path ETTh1.csv --model_id ETTh1_96'_'96 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 7 --c_out 7  --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
# purely test
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small/  --data_path ETTh1.csv --model_id ETTh1_96'_'96 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 7 --c_out 7  --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2


python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_192 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 192 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_192 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 192 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2


python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv  --model_id ETTh1_96'_'336 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 336 --e_layers 2 --enc_in 7 --c_out 7  --des 'Exp' --itr 1  --d_model 16  --d_ff 32 --learning_rate 0.01 --train_epochs 10  --patience 10 --batch_size 128 --down_sampling_layers 3  --down_sampling_method avg  --down_sampling_window 2
python -u run.py  --task_name long_term_forecast  --is_training 0  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv  --model_id ETTh1_96'_'336 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 336 --e_layers 2 --enc_in 7 --c_out 7  --des 'Exp' --itr 1  --d_model 16  --d_ff 32 --learning_rate 0.01 --train_epochs 10  --patience 10 --batch_size 128 --down_sampling_layers 3  --down_sampling_method avg  --down_sampling_window 2


# train and test
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_720 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 720 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
# test
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_720 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 720 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
