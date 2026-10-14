#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_96 --model TimeMixer --data custom --features M --seq_len 96  --label_len 0  --pred_len 96  --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --batch_size 128 --learning_rate 0.01 --train_epochs 20 --patience 10 --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_96 --model TimeMixer --data custom --features M --seq_len 96  --label_len 0  --pred_len 96  --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --batch_size 128 --learning_rate 0.01 --train_epochs 20 --patience 10 --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2


python -u run.py --task_name long_term_forecast  --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_192 --model TimeMixer --data custom  --features M  --seq_len 96  --label_len 0 --pred_len 192 --e_layers 3 --factor 3 --enc_in 21 --dec_in 21  --c_out 21   --des 'Exp'  --itr 1  --d_model 16 --d_ff 32  --batch_size 128 --learning_rate 0.01 --train_epochs 20 --patience 10  --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2
python -u run.py --task_name long_term_forecast  --is_training 0 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_192 --model TimeMixer --data custom  --features M  --seq_len 96  --label_len 0 --pred_len 192 --e_layers 3 --factor 3 --enc_in 21 --dec_in 21  --c_out 21   --des 'Exp'  --itr 1  --d_model 16 --d_ff 32  --batch_size 128 --learning_rate 0.01 --train_epochs 20 --patience 10  --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2


python -u run.py --task_name long_term_forecast --is_training 1  --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_336 --model TimeMixer --data custom --features M --seq_len 96 --label_len 0 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --batch_size 128  --learning_rate 0.01 --train_epochs 20 --patience 10 --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2
python -u run.py --task_name long_term_forecast --is_training 0  --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_336 --model TimeMixer --data custom --features M --seq_len 96 --label_len 0 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --batch_size 128  --learning_rate 0.01 --train_epochs 20 --patience 10 --down_sampling_layers 3 --down_sampling_method avg  --down_sampling_window 2


python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_720 --model TimeMixer --data custom --features M --seq_len 96 --label_len 0 --pred_len 720  --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21  --c_out 21 --des 'Exp'  --itr 1  --d_model 16 --d_ff 32 --batch_size 128  --learning_rate 0.01 --train_epochs 20 --patience 10  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_720 --model TimeMixer --data custom --features M --seq_len 96 --label_len 0 --pred_len 720  --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21  --c_out 21 --des 'Exp'  --itr 1  --d_model 16 --d_ff 32 --batch_size 128  --learning_rate 0.01 --train_epochs 20 --patience 10  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
