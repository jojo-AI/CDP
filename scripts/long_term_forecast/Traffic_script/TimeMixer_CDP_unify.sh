# CDP (TimeMixer backbone, frozen) - Traffic 全 4 个 horizon
# Stage 1 baseline checkpoint 已存在于 ./checkpoints/（TimeMixer 训练产物）
# Stage 2: 冻结骨干，只训 CDP
# 用法: bash scripts/long_term_forecast/Traffic_script/TimeMixer_CDP_unify.sh
#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer_CDP

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
d_model=32
d_ff=64
batch_size=8
cdp_lr=0.0001

root_path=./dataset/traffic/
data_path=traffic.csv

for pred_len in 96 192 336 720
do
  base_ckpt="./checkpoints/long_term_forecast_Traffic_96'_'${pred_len}_none_TimeMixer_custom_sl96_pl${pred_len}_dm32_nh8_el3_dl1_df64_fc3_ebtimeF_dtTrue_Exp_0"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id Traffic_CDP_96'_'${pred_len} \
    --model TimeMixer_CDP \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --d_model 32 \
    --d_ff 64 \
    --batch_size 8 \
    --learning_rate 0.01 \
    --train_epochs 50 \
    --patience 5 \
    --num_workers 0 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --cdp_M 4 \
    --cdp_S 16 \
    --cdp_err_decay 0.9 \
    --cdp_lambda_t 0.01 \
    --cdp_learning_rate ${cdp_lr} \
    --cdp_backbone_checkpoint "$base_ckpt"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id Traffic_CDP_96'_'${pred_len} \
    --model TimeMixer_CDP \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --d_model 32 \
    --d_ff 64 \
    --batch_size 8 \
    --learning_rate 0.01 \
    --num_workers 0 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2
done
