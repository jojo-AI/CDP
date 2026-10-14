#!/bin/bash
# ETTh1-96 专攻 MAE：MAE 目前只有 −1.9%，是唯一拖后腿的指标。
# 假设：后验线性校准改不动「条件中位数」，必须让骨干训练本身以 L1 为主导目标。
#   m1 : stage1 dynamic + mse_l1(mu=1.0)
#   m2 : stage1 dynamic + mse_l1(mu=4.0)   ← L1 主导
#   m3 : stage1 dynamic + 纯 L1
# 然后各自接 stage2（cdp_loss=l1 或 mse_l1）
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/stage1_results.txt"
DS=ETTh1; PL=96

for MU in 1.0 4.0; do
  TLOSS=mse_l1 TL1MU=$MU EP_OVR=30 bash scripts/run_stage1_st.sh $DS $PL dynamic
done
TLOSS=l1 TL1MU=0.0 EP_OVR=30 bash scripts/run_stage1_st.sh $DS $PL dynamic
echo "MAE ROUND STAGE1 DONE"
