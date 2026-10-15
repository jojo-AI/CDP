#!/usr/bin/env bash
# =====================================================================
# CDP M/K 扫描一键执行（A100 / Linux, ssh）
# 用法:  解压后进入本目录,  bash run_all.sh
#   - 自动找带 CUDA torch 的 python（可用 CDP_PY 环境变量指定）
#   - 全部跑完把 回传_MK扫描结果_*.zip 复制到你启动命令的目录
#   - 中断后重新执行同一命令即断点续跑（已完成的点自动跳过）
# 选卡:  CUDA_VISIBLE_DEVICES=0 bash run_all.sh
# =====================================================================
set -u
LAUNCH_DIR="$(pwd)"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"

echo "== [1/3] 检查 python + CUDA torch =="
PYBIN=""
for c in "${CDP_PY:-}" python3 python; do
  [ -z "$c" ] && continue
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      PYBIN="$(command -v "$c")"; break
    fi
  fi
done
if [ -z "$PYBIN" ]; then
  echo "  未找到带 CUDA torch 的 python → 尝试创建 venv 并安装依赖(不含 torch,"
  echo "  torch 需集群另行提供或参考 README_A100.md 的安装说明)..."
  python3 -m venv mk_venv || { echo "!! venv 创建失败"; exit 1; }
  ./mk_venv/bin/pip install -q --upgrade pip
  ./mk_venv/bin/pip install -q -r requirements_a100.txt || { echo "!! 依赖安装失败"; exit 1; }
  PYBIN="$PKG_DIR/mk_venv/bin/python"
fi
echo "  使用 python: $PYBIN"
"$PYBIN" - <<'EOF'
import torch
print("  torch %s | cuda %s | %s" % (torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
EOF

echo "== [2/3] 运行 M/K 扫描（预计 3~5 小时，A100 单卡）=="
"$PYBIN" mk_scan/run_batch_mk.py 2>&1 | tee mk_scan_console.log
RC=${PIPESTATUS[0]}

echo "== [3/3] 导出回传包 =="
ZIP=$(ls -t *.zip 2>/dev/null | head -n 1)
if [ -n "${ZIP:-}" ]; then
  cp -v "$ZIP" "$LAUNCH_DIR/"
  echo ""
  echo ">>> 回传包已导出: $LAUNCH_DIR/$ZIP"
  echo ">>> 下载它发回即可。"
else
  echo "!! 未找到回传 zip（RC=$RC）。查看 mk_scan_console.log 与 mk_logs/ 排错。"
  echo "   修复后重新执行 bash run_all.sh 即可续跑。"
  exit 1
fi
