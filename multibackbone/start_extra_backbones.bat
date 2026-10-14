@echo off
title CycleNet + iTransformer CDP batch (third backbones)
cd /d %~dp0
set CDP_WORKERS=2
set CDP_THREADS=3
"C:\Users\yjy\.workbuddy\binaries\python\envs\default\Scripts\python.exe" run_batch_extra.py
pause
