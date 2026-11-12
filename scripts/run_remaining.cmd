@echo off
rem Runs (or resumes after a restart) the remaining reruns on the clean catalogue:
rem router variants, Table 8, Table 10, Table 11, significance family, and
rem finally leave-one-domain-out (Table 7, the long overnight step).
rem Every language model call is cached per item, so rerunning this file after a
rem power cut only redoes unfinished work. Double-click it, or run from any folder.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe

lms server start
lms ps | findstr /c:"qwen/qwen3.5-9b" >nul || lms load qwen/qwen3.5-9b -y

call :step run_router_variants router_variants || exit /b 1
call :step run_routing_analysis table8_routing_analysis || exit /b 1
call :step run_ablations table10_ablations || exit /b 1
call :step run_efficiency table11_efficiency || exit /b 1
call :step run_significance significance || exit /b 1
call :step run_leave_one_domain_out table7_leave_one_domain_out || exit /b 1
echo All remaining reruns complete. Results are in the results folder.
exit /b 0

:step
echo === %1 (log: logs\%2.log) ===
%PY% -m experiments.%1 --config configs/default.yaml > logs\%2.log 2> logs\%2.err.log
if errorlevel 1 (
  echo FAILED: see logs\%2.err.log
  exit /b 1
)
exit /b 0
