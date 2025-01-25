@echo off
set uv_path=e:\data\.rye\shims
set python_path=e:\prog\py310
set path=%uv_path%;%python_path%;%path%
if exist ".venv" (
  echo .venv ok
) else (
  uv venv
)
call .venv\scripts\activate.bat
set UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
start cmd
