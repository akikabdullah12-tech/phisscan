@echo off
REM PhisScan - one-time setup on Windows. Run from the project folder.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup complete. Start the app with:  run.bat
echo (Optional) retrain all models with:  train_all.bat
pause
