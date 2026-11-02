@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m ml.train_url_models || goto :error
python -m ml.train_email_models || goto :error
python -m ml.train_url_models --feature-set full || goto :error
python -m ml.evaluate_models
pause
exit /b 0
:error
echo Training failed. See the messages above.
pause
