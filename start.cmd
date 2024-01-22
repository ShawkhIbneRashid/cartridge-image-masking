@echo off

echo.
echo Restoring backend python packages
echo.
call python -m pip install --user -r requirements.txt
if "%errorlevel%" neq "0" (
    echo Failed to restore backend python packages
    exit /B %errorlevel%
)

echo.
echo Running streamlit app
echo.
cd model-deployment-streamlit-app
call python -m streamlit run app.py
if "%errorlevel%" neq "0" (
    echo Failed to start backend
    exit /B %errorlevel%
)
