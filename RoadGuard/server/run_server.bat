@echo off
title RoadGuard Supabase Backend API Server
echo ==============================================================================
echo RoadGuard REST API Server (Supabase Cloud Backend)
echo Department of Computer Science & Engineering | IDEA Lab
echo Ramdeobaba University, Academic Year 2026-27
echo ==============================================================================
echo.

cd /d "%~dp0"
echo Starting FastAPI Uvicorn Server on http://0.0.0.0:8000 ...
python app.py
pause
