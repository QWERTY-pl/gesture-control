@echo off
chcp 65001 >nul
title Gesture Control

cd /d "%~dp0"

echo ========================================
echo   Gesture Recognition Control Program
echo ========================================
echo.
echo Starting program...
echo.

python hand_gesture_control.py

if errorlevel 1 (
    echo.
    echo Error: Failed to run the program
    echo Please install dependencies first: pip install -r requirements.txt
    echo.
    pause
)
