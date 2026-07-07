@echo off
chcp 65001 >nul
title 手势识别电脑控制程序

cd /d "%~dp0"

echo ========================================
echo   手势识别电脑控制程序
echo ========================================
echo.
echo 正在启动程序...
echo.

python hand_gesture_control.py

if errorlevel 1 (
    echo.
    echo 程序运行出错，请检查是否已安装必要的依赖库
    echo 可运行以下命令安装依赖: pip install -r requirements.txt
    echo.
    pause
)
