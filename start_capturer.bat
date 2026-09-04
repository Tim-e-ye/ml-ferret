@echo off
chcp 65001 >nul
echo 正在启动围棋棋盘连续截图工具...
python tools\go_screen_capturer.py
pause
