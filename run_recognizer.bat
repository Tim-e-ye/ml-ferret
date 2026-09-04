@echo off
chcp 65001 >nul
echo 正在运行围棋棋盘自动识别与数据集生成工具...
D:\Code\Python\Anaconda\MIniconda\Application\envs\go_env\python.exe tools\go_board_recognizer.py --image_dir Datasets\Dataset1 --output_dir Datasets --num_viz 20 --workers 4
pause
