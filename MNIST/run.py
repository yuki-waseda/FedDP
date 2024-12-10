import subprocess
import os

# 実行するPythonスクリプトのリスト
scripts = [
    "run_noDP_MNIST.py",
    "run_malDP_MNIST.py",
    "run_kmeans_MNIST.py",
    "run_krum_MNIST.py"
]

# 実行するPythonインタプリタ（必要に応じてフルパスを指定）
python_executable = "/home/y.okura/local/python/bin/python3"

# 実行ログを保存するディレクトリ
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # ログディレクトリがなければ作成

for script in scripts:
    log_file = os.path.join(log_dir, f"{os.path.splitext(script)[0]}.log")
    print(f"Running {script}...")
    
    with open(log_file, "w") as log:
        try:
            # スクリプトの実行
            subprocess.run([python_executable, script], stdout=log, stderr=log, check=True)
            print(f"{script} executed successfully. Logs saved to {log_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}. Check logs: {log_file}")
