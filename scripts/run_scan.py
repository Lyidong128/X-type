print("Cloud agent running!")

import os
os.makedirs("outputs", exist_ok=True)

with open("outputs/test.txt", "w") as f:
    f.write("success")
import datetime

# 日志记录函数
def log_activity(command, result_summary):
    log_file = "logs/activity_log.txt"

    # 获取当前时间
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 打开文件，追加写入
    with open(log_file, "a") as log:
        log.write(f"### 指令提交记录 ###\n")
        log.write(f"[{timestamp}]\n")
        log.write(f"提交指令：{command}\n")
        log.write(f"生成结果：{result_summary}\n\n")
        log.write(f"### 生成的结果 ###\n")
        log.write(f"[{timestamp}]\n")
        log.write(f"生成的结果：{result_summary}\n\n")

# 示例：调用日志记录函数
command = "扫描 v, t, lm 参数并计算 gap"
result_summary = "成功计算了 gap，结果已写入 results.csv"

log_activity(command, result_summary)
