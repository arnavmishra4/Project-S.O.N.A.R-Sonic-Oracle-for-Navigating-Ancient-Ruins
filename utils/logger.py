import datetime
import sys

def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    color = {"INFO": "\033[94m", "WARN": "\033[93m", "ERROR": "\033[91m"}.get(level, "\033[0m")
    reset = "\033[0m"
    sys.stdout.write(f"{color}[{timestamp}] [{level}] {msg}{reset}\n")
    sys.stdout.flush()
