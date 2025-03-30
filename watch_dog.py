import subprocess
import time
import datetime
import sys
import os
import signal
import psutil  # You may need to: pip install psutil

# Configuration
PYTHON_PATH = sys.executable  # Path to the current Python interpreter
BOT_SCRIPT_PATH = r"C:\trading_bot\enhanced-bots.py"
LOG_PATH = r"C:\trading_bot\watchdog.log"
MAX_MEMORY_PERCENT = 90  # Restart if memory usage exceeds this percentage
HEALTH_CHECK_INTERVAL = 30  # Seconds between health checks
RESTART_DELAY = 10  # Seconds to wait before restart after normal exit
ERROR_RESTART_DELAY = 60  # Seconds to wait before restart after error

def log_message(message):
    """Write a timestamped message to the log file and console"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    
    try:
        with open(LOG_PATH, "a") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")
        
    print(log_entry)

def is_process_healthy(process):
    """Check if a process is running and using resources normally"""
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        return False
        
    # Check memory usage
    try:
        mem_percent = process.memory_percent()
        if mem_percent > MAX_MEMORY_PERCENT:
            log_message(f"Memory usage too high: {mem_percent:.1f}% > {MAX_MEMORY_PERCENT}%")
            return False
    except:
        pass
        
    return True

def create_process():
    """Start the trading bot process and return it"""
    try:
        process = subprocess.Popen(
            [PYTHON_PATH, BOT_SCRIPT_PATH],
            cwd=os.path.dirname(BOT_SCRIPT_PATH),
            stdout=open(r"C:\trading_bot\bot_output.log", "a"),
            stderr=open(r"C:\trading_bot\bot_error.log", "a"),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Important for Windows
        )
        return process
    except Exception as e:
        log_message(f"Failed to create process: {e}")
        return None

log_message(f"Watchdog started - Python: {PYTHON_PATH}, Bot: {BOT_SCRIPT_PATH}")

# Main watchdog loop
restart_count = 0
while True:
    try:
        # Start the trading bot
        log_message(f"Starting trading bot (restart #{restart_count})")
        process = create_process()
        
        if not process:
            log_message("Failed to start bot process, retrying in 60 seconds...")
            time.sleep(60)
            continue
            
        process_psutil = psutil.Process(process.pid)
        log_message(f"Bot process started with PID: {process.pid}")
        
        # Monitor the process
        while True:
            # Check if process has exited naturally
            if process.poll() is not None:
                exit_code = process.returncode
                log_message(f"Bot process exited with code {exit_code}")
                break
                
            # Check process health
            if not is_process_healthy(process_psutil):
                log_message("Process is not healthy, terminating...")
                try:
                    process.terminate()
                    # Wait up to 10 seconds for graceful termination
                    for _ in range(10):
                        if process.poll() is not None:
                            break
                        time.sleep(1)
                    # Force kill if still running
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
                break
                
            # Process is running normally, wait before next check
            time.sleep(HEALTH_CHECK_INTERVAL)
            
        # Prepare for restart
        restart_count += 1
        delay = RESTART_DELAY
        
        # Use longer delay if there was an error
        if process.returncode != 0:
            delay = ERROR_RESTART_DELAY
            
        log_message(f"Restarting in {delay} seconds...")
        time.sleep(delay)
        
    except KeyboardInterrupt:
        log_message("Watchdog terminated by user")
        break
    except Exception as e:
        log_message(f"Watchdog error: {e}")
        log_message(f"Restarting watchdog in {ERROR_RESTART_DELAY} seconds...")
        time.sleep(ERROR_RESTART_DELAY)

log_message("Watchdog stopped")