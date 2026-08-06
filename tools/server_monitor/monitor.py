import psutil, subprocess, requests, socket
from datetime import datetime

API_KEY = "serversilabatsumerukun"
CONTROLLER_URL = "http://10.40.34.3:8088/api/collect"

def get_gpu_info():
    try:
        res = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ])
        gpus = []
        for line in res.decode().strip().split('\n'):
            name, util, used, total = line.split(', ')
            gpus.append({
                "name": name.strip(),
                "util": int(util),
                "mem_used": int(used),
                "mem_total": int(total)
            })
        return gpus
    except Exception:
        return []

def collect_info():
    return {
        "server": socket.gethostname(),
        "timestamp": datetime.now().isoformat(),
        "cpu": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/home')._asdict(),
        "gpu": get_gpu_info(),
    }

def send_to_controller():
    data = collect_info()
    headers = {"x-api-key": API_KEY}
    try:
        r = requests.post(CONTROLLER_URL, json=data, headers=headers, timeout=5)
        print(f"[{data['server']}] sent ({r.status_code})")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    send_to_controller()
