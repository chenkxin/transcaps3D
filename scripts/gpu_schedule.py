# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time

cmd = "./scripts/run_shrec15_divide.sh"


def gpu_info():
    gpu_status = os.popen("nvidia-smi | grep %").read().split("|")
    gpu_memory = int(gpu_status[2].split("/")[0].split("M")[0].strip())
    gpu_power = int(gpu_status[1].split("   ")[-1].split("/")[0].split("W")[0].strip())
    return gpu_power, gpu_memory


def criterion(gpu):
    return gpu > 1000


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()

    def gpu_availuable():
        global gpu_power, gpu_memory
        gpu_power, gpu_memory = gpu_info()
        busy = criterion(gpu_memory)
        if not busy:
            time.sleep(60)
            gpu_power, gpu_memory = gpu_info()
            busy = criterion(gpu_memory)
        return not busy

    i = 0
    while not gpu_availuable():
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = "monitoring: " + ">" * i + " " * (10 - i - 1) + "|"
        gpu_power_str = "gpu power:%d W |" % gpu_power
        gpu_memory_str = "gpu memory:%d MiB |" % gpu_memory
        sys.stdout.write("\r" + gpu_memory_str + " " + gpu_power_str + " " + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print("\n" + cmd)
    os.system(cmd)


if __name__ == "__main__":
    narrow_setup()
