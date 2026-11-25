"""Keep Windows awake and the screen on while long jobs run (the laptop holds the GPU at 33 W instead of 80 W when
the screen is off). Stops by itself after the given number of hours (default 14), or when killed.

    .venv/Scripts/python.exe scripts/keep_awake.py [hours]
"""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
hours = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
end = time.time() + hours * 3600
print(f"keeping the system awake and the screen on for up to {hours:g} h", flush=True)
try:
    while time.time() < end:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        time.sleep(30)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("released", flush=True)
