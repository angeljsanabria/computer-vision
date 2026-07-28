Ver logs:

cd (enter) -> Te lleva al home
tail -f .local/state/anpr-kiosk.log


Reset:

sudo systemctl restart lightdm

## Ver uso de recursos
- Ver uso de NPU y RGA:
    - watch -n 0.1 cat /sys/kernel/debug/rknpu/load
    - watch -n 0.1 cat /sys/kernel/debug/rkrga/load
    - watch -n 0.1 'sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null'
    - watch -n 0.1 'sudo cat /sys/kernel/debug/rkrga/load 2>/dev/null'
- Ver uso de CPU:
    - Obtener PID (process id)
    - pgrep -f main_mov.py
    - o: 
        - PID=$(pgrep -f "python3 main_mov.py")
        - echo $PID
        - PID=$(pgrep -f "python3 main_mov.py")

        - or:
        - PID=$(pgrep -f "/opt/conda/envs/anpr/bin/python /opt/anpr-core/src/main_track.py")
    - pidstat -u -r -p $PID 1