# Microbot Control

Control software for a magnetic helical microrobot: two cameras track the bot and a 3-axis coil steers it.

## Install

Needs Python 3.10 or newer on Windows (the cameras use Windows video drivers).

```
pip install -r requirements.txt
python main.py
```

## Use

1. **Control tab:** choose the coil's COM port and click **Connect**.
2. **Cameras:** click **Connect** on both camera views. With the tank empty, click **Recapture Background** on each camera.
3. **Closed Loop tab:** choose **Pursuit** or **Line following** and a route, then click **Start**. Tick **Loop the route until STOP** to keep going round. **STOP** switches the coils off.
4. **Keyboard drive:** arrow keys move the bot in X/Y, W/S move it up and down.

You can also add your own waypoints in the **Waypoints** box. The cameras' position calibration only holds for the rig's original camera positions.
