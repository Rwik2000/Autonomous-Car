# Autonomous-Car
Autonomous High Speed Racing Car Project



## CARLA Simulator 

RUN IN TERMINAL 1 to launch the Simulator

`.\CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=20`
Adjust FPS so that it can be handled by the PC.



RUN THIS IN TERMINAL 2 to start running the simulator
`py -3.6 PythonClient/controlManualData.py --images-to-disk --location=data/`

Drive around to collect data.