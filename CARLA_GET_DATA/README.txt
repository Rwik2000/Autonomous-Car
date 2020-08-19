RUN IN TERMINAL 1
.\CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=20

RUN THIS IN TERMINAL 2
py -3.6 PythonClient/controlManualData.py --images-to-disk --location=data/