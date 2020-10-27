import json
import matplotlib.pyplot as plt
import math
import numpy as np
# from filter import Filter

# butterFilter = Filter()

f = open('xyPoints150.txt','r')

points = json.load(f)

x = []
y = []
for point in points:
    x.append(point[0])
    y.append(point[1]+5)

ftrack = open('track150.txt', 'r')
trackPoints = json.load(ftrack)

xtrack = []
ytrack = []
for point in trackPoints:
    xtrack.append(point[0])
    ytrack.append(point[1])


# plt.pause(5)

origTheta = []
for i in range(1, len(x)):

    ydiff = (y[i] - y[i-1])
    xdiff = (x[i] - x[i-1])


    dir = math.atan2(ydiff,  xdiff)

    if i!= 1 and origTheta[-1] < 0 and dir > 2.5:
        dir -= 2*math.pi
    origTheta.append(dir)




origThetaDash = []
for i in range(1, len(origTheta)):

    ydiff = (origTheta[i] - origTheta[i-1])
    origThetaDash.append(ydiff)



def moving_average(a, n=5) :
    
    # a += [a[-1] for i in range(n+1)]
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    retval = (ret[n - 1:] / n).tolist()
    retval =  retval + [a[-1] for i in range(n+1)]

    return retval


smoothThetaDash = moving_average(origThetaDash, n = 4)

# dt  = x[1] - x[0]

# def low_pass(x_new, y_old, cutoff = 10):
    
#     alpha = dt / (dt + 1 / (2 * np.pi * cutoff))
#     y_new = x_new * alpha + (1 - alpha) * y_old
#     return y_new

# def continuous_filter(xs):
#     y_prev_low = 0  # initialization

#     for x in xs:
#         # y_prev_high = high_pass(x, x_prev, y_prev_high)
#         y_prev_low = low_pass(x, y_prev_low)
#         # x_prev = x
#         yield y_prev_low


# smoothThetaDash = np.array([out for out in continuous_filter(origThetaDash)])


smoothTheta = [smoothThetaDash[0]]


offset = smoothTheta[0] - origTheta[0]


print("Offset =", offset)
offset -= 0.15
print("Offset =", offset)
for i in range(1, len(smoothThetaDash)):
    smoothTheta.append(smoothTheta[-1] + smoothThetaDash[i])

# smoothTheta.append(origTheta[-2])
# smoothTheta.append(origTheta[-1])

for i in range(len(smoothTheta)):
    smoothTheta[i] -= offset




startSmooth = 1
smoothTheta = [smoothTheta[0] for i in range(startSmooth+1)] + smoothTheta

for i in range(startSmooth+1):
    smoothTheta.pop(-1)






projectedx = x.copy()
projectedy = [y[0]]

for i in range(1, len(smoothTheta)):
    projectedy.append(projectedy[-1] + math.tan(smoothTheta[i])*(projectedx[i]  - projectedx[i-1]))
    




plt.figure(2)
plt.plot(origThetaDash, label = 'Original')
plt.plot(smoothThetaDash, label = 'Smoothened')
plt.legend()




plt.figure(1)
plt.plot(origTheta, label = 'Original')
plt.plot(smoothTheta, label = 'Smoothened')
plt.legend()


plt.figure(3)
plt.plot(x, y, label = 'Original')
plt.plot(projectedx, projectedy, label = 'Smoothened')
plt.plot(xtrack, ytrack,'+', label = 'track')
plt.legend()


# plt.figure(4)


plt.show()


# order = 6
# fs = 30.0       # sample rate, Hz
# cutoff = 3.667  # desired cutoff frequency of the filter, Hz
# T = 5.0         # seconds
# n = int(T * fs) # total number of samples
# t = np.linspace(0, T, n, endpoint=False)
# data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# filteredOuptut = butterFilter.butter_lowpass_filter(data, cutoff, fs, order)