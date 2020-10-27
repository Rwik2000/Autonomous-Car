import cv2

img = cv2.imread('20.png')
f = open('coe.coe','w')
f.write('memory_initialization_radix=2;\n')
f.write('memory_initialization_vector=\n')
for row in img:
    for pixel in row:
        # print(pixel)
        # pixelString = ''
        pixelString = str(bin(pixel[0])[2:]).zfill(8)
        pixelString += str(bin(pixel[1])[2:]).zfill(8)
        pixelString += str(bin(pixel[2])[2:]).zfill(8)
        # print(pixelString)
        f.write(pixelString + ',\n')
        # print(str(bin(pixel)[2:]).zfill(8))

f.write(';')