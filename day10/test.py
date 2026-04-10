from pyzbar.pyzbar import decode
import cv2

img = cv2.imread('orange.jpg')

decoded = decode(img)

for d in decoded:
    print(d.data.decode('utf-8'))