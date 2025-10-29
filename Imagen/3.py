import cv2 as cv
import numpy as np

img = cv.imread("/home/likcos/Imágenes/tr.png", 1)
print(img.shape[:2])
imgn = np.zeros(img.shape[:2], np.uint8)
b,g,r =cv.split(img)
#img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#img3 =cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgb = cv.merge([b, imgn, imgn])
imgg = cv.merge([imgn, g, imgn])
imgr = cv.merge([imgn, imgn, r])
grb = cv.merge([imgn, r, b])


cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)
cv.imshow('img',img)
#cv.imshow('img2', img2)
cv.imshow('imgb', imgb )
cv.imshow('imgg', imgg )
cv.imshow('imgr', imgr )
cv.imshow('grb', grb )



cv.waitKey()
cv.destroyAllWindows()