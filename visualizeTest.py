import cv2 as cv
import numpy as np


# i, j, avg_na, avg_nb, time_elapsed, avg_dt, processed, needed

img = np.zeros((415, 440))
cv.putText(img, "i:12", (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
cv.putText(img, "j:412", (135, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

cv.putText(img, "avg na   avg nb", (25, 105), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
cv.putText(img, "1234", (40, 140), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
cv.putText(img, "5678", (195, 140), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

cv.putText(img, "Total Time: 123.123 min", (25, 200), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
cv.putText(img, "Avg dT: 0.123123 sec", (25, 235), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

cv.putText(img, "Progress:", (25, 300), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
cv.rectangle(img, (25, 315), (400, 350), (255, 255, 255), 1)
cv.rectangle(img, (25, 315), (222, 350), (255, 255, 255), -1)
cv.putText(img, "(1123/1234)", (25, 385), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()