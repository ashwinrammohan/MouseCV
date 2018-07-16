import numpy as np
import cv2 as cv
import wholeBrain as wb
import matplotlib.pyplot as plt
import math

def load_mp4(vid_name):
	cap = cv.VideoCapture("Assets/" + vid_name + ".mp4")
	frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

	vid = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

	fc = 0
	ret = True

	while (True):
		result = cap.read()
		if not(result[0]):
			break

		vid[fc] = result[1][:,:,0]
		fc += 1

	cap.release()
	return vid

extra_space = 0.05 #10% extra on the bounding box (5% on each side)
max_ratio = 0.25
vid = load_mp4("greyscale_brain")
box_x1 = int(vid.shape[2]/2)
box_y1 = int(vid.shape[1]/2)
box_x2 = box_x1 + 1
box_y2 = box_y1 + 1
frame = 0
done = False
while frame < vid.shape[0] - 1:
	retval, result = cv.threshold(vid[frame], 150, 255, cv.THRESH_BINARY);
	blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		rect = cv.boundingRect(contour)
		ratio = rect[2]/rect[3]
		if ratio < (1 - max_ratio) or ratio > (1 + max_ratio):
			continue

		x1 = rect[0]
		y1 = rect[1]
		x2 = rect[0] + rect[2]
		y2 = rect[1] + rect[3]

		if x1 < box_x1:
			box_x1 = x1
		if x2 > box_x2:
			box_x2 = x2
		if y1 < box_y1:
			box_y1 = y1
		if y2 > box_y2:
			box_y2 = y2

	cv.drawContours(vid[frame], contours, -1, (255,255,255), -1)
	cv.rectangle(vid[frame], (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

	cv.imshow("Frame threshold", result)
	cv.imshow("Original", vid[frame])
	if cv.waitKey(0) == ord('q'):
		done = True
		break
	frame += 1

box_x1 = int(box_x1 * (1 - extra_space))
box_y1 = int(box_y1 * (1 - extra_space))
box_x2 = int(box_x2 * (1 + extra_space))
box_y2 = int(box_y2 * (1 + extra_space))
cv.rectangle(vid[frame], (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
cv.imshow("Original", vid[frame])

print(box_x1, box_y1)
print(box_x2, box_y2)
while not(done):
	if cv.waitKey(50) == ord('q'):
		break

cv.destroyAllWindows()
