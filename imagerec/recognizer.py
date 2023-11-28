import cv2
import numpy as np

image_path = "../resources/img_resources/morning-coffee-0000.jpg"
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 218])
upper = np.array([157, 54, 255])
mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dilate = cv2.dilate(mask, kernel, iterations=5)

cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

letters = []
for idx, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    # if hierarchy[0][idx][3] == 0:
    cv2.rectangle(image, (x, y), (x + w, y + h), (70, 0, 0), 1)
    letter_crop = mask[y:y + h, x:x + w]
    # print(letter_crop.shape)

    # Resize letter canvas to square
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        # Enlarge image top-bottom
        # ------
        # ======
        # ------
        y_pos = size_max // 2 - h // 2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    elif w < h:
        # Enlarge image left-right
        # --||--
        x_pos = size_max // 2 - w // 2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop
    else:
        letter_square = letter_crop

    # Resize letter to 28x28 and add letter and its X-coordinate
    letters.append((x, w, cv2.resize(letter_square, (100, 100), interpolation=cv2.INTER_AREA)))



cv2.imshow("orig", image)
cv2.imshow("hsv", hsv)
cv2.imshow("mask", mask)
cv2.imshow("0", letters[0][2])
cv2.imshow("1", letters[5][2])
cv2.imshow("2", letters[2][2])
cv2.imshow("3", letters[3][2])
cv2.imshow("4", letters[4][2])
cv2.waitKey(0)
