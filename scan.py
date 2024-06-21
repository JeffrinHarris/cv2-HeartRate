import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read Video from file
cap = cv2.VideoCapture('IMG_1270.MOV')

# Skin Threshold
min_HSV = np.array([0, 50, 150],np.uint8)
max_HSV = np.array([20, 255, 255],np.uint8)

avg_blue = []
avg_green = []
avg_red = []

numbers = []
i = 1

# Iterate every frame at 30fps
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    mask = cv2.inRange(hsv, min_HSV, max_HSV)

    skin = cv2.bitwise_and(frame, frame, mask=mask)

    # Getting Mean of the skin
    mean = cv2.mean(frame, mask=mask)
    numbers.append(i)
    i += 1


    # mean should only have 2 decimals
    mean = [round(m, 2) for m in mean]

    avg_blue.append(mean[0])
    avg_green.append(mean[1])
    avg_red.append(mean[2])

    # mean value is in text in frame
    cv2.putText(skin, str(mean), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Skin Detection', skin)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Plotting the values
plt.figure(figsize=(5, 10))
plt.plot(numbers, avg_red, label='Red', color='red', marker='o')
plt.plot(numbers, avg_green, label='Green', color='green', marker='o')
plt.plot(numbers, avg_blue, label='Blue', color='blue', marker='o')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Average RGB Value')
plt.title('Average RGB Values vs Time')
plt.legend()
plt.grid(True)
plt.show()