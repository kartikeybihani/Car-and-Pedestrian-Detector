import cv2

# our image
img_file = "Car Image.jpg"
video = cv2.VideoCapture("dashcam2.mp4")

# pre-trained classifier file
car_tracker_file = "car_detector.xml"
pedetrian_tracker_file = "pedestrian_tracker.xml"

# create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedetrian_tracker = cv2.CascadeClassifier(pedetrian_tracker_file)

# works until car stops
while True:
    # read the current frame
    (read_succesfull, frame) = video.read()

    if read_succesfull:
        # must convert to grysacale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedetrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangle around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x + 1, y + 2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw rectangle around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Car Detector', frame)

    # key = cv2.waitKey(1)
    # stop on q press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#
video.release()
cv2.destroyAllWindows()

"""

# create opencv image
img = cv2.imread(img_file)

# convert to grayscale to reduce data of colors
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect car:
cars = car_tracker.detectMultiScale(black_n_white)

# print(cars) ## Coordinates

# draw rectangle around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+w), (0, 0, 255), 2)

# Display the image
cv2.imshow('Car Detector', img)

# Don't autoclose
cv2.waitKey()

"""

print("Code completed")
