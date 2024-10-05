import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

# Load the palm cascade file
palm_cascade = cv2.CascadeClassifier('palm.xml')
if palm_cascade.empty():
    print("Error loading cascade file. Please check the path to palm.xml.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    palms = palm_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in palms:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Palm Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
