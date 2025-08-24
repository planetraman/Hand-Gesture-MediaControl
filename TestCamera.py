import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    cv2.imshow("Img", img)

    key = cv2.waitKey(1) & 0xFF
    # Press 'q' OR close window to exit
    if key == ord('q') or cv2.getWindowProperty("Img", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
