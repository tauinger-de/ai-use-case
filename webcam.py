import cv2

# Initialize video capture object with webcam ID (usually 0)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

# Capture frame-by-frame
ret, frame = cap.read()

if ret:
    # Display captured frame
    cv2.imshow('Webcam Capture', frame)

    # Wait for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Capture an image when 's' key is pressed
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured and saved as captured_image.jpg")

# Release capture object and close windows
cap.release()
cv2.destroyAllWindows()
