import cv2
import pyttsx3
from deepface import DeepFace

# Load video from webcam
cap = cv2.VideoCapture(0)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

while True:
    key, img = cap.read()
    if not key:
        print("Failed to grab frame")
        break

    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Extract emotion
    emotion = results[0]['dominant_emotion']

    # Put emotion text on frame
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak the emotion every time
    engine.say(f"{emotion}")
    engine.runAndWait()

    # Show window
    cv2.imshow("Emotion Recognition", img)

    # Close window on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Close when clicking the window CLOSE (X) button
    if cv2.getWindowProperty("Emotion Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
