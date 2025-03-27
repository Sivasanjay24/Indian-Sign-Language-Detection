import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Lower the detection confidence threshold
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1)

# Create labels_dict dynamically
import string
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}


def landmarks_to_image(landmarks, image_size=(100, 100)):
    # Create a blank image
    image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    # Scale landmarks to fit the image
    scaled_landmarks = []
    for landmark in landmarks:
        x = int(landmark.x * image_size[0])
        y = int(landmark.y * image_size[1])
        scaled_landmarks.append((x, y))

    # Draw landmarks on the image
    for (x, y) in scaled_landmarks:
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)  # Draw a white dot

    return image

try:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            print("No hand landmarks detected.")
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Clear x_ and y_ for each hand
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Ensure x_ and y_ are not empty
            if not x_ or not y_:
                print("Warning: No landmarks detected.")
                continue

            # Convert landmarks to an image
            landmark_image = landmarks_to_image(hand_landmarks.landmark)

            # Flatten the image
            data_aux = landmark_image.flatten()

            # Ensure the input has the correct shape
            if len(data_aux) == 10000:  # Example: 100x100 image
                prediction = model.predict([data_aux])
                predicted_character = labels_dict[prediction[0]]  # Use prediction directly
            else:
                print("Warning: Input data has incorrect shape.")
                continue

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()