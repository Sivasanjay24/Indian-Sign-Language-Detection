import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        try:
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Error reading image {img_path}. Skipping...")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to square if needed
            img_rgb_resized = cv2.resize(img_rgb, (224, 224))

            results = hands.process(img_rgb_resized)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

            else:
                print(f"No landmarks detected in {img_path}")
                continue  # Skip the image if no landmarks are found
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue  # Skip the image if there's an error

        print(f"Processed {img_path}")  # Show progress

# Save data to pickle
try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data saved to 'data.pickle'")
except Exception as e:
    print(f"Error saving pickle file: {e}")
