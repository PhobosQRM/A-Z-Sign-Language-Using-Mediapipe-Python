"""
Requirements:
  pip install opencv-python mediapipe numpy
  pip install scikit-learn joblib

How it works:
  - Press a letter key (a..z) to capture a sample for that letter (one hand only).
  - Press '1' to save dataset to features.npy and labels.npy.
  - Press '2' to train a KNN classifier from current dataset.
  - Press '3' to run real-time recognition using the trained model.
  - Press '4' to go back to collect mode.
  - Press ESC to quit.
"""

"""
ASL A-Z Sign Recognizer with Bounding Box and Accuracy Score
"""

import cv2
import mediapipe as mp
import numpy as np
import string
import os

try:
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATASET_FEAT = 'features.npy'
DATASET_LABEL = 'labels.npy'
MODEL_FILE = 'knn_model.joblib'

def lm_to_vector(landmarks, image_w, image_h):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0].copy()
    coords = coords - wrist
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs
    return coords.flatten()

# -------- NEW ACCURACY SCORE CALCULATION -------- #
def compute_similarity_score(feat, X, y, predicted_label):
    X = np.array(X)
    y = np.array(y)

    same_label_samples = X[y == predicted_label]

    if len(same_label_samples) == 0:
        return 0.0

    distances = np.linalg.norm(same_label_samples - feat, axis=1)

    avg_dist = np.mean(distances)
    scaled = max(0.0, 1 - avg_dist)
    return scaled * 100.0


class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict_single(self, x):
        dists = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(dists)[: self.k]
        votes = self.y[idx]
        values, counts = np.unique(votes, return_counts=True)
        confidence = 1 - (np.min(dists) / np.max(dists)) if np.max(dists) > 0 else 1.0
        return values[np.argmax(counts)], confidence

    def predict(self, X):
        return np.array([self.predict_single(x)[0] for x in X])


def load_dataset():
    if os.path.exists(DATASET_FEAT) and os.path.exists(DATASET_LABEL):
        X = np.load(DATASET_FEAT)
        y = np.load(DATASET_LABEL)
        print(f"Loaded dataset: {len(X)} samples")
        return list(X), list(y)
    return [], []


def save_dataset(X, y):
    np.save(DATASET_FEAT, np.array(X))
    np.save(DATASET_LABEL, np.array(y))
    print(f"Saved dataset: {len(X)} samples")


def train_model(X, y, k=5):
    if len(X) == 0:
        print("No data to train on. Capture samples first.")
        return None
    if SKLEARN_AVAILABLE:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        print("Trained sklearn KNN and saved.")
        return clf
    else:
        clf = SimpleKNN(k=k)
        clf.fit(X, y)
        print("Trained SimpleKNN")
        return clf


def load_model():
    if SKLEARN_AVAILABLE and os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None


def main():
    X, y = load_dataset()
    clf = load_model()

    LETTER_KEYS = {ord(c): c.upper() for c in string.ascii_lowercase}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        mode = 'collect'
        print("Controls: a..z capture, 1 save, 2 train, 3 run, 4 collect, ESC quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            img = cv2.flip(frame, 1)
            h, w, _ = img.shape

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            display_text = f"Mode: {mode} | Samples: {len(X)}"

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

                # ---- Improved Bounding Box ---- #
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x_min, x_max = min(xs) - 15, max(xs) + 15
                y_min, y_max = min(ys) - 15, max(ys) + 15

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                feat = lm_to_vector(lm.landmark, w, h)

                # ---- RUN MODE: Predicting ---- #
                if mode == "run" and clf is not None:
                    try:
                        if not SKLEARN_AVAILABLE:
                            pred, conf = clf.predict_single(feat)
                            percentage = conf * 100
                        else:
                            pred = clf.predict([feat])[0]
                            percentage = compute_similarity_score(feat, X, np.array(y), pred)

                        cv2.putText(img, f"{pred} {percentage:.2f}%", 
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 3)

                    except Exception as e:
                        print("Prediction error:", e)

            cv2.putText(img, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            img_big = cv2.resize(img, (900, 640))
            cv2.imshow("ASL A-Z", img_big)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            # Save dataset
            if key == ord("1"):
                save_dataset(X, y)

            # Train
            elif key == ord("2"):
                clf = train_model(X, y)

            # Run
            elif key == ord("3"):
                if clf is None:
                    print("Train first.")
                else:
                    mode = "run"

            # Back to collect
            elif key == ord("4"):
                mode = "collect"

            # Capture letters
            elif key in LETTER_KEYS and mode == "collect":
                if res.multi_hand_landmarks:
                    feat = lm_to_vector(res.multi_hand_landmarks[0].landmark, w, h)
                    X.append(feat)
                    y.append(LETTER_KEYS[key])
                    print(f"Captured '{LETTER_KEYS[key]}'")
                else:
                    print("No hand detected.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
