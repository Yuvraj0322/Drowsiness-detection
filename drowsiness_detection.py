import cv2
import imutils
import datetime
import os
import numpy as np
from playsound import playsound # type: ignore
import tkinter as tk
from tkinter import filedialog
import sqlite3
import mediapipe as mp

def start_detection():
    # Connect to SQLite database
    conn = sqlite3.connect('drowsiness_data.db')
    c = conn.cursor()

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS drowsiness (
                 id INTEGER PRIMARY KEY,
                 timestamp DATETIME,
                 eye_aspect_ratio REAL
                 )''')

    def eye_aspect_ratio(eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye[0] - eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    def sound_alarm():
        alarm_path = alarm_entry.get()
        if not alarm_path or not os.path.exists(alarm_path):
            print("Alarm sound file not selected or does not exist.")
            return
        playsound(alarm_path)

    # Function to capture screenshot with EAR value displayed
    def capture_screenshot(frame, ear):
        output_folder = screenshot_entry.get()
        if not output_folder or not os.path.isdir(output_folder):
            print("Screenshot folder not selected or does not exist.")
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_folder, f"screenshot_{timestamp}.png")

        # Draw EAR value on the frame
        ear_text = f"EAR: {ear:.2f}"
        cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the frame with EAR value displayed
        cv2.imwrite(filename, frame)

    # Initialize video capture
    vs = cv2.VideoCapture(0)

    # Initialize dlib's face detector and predictor
    
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Set eye aspect ratio threshold
    EAR_THRESHOLD = 0.25

    # Initialize counters
    COUNTER = 0
    ALARM_ON = False

    def process_frame():
        nonlocal COUNTER, ALARM_ON
        ret, frame = vs.read()
        if not ret:
            vs.release()
            conn.close()
            return

        frame = imutils.resize(frame, width=450)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Mediapipe face mesh landmarks for eyes
                # Left eye indices: 33, 160, 158, 133, 153, 144
                # Right eye indices: 362, 385, 387, 263, 373, 380
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                h, w = frame.shape[:2]
                left_eye = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices])
                right_eye = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices])

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= 30:
                        if not ALARM_ON:
                            ALARM_ON = True
                            sound_alarm()
                            capture_screenshot(frame, ear)
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Insert data into SQLite database
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        c.execute("INSERT INTO drowsiness (timestamp, eye_aspect_ratio) VALUES (?, ?)", (timestamp, ear))
                        conn.commit()
                else:
                    COUNTER = 0
                    ALARM_ON = False

                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vs.release()
            conn.close()
            cv2.destroyAllWindows()
        else:
            root.after(10, process_frame)

    process_frame()

def select_alarm_file():
    alarm_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3")])
    alarm_entry.delete(0, tk.END)
    alarm_entry.insert(0, alarm_path)

def select_screenshot_folder():
    output_folder = filedialog.askdirectory()
    screenshot_entry.delete(0, tk.END)
    screenshot_entry.insert(0, output_folder)

# Create main application window
root = tk.Tk()
root.title("Drowsiness Detection System")

# Styling variables
label_color = '#FFFFFF'
input_bg = '#FFFFFF'
input_fg = '#6A5ACD'
button_bg = '#006699'
button_fg = '#FFFFFF'

# GUI elements for alarm and screenshot settings
alarm_label = tk.Label(root, text="Select Alarm Sound:", bg='#333333', fg=label_color)
alarm_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

alarm_entry = tk.Entry(root, width=40, bg=input_bg, fg=input_fg)
alarm_entry.grid(row=0, column=1, padx=10, pady=10)

alarm_button = tk.Button(root, text="Browse", bg=button_bg, fg=button_fg, command=select_alarm_file)
alarm_button.grid(row=0, column=2, padx=10, pady=10)

screenshot_label = tk.Label(root, text="Select Screenshot Folder:", bg='#333333', fg=label_color)
screenshot_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

screenshot_entry = tk.Entry(root, width=40, bg=input_bg, fg=input_fg)
screenshot_entry.grid(row=1, column=1, padx=10, pady=10)

screenshot_button = tk.Button(root, text="Browse", bg=button_bg, fg=button_fg, command=select_screenshot_folder)
screenshot_button.grid(row=1, column=2, padx=10, pady=10)

start_button = tk.Button(root, text="Start Detection", bg=button_bg, fg=button_fg, command=start_detection)
start_button.grid(row=2, column=1, padx=10, pady=20)

# Run the application
root.mainloop()
