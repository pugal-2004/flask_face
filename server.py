from flask import Flask, request, jsonify
import cv2
import os
import pandas as pd
import logging
from deepface import DeepFace

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure students.csv is accessible
students_csv = os.path.join(BASE_DIR, "students.csv")

if os.path.exists(students_csv):
    students_data = pd.read_csv(students_csv)
    logging.info(f"Loaded student data from {students_csv}")
    print(f"Loaded student data from {students_csv}")
else:
    logging.error(f"Error: {students_csv} not found!")
    print(f"Error: {students_csv} not found!")
    students_data = pd.DataFrame(columns=['regno', 'name', 'image_path'])  # Empty DataFrame to prevent crashes

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def recognize_faces(image_path):
    print(f"Processing image: {image_path}")  # Print when processing starts
    logging.info(f"Processing image: {image_path}")
    
    present_students = []
    absent_students = list(students_data['regno'])
    unknown_faces = 0

    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Failed to load image. Invalid file format.")
        print("Error: Invalid image file")
        return {"error": "Invalid image file"}

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Detected {len(faces)} faces.")
    logging.info(f"Detected {len(faces)} faces.")

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_path = os.path.join(BASE_DIR, "temp_detected_face.jpg")
        cv2.imwrite(face_path, face)

        matched = False
        for _, student in students_data.iterrows():
            student_image_path = student['image_path']
            if os.path.exists(student_image_path):
                try:
                    result = DeepFace.verify(face_path, student_image_path, model_name='VGG-Face', enforce_detection=False)
                    print(f"Comparing with {student['name']}: {result}")
                    logging.info(f"Comparing with {student['name']}: {result}")

                    if result.get("verified", False):
                        present_students.append(student['regno'])
                        if student['regno'] in absent_students:
                            absent_students.remove(student['regno'])
                        matched = True
                        break
                except Exception as e:
                    logging.error(f"Error processing {student['name']}: {e}")
                    print(f"Error processing {student['name']}: {e}")

        if not matched:
            unknown_faces += 1

    attendance_result = {
        "present_students": present_students,
        "absent_students": absent_students,
        "unknown_faces": unknown_faces
    }
    
    print("Final attendance result:", attendance_result)  # Print final attendance result
    return attendance_result

@app.route('/')
def home():
    return jsonify({"message": "Face Recognition API is running!"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.warning("No file uploaded.")
        print("No file uploaded.")  # Print to track missing files
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    print("File uploaded successfully:", file.filename)  # Print filename when uploaded
    logging.info(f"File {file.filename} uploaded successfully.")

    result = recognize_faces(image_path)
    print("Recognition result:", result)  # Print recognition result

    logging.info(f"Recognition result: {result}")
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
