from deepface import DeepFace
import cv2
import pandas as pd
import csv
import os
import pywhatkit as kit
from datetime import datetime, timedelta

# Detect and align faces in the classroom image for better accuracy
def detect_and_align_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    aligned_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]  # Crop the face from the image
        aligned_face = cv2.resize(face, (224, 224))  # Resize the face to the input size for DeepFace models
        aligned_faces.append(aligned_face)
    
    return aligned_faces

# Load student data from the CSV file
def load_students_data(csv_file):
    students_data = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if os.path.exists(row['image_path']):  # Check if image exists
                students_data.append({
                    'regno': row['regno'],
                    'name': row['name'],
                    'image_path': row['image_path']
                })
    return students_data

# Recognize faces in the classroom image using an ensemble of models
def recognize_faces_deepface(class_image_path, students_data):
    present_students = []
    absent_students = [student['regno'] for student in students_data]
    unknown_faces = 0
    
    # Detect and align faces in the classroom image
    aligned_faces = detect_and_align_faces(class_image_path)
    
    if not aligned_faces:
        print("No faces detected in the classroom image.")
        return present_students, absent_students, unknown_faces

    # Loop through each detected face in the classroom image
    for detected_face in aligned_faces:
        temp_face_path = 'temp_detected_face.jpg'
        cv2.imwrite(temp_face_path, detected_face)

        matched = False
        # Loop through the student data and compare each student image with the detected face
        for student in students_data:
            try:
                # Use multiple models for verification
                result_vgg = DeepFace.verify(
                    img1_path=temp_face_path, 
                    img2_path=student['image_path'], 
                    model_name='VGG-Face',
                    enforce_detection=False
                )

                result_facenet = DeepFace.verify(
                    img1_path=temp_face_path, 
                    img2_path=student['image_path'], 
                    model_name='Facenet',
                    enforce_detection=False
                )

                # If either model verifies the face, mark the student as present
                if result_vgg["verified"] or result_facenet["verified"]:
                    if student['regno'] not in present_students:
                        present_students.append(student['regno'])
                        if student['regno'] in absent_students:
                            absent_students.remove(student['regno'])
                    matched = True
                    break
            except Exception as e:
                print(f"Error processing {student['name']}: {e}")
        
        if not matched:
            unknown_faces += 1
    
    return present_students, absent_students, unknown_faces

# Function to create a summary message with attendance details
def create_attendance_summary(present_students, absent_students, unknown_faces):
    present_str = ', '.join(present_students) if present_students else "None"
    absent_str = ', '.join(absent_students) if absent_students else "None"
    
    summary = f"""
    Attendance Report:
    
    Present Students:
    {present_str}
    
    Absent Students:
    {absent_str}
    
    Unknown Faces Detected: {unknown_faces}
    """
    
    return summary

# Function to send WhatsApp message with the attendance summary
def send_whatsapp_message(phone_number, message):
    try:
        # Get current time and add 1 minute to send message
        current_time = datetime.now()
        future_time = current_time + timedelta(minutes=1)
        hours = future_time.hour
        minutes = future_time.minute

        # Send message
        kit.sendwhatmsg(phone_number, message, hours, minutes)
        print("Message sent successfully!")
    except Exception as e:
        print(f"Error sending message: {e}")

# Main function
def main():
    # Load student data from the CSV file (Modify the CSV path accordingly)
    students_csv = 'students.csv'  # Path to students CSV file
    students_data = load_students_data(students_csv)

    # Classroom image file (Modify the image path accordingly)
    classroom_image = 'news.jpeg'  # Path to uploaded classroom image
    
    # Recognize faces and get attendance (Present and Absent students)
    present_students, absent_students, unknown_faces = recognize_faces_deepface(classroom_image, students_data)

    # Create attendance summary
    summary_message = create_attendance_summary(present_students, absent_students, unknown_faces)

    # Send the attendance summary via WhatsApp
    phone_number = '+918056367687'  # Ensure the phone number is in international format
    send_whatsapp_message(phone_number, summary_message)

    # Output to the console for verification
    print("Present Students:", present_students)
    print("Absent Students:", absent_students)
    print("Unknown Faces Detected:", unknown_faces)

if __name__ == "__main__":
    main()