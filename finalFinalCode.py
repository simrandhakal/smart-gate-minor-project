import cv2
import face_recognition
import easyocr
import pandas as pd
from collections import Counter
import numpy as np
import csv
from datetime import datetime
import asyncio
from telegram import Bot
import serial
import time


authorized = False
# Set the correct port for your Arduino
arduino = serial.Serial(port='COM8', timeout=0)
time.sleep(2)

def vehicleDetection():

    #  Your Telegram bot token
    TELEGRAM_BOT_TOKEN = "enter your token"
    # Your Telegram chat ID
    TELEGRAM_CHAT_ID = 'enter your chat id'

    # Create a Telegram bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)


    # Specify the language for OCR
    ocr_language = 'en'
    authorized_vehicle_detected = False

    # Load the OCR reader with the specified language
    reader = easyocr.Reader([ocr_language])

    # Read data from Excel sheet
    excel_file = 'humanNvehicle/license_value/authorized_plate_value.xlsx'  # Replace with your Excel file name
    df = pd.read_excel(excel_file)

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    csv_filename = current_date + '.csv'
    f = open(csv_filename, 'w+', newline='')
    lnwriter = csv.writer(f)
    lnwriter.writerow(['Plate no', 'Date'])

    # Open the laptop camera (camera index 0)
    cap = cv2.VideoCapture(0)

    async def send_telegram_message(message):
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    min_area = 500
    count = 0
    detected_plates = []




    while count < 10:
        success, img = cap.read()

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for contrast enhancement
        img_gray = cv2.equalizeHist(img_gray)

        # Apply Gaussian blur for noise reduction
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Use the pre-trained Haarcascade classifier for license plate detection
        harcascade = "humanNvehicle/license_value/model/haarcascade_russian_plate_number.xml"
        plate_cascade = cv2.CascadeClassifier(harcascade)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                
                # Extract the region of interest (ROI) containing the license plate
                img_roi = img[y: y + h, x:x + w]

                # Draw green border around the license plate
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the license plate image
                cv2.imwrite(f"humanNvehicle/license_value/plates/scaned_img_{count}.jpg", img_roi)

                # Perform OCR on the license plate image
                result = reader.readtext(f"humanNvehicle/license_value/plates/scaned_img_{count}.jpg", paragraph=True)

                # Check if the result is not empty, does not contain lowercase letters, and store in detected_plates
                if result and not any(char.islower() for char in result[0][1]):
                    detected_plate = result[0][1]
                    detected_plates.append(detected_plate)

                count += 1
                

        # Check if 10 plates have been detected
        if count == 10:
            # Find the plate value with the maximum frequency using Counter
            max_plate = Counter(detected_plates).most_common(1)[0][0]

            # Check if the detected plate is authorized
            if max_plate in df['plate value'].values:
                datatosend = 's'
                arduino.write(datatosend.encode())
                print('Hello')
                print(f"Detected Plate: {max_plate} - Authorized")
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                lnwriter.writerow([max_plate, current_time])
                authorized_vehicle_detected = True
                

            else:
                print(f"Detected Plate: {max_plate} - Unauthorized")
                # Setup asyncio event loop and run the send_telegram_message function
                loop = asyncio.get_event_loop()
                loop.run_until_complete(send_telegram_message(f"Unauthorized vehicle detected at your door = {max_plate}"))
                authorized_vehicle_detected = False  # Reset the flag when an unauthorized person is detected

            # Clear the detected plates for the next iteration
            detected_plates = []


        # Display the result image
        cv2.imshow("Result", img)
        cv2.waitKey(500)  # Add a delay for better visualization

    # # Release the camera and close the OpenCV windows
    # cap.release()
    # cv2.destroyAllWindows()


# Your Telegram bot token
TELEGRAM_BOT_TOKEN = "enter your token"
# Your Telegram chat ID
TELEGRAM_CHAT_ID = 'enter your chat id'

# Create a Telegram bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize the vehicle cascade
vehicle_cascade = cv2.CascadeClassifier('D:/Projects/python/hackathon/humanNvehicle/cars.xml')

# open webcam video stream
cap = cv2.VideoCapture(0)

# Set the video capture dimensions
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Load authorized faces
simran_image = face_recognition.load_image_file("humanNvehicle/copy/photos/simran.jpg")
simran_encoding = face_recognition.face_encodings(simran_image)[0]

stuti_image = face_recognition.load_image_file("humanNvehicle/copy/photos/stuti.jpg")
stuti_encoding = face_recognition.face_encodings(stuti_image)[0]

known_face_encoding = [simran_encoding, stuti_encoding]
known_faces_names = ["Simran", "Stuti"]
students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

csv_filename = current_date + '.csv'
f = open(csv_filename, 'w+', newline='')
lnwriter = csv.writer(f)
lnwriter.writerow(['Name', 'Date'])

video_capture = cv2.VideoCapture(0)

async def send_telegram_message(message):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

authorized_person_detected = False
vehicle_detected_counter = 0
vehicle_detection_threshold = 10 

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # detect vehicles in the image
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

     # Reset the vehicle detection counter if no vehicles are detected
    if len(vehicles) == 0:
        vehicle_detected_counter = 0

    # Draw rectangles and display messages for faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Face Recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = "Unauthorized"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_faces_names[first_match_index]

            face_names.append(name)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            thickness = 3
            lineType = 2

            if name in known_faces_names:
                # Authorized person detected
                cv2.putText(frame, f'{name} Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            (0, 255, 0),  # Green color for authorized persons
                            thickness,
                            lineType)
                 

                if not authorized_person_detected:
                    # Save the authorized person's name and date in the CSV file only once
                    datatosend = 's'
                    arduino.write(datatosend.encode())
                    print('Hello')
                   
                    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    authorized_person_detected = True

            else:
                # Unauthorized person detected
                cv2.putText(frame, 'Unauthorized',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            (0, 0, 255),  # Red color for unauthorized persons
                            thickness,
                            lineType)

                if name not in students:
                    students.append(name)
                    print(students)

                    # Setup asyncio event loop and run the send_telegram_message function
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(send_telegram_message(f"Unauthorized person detected at your door"))
                    authorized_person_detected = False  # Reset the flag when an unauthorized person is detected

    # Draw rectangles and display messages for vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Vehicle Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


        # Increment the vehicle detection counter
        vehicle_detected_counter += 1

        # Print the message when the threshold is reached
        if vehicle_detected_counter >= vehicle_detection_threshold:
            # ya print garnu ko satta ma license detect garna lagauxam hami 

            print("Vehicle Detected!")

            vehicleDetection()
            vehicle_detected_counter = 0  # Reset the counter after printing the message

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # while(authorized):
    #     arduino.write(b's')  # Encode and send '1' as a byte
    #     print("LED turned on")

    

    # Check for 'q' key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# When everything is done, release the capture
cap.release()
video_capture.release()
# Finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
f.close()

