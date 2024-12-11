import cv2
import os
import json
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

"""
This program should show the image processsing from the IoT device to the edge layer.
Also, it should save these images in the persons folder.
"""

#TODO: Improve programm by making it more flexible with using enviroment variables

#Copied from IoT_Device.py
frame_rate = 30 # Framerate of the videos
seconds_between_images = 3 # 3 seconds between each image 
delay_images = frame_rate * seconds_between_images # 3 seconds delay between each image

iot_id = 5
model = YOLO("yolov8n.pt")  # Hope this is standart YOLO weights file

save_dir = "persons"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

def send_frame(frame):
    try:
        _, encoded_image = cv2.imencode(".jpg", frame)
        message = {
            "iot_id": iot_id,
            "frame": encoded_image.tobytes().decode("latin1"),  # Convert bytes to string
        }
        message = json.dumps(message)
        # print(f"Frame decoded: {message}", flush=True)
        receive_frame(message)
    except Exception as e:
        print(f"Error sending frame: {e}", flush=True)
        print("Retrying to send frame in 5 seconds...", flush=True)
        time.sleep(50)
        send_frame(frame)  # Retry sending the frame after 5 seconds

def send_images():
    path_to_video = f"wisenet_dataset/video_sets/set_1/"
    for i in range(1, 2):
        try:
            videos = os.listdir(path_to_video)
            for video in videos:
                cap = cv2.VideoCapture(path_to_video + video)
                counter = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    counter += 1
                    if counter > delay_images:
                        print("Sending frame...", flush=True)
                        send_frame(frame)
                        counter = 0
                        time.sleep(2)
                cap.release()
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error sending images: {e}", flush=True)
            print("Retrying image sending in 5 seconds...", flush=True)
            time.sleep(5)

def receive_frame(message):
    """
    Receive the image
    """
    body = json.loads(message)
    frame_data = body["frame"].encode("latin1")  # Decode string to bytes
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    iot_id = body["iot_id"]

    print(f"Received image from IoT {iot_id}.", flush=True)

    process_yolo(frame, iot_id)

def process_yolo(frame, iot_id):
    try:
        results = model(frame)
        # print(f"YOLO results: {results}", flush=True)  Very few info from this. Main info is from the boxes

        detections = results[0].boxes.data
        i = 0
        for dection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = dection    #Either use these variables or array index
            if dection[5] == 0: # 0 is the class for person
                if dection[4] > 0.5:
                    person_detected = True
                    # results[0].save(f"persons/{i}.jpg")       #Is the whole image
                    i += 1

                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    cropped_frame = frame[y_min:y_max, x_min:x_max]

                    # Display or process the cropped frame
                    # cv2.imshow("Cropped Frame", cropped_frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Generate a unique filename (e.g., timestamp-based)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(save_dir, f"person_{timestamp}.jpg")

                    # Save the cropped image
                    cv2.imwrite(filename, cropped_frame)
                    print(f"Saved cropped image to {filename}")
        
        
        # Old version where we read out all the infos from the detections
        # person_detected = any(int(detection[-1]) == 0 for detection in detections)

        # if person_detected:
        #     print(f"Person detected for IoT device {iot_id}.", flush=True)
        #     for result in results:
        #         print(f"YOLO result:", flush=True)
        #         result.show()

        # for result in results:
        #     print(f"YOLO result:", flush=True)
        #     result.show()
    except Exception as e:
        print(f"Error processing YOLO: {e}", flush=True)


def main():
    print("Starting the local process...", flush=True)
    send_images()


if __name__ == "__main__":
    main()