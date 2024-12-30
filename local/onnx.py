import cv2
import os
import json
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub

#Copied from IoT_Device.py
frame_rate = 30 # Framerate of the videos
seconds_between_images = 3 # 3 seconds between each image 
delay_images = frame_rate * seconds_between_images # 3 seconds delay between each image

iot_id = 5
# Load a pre-trained SSD MobileNet model for object detection
detector_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")  # SSD MobileNet

# Define the directory to save images
save_dir = "persons"
save_dir = "SSDmodel"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

def process_person_detection(frame, iot_id):
    try:
        # Convert the frame to RGB (TensorFlow models expect RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = tf.convert_to_tensor(frame_rgb)

        # Run inference on the frame
        result = detector_model(frame_rgb[None, ...])  # Add batch dimension

        # Post-process the result
        boxes = result["detection_boxes"].numpy()  # Shape [1, num_boxes, 4]
        scores = result["detection_scores"].numpy()  # Shape [1, num_boxes]
        classes = result["detection_classes"].numpy()  # Shape [1, num_boxes]

        person_detected = False
        #i = 0
        for idx in range(len(scores[0])):
            if classes[0][idx] == 1:  # Class 1 is person in the COCO dataset
                if scores[0][idx] > 0.55:  # Threshold for confidence score
                    person_detected = True
                    #i += 1

                    # Extract bounding box
                    y_min, x_min, y_max, x_max = boxes[0][idx]
                    x_min, y_min, x_max, y_max = map(int, [x_min * frame.shape[1], y_min * frame.shape[0],
                                                           x_max * frame.shape[1], y_max * frame.shape[0]])

                    # Crop the detected person from the frame
                    cropped_frame = frame[y_min:y_max, x_min:x_max]

                    # Generate a unique filename (e.g., timestamp-based)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join("SSDmodel", f"person_{timestamp}.jpg")

                    # Save the cropped image
                    cv2.imwrite(filename, cropped_frame)
                    print(f"Saved cropped image to {filename}")

        if not person_detected:
            print("No person detected.", flush=True)

    except Exception as e:
        print(f"Error processing detection: {e}", flush=True)

def send_frame(frame):
    try:
        _, encoded_image = cv2.imencode(".jpg", frame)
        message = {
            "iot_id": iot_id,
            "frame": encoded_image.tobytes().decode("latin1"),  # Convert bytes to string
        }
        message = json.dumps(message)
        receive_frame(message)
    except Exception as e:
        print(f"Error sending frame: {e}", flush=True)
        print("Retrying to send frame in 5 seconds...", flush=True)
        time.sleep(5)
        send_frame(frame)  # Retry sending the frame after 5 seconds


def receive_frame(message):
    """
    Receive the image
    """
    body = json.loads(message)
    frame_data = body["frame"].encode("latin1")  # Decode string to bytes
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    iot_id = body["iot_id"]

    print(f"Received image from IoT {iot_id}.", flush=True)

    # Call the person detection function
    process_person_detection(frame, iot_id)


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


def main():
    print("Starting the local process...", flush=True)
    send_images()


if __name__ == "__main__":
    main()
