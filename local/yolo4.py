import cv2
import os
import json
import time
import numpy as np
from datetime import datetime

# Configuration using environment variables
FRAME_RATE = 30
SECONDS_BETWEEN_IMAGES = 3
DELAY_IMAGES = FRAME_RATE * SECONDS_BETWEEN_IMAGES
IOT_ID = 5
VIDEO_PATH = "wisenet_dataset/video_sets/set_4/"
MODEL_CFG = "yolov4_tiny.cfg"
MODEL_WEIGHTS = "yolov4-tiny.weights"
COCO_NAMES = "coco.names"
SAVE_DIR = "yolo4"
CONFIDENCE_THRESHOLD = 0.6

# Load YOLOv3 model
classes = open(COCO_NAMES).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


os.makedirs(SAVE_DIR, exist_ok=True)

def send_frame(frame):
    try:
        _, encoded_image = cv2.imencode(".jpg", frame)
        message = {
            "iot_id": IOT_ID,
            "frame": encoded_image.tobytes().decode("latin1"),
        }
        receive_frame(json.dumps(message))
    except Exception as e:
        print(f"Error sending frame: {e}")
        time.sleep(5)
        send_frame(frame)

def send_images():
    try:
        videos = os.listdir(VIDEO_PATH)
        for video in videos:
            cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))
            counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                counter += 1
                if counter > DELAY_IMAGES:
                    print("Sending frame...")
                    send_frame(frame)
                    counter = 0
                    time.sleep(2)
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error sending images: {e}")
        time.sleep(5)

def receive_frame(message):
    try:
        body = json.loads(message)
        frame_data = body["frame"].encode("latin1")
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        iot_id = body["iot_id"]
        print(f"Received image from IoT {iot_id}.")
        process_yolo(frame, iot_id)
    except Exception as e:
        print(f"Error receiving frame: {e}")

def process_yolo(frame, iot_id):
    try:
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > CONFIDENCE_THRESHOLD:
                    x, y, w, h = detection[:4] * np.array([W, H, W, H])
                    p0 = int(x - w // 2), int(y - h // 2)
                    boxes.append([*p0, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
        
        # Check if indices is not empty and flatten it if it is a tuple
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices[0]

            for i in indices:
                x, y, w, h = boxes[i]
                if classIDs[i] == 0:  # Class 0: Person
                    cropped_frame = frame[y:y+h, x:x+w]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(SAVE_DIR, f"person_{timestamp}.jpg")
                    cv2.imwrite(filename, cropped_frame)
                    print(f"Saved cropped image to {filename}")
    except Exception as e:
        print(f"Error processing YOLO: {e}")


def main():
    print("Starting the local process...")
    send_images()

if __name__ == "__main__":
    main()