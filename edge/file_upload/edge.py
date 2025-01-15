import threading
import boto3
import json
import time
import cv2
import os
import base64
import numpy as np
import time
import botocore.exceptions
from datetime import datetime


"""
This programm receives images from the IoT devices, processes them with YOLO, and sends them to the cloud.
This programm requires environment variables to be set:
- CLOUD_LAMBDA_FUNCTION: the name of the cloud Lambda function
- IMAGES_QUEUE_NAME: the name of the images SQS queue
- ALARM_QUEUE_NAME: the name of the alarm SQS queue
- REGION_NAME: the AWS region name
"""


# Configuration for the EC2
REGION_NAME = "us-east-1"
CLOUD_LAMBDA_FUNCTION = "cloud_Lambda"  #TODO: Set the actual Lambda function name
IMAGES_QUEUE_NAME = "images"
ALARM_QUEUE_NAME = "alarm"

# Congiguration for YOLO
MODEL_CFG = "yolov4-tiny.cfg"
MODEL_WEIGHTS = "yolov4-tiny.weights"
COCO_NAMES = "coco.names"
SAVE_DIR = "yolo4"
CONFIDENCE_THRESHOLD = 0.6


# Initialize AWS services
sqs = boto3.resource("sqs", region_name=REGION_NAME)
lambda_client = boto3.client("lambda", region_name=REGION_NAME)


#Function to get the queue
def get_queue(queue_name):
    """Retry logic to fetch the SQS queue."""
    sqs = boto3.resource("sqs", region_name="us-east-1")
    while True:
        try:
            print(f"Attempting to fetch queue: {queue_name}", flush=True)
            return sqs.get_queue_by_name(QueueName=queue_name)
        except botocore.exceptions.ClientError as e:
            print(f"Failed to fetch queue {queue_name}: {e}", flush=True)
            print(f"Retrying in 5 seconds...", flush=True)
            time.sleep(5)

# connect to the queues
images_queue = get_queue(IMAGES_QUEUE_NAME)
alarm_queue = get_queue(ALARM_QUEUE_NAME)

# Initialize YOLO
classes = open(COCO_NAMES).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

os.makedirs(SAVE_DIR, exist_ok=True)

def process_yolo(frame, iot_id):
    """
    Run YOLO filtering on the frame and send valid images to the cloud Lambda function.
    Also start a thread for sending and receiving the message to the cloud.
    """
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
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices[0]
            for i in indices:
                x, y, w, h = boxes[i]
                if classIDs[i] == 0:  # Class 0: Person
                    cropped_frame = frame[y:y+h, x:x+w]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    print(f"Person detected by YOLO for IoT device {iot_id}. Sending to cloud.", flush=True)
                    # Send the cropped image to the cloud
                    threading.Thread(target=send_to_cloud, args=(cropped_frame, iot_id)).start()
    
                    # Save the cropped image
                    filename = os.path.join(SAVE_DIR, f"person_{timestamp}.jpg")
                    cv2.imwrite(filename, cropped_frame)
                    print(f"Saved cropped image to {filename}")
                else:
                    print(f"No person detected in the image from IoT device {iot_id}. Skipping.", flush=True)
    except Exception as e:
        print(f"Error in YOLO processing: {e}", flush=True)


def send_to_cloud(frame, iot_id):
    """
    Send the image to the cloud Lambda function for further processing and waits for the response to (not) send alarm.
    """
    try:
        _, encoded_image = cv2.imencode(".jpg", frame)
        payload = {
            "iot_id": iot_id,
            "image": base64.b64encode(encoded_image).decode("utf-8")    # Base64 -> UTF-8 String
        }

        response = lambda_client.invoke(
            FunctionName=CLOUD_LAMBDA_FUNCTION,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )

        response_payload = json.loads(response["Payload"].read())   #TODO: Check for response format
        if response_payload.get("status") == "unknown":
            trigger_alarm(iot_id)
        elif response_payload.get("status") == "error":
            print(f"Error processing image for IoT device {iot_id}.", flush=True)
            print(f"Error message: {response_payload.get('error')}", flush=True)
        elif response_payload.get("status") == "known":
            print(f"Known person recognized for IoT device {iot_id}.", flush=True)
        elif response_payload.get("status") == "no_face":
            print(f"No face detected in the image from IoT device {iot_id}.", flush=True)
        else:
            print(f"Unknown response from cloud: {response_payload}", flush=True)
    except Exception as e:
        print(f"Error sending to cloud: {e}", flush=True)






def trigger_alarm(iot_id):
    """
    Send an alarm to the IoT device via the alarm SQS queue.
    """
    try:
        alarm_message = {"iot_id": iot_id, "message": "Unknown person detected"}
        alarm_queue.send_message(MessageBody=json.dumps(alarm_message))
        print(f"Alarm triggered for IoT device {iot_id}.", flush=True)
    except Exception as e:
        print(f"Error triggering alarm: {e}", flush=True)

def listen_for_images():
    """
    Listen for image messages from the IoT devices via the SQS queue.
    """
    while True:
        try:
            messages = images_queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=10)
            for message in messages:
                body = json.loads(message.body)
                frame_data = base64.b64decode(body["frame"])    # Decode the base64 string to bytes
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                iot_id = body["iot_id"]

                print(f"Received image from IoT {iot_id}.", flush=True)

                message.delete()
                process_yolo(frame, iot_id)

        except Exception as e:
            print(f"Error receiving image messages: {e}", flush=True)
            time.sleep(5)  # Retry delay


def main():
    # Create threads for image and alarm listeners
    image_thread = threading.Thread(target=listen_for_images)

    # Start the threads
    image_thread.start()

    # No join threads because they run indefinitely


if __name__ == "__main__":
    main()
