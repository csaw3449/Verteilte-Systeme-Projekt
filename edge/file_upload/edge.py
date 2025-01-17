import threading
import boto3
import json
import time
import cv2
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
import botocore.exceptions
from datetime import datetime
from queue import Queue

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
ALARM_DIR = "Unknown_persons"
CONFIDENCE_THRESHOLD = 0.6


# Initialize AWS services
sqs = boto3.resource("sqs", region_name=REGION_NAME)
lambda_client = boto3.client("lambda", region_name=REGION_NAME)

# initlize queue for the producer-consumer-pattern of sending the images
image_queue = Queue(maxsize=10)

def consumer():
    """
    Consumer thread: waits for frames from the queue, then sends them to the cloud.
    """
    while True:
        frame, iot_id = image_queue.get()
        try:
            send_to_cloud(frame, iot_id)
        finally:
            image_queue.task_done()

#Function to get the queue
def get_queue(queue_name):
    """Retry logic to fetch the SQS queue."""
    sqs = boto3.resource("sqs", region_name="us-east-1")
    while True:
        try:
            print(f"EDGE: Attempting to fetch queue: {queue_name}", flush=True)
            return sqs.get_queue_by_name(QueueName=queue_name)
        except botocore.exceptions.ClientError as e:
            print(f"EDGE: Failed to fetch queue {queue_name}: {e}", flush=True)
            print(f"EDGE: Retrying in 5 seconds...", flush=True)
            time.sleep(5)

# connect to the queues
images_queue = get_queue(IMAGES_QUEUE_NAME)
alarm_queue = get_queue(ALARM_QUEUE_NAME)

# Initialize YOLO
classes = open(COCO_NAMES).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

def process_yolo(frame, iot_id,end_time):
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
                end_time.append(time.time())
                
                if classIDs[i] == 0:  # Class 0: Person
                    cropped_frame = frame[y:y+h, x:x+w]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    print(f"EDGE: Person detected by YOLO for IoT device {iot_id}. Sending to cloud.", flush=True)
                    # push image to the queue for the consumer
                    image_queue.put((cropped_frame, iot_id))
                    
                    print(f"EDGE: No person detected in the image from IoT device {iot_id}. Skipping.", flush=True)
    except Exception as e:
        print(f"EDGE: Error in YOLO processing: {e}", flush=True)


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

            trigger_alarm(iot_id, frame)
        elif response_payload.get("status") == "error":
            print(f"CLOUD: Error processing image for IoT device {iot_id}.", flush=True)
            print(f"CLOUD: Error message: {response_payload.get('error')}", flush=True)
        elif response_payload.get("status") == "known":
            print(f"CLOUD: Known person recognized for IoT device {iot_id}.", flush=True)
        elif response_payload.get("status") == "no_face":
            print(f"CLOUD: No face detected in the image from IoT device {iot_id}.", flush=True)
        else:
            print(f"CLOUD: Unknown response from cloud: {response_payload}", flush=True)
    except Exception as e:
        print(f"EDGE: Error sending to cloud: {e}", flush=True)

def plot_benchmarks(start_time, end_time):
    try:
        os.makedirs("plots", exist_ok=True)
        time_diff = np.array(end_time) - np.array(start_time)
        x_values = np.arange(1, len(time_diff) + 1)
        # Create the plot
        plt.figure(figsize=(10, 6))  # Set figure size
        plt.plot(x_values, time_diff, color='blue', linestyle='-', marker='o', label='Time per message')
        plt.xlabel("Message Number")
        plt.ylabel("Time (seconds)")
        plt.title("Time Taken for Each Message")
        plt.grid(True)
        plt.legend()
        #Convert plot to a jpg image and save it in the PLOTS directory
        plt.savefig("./plots/time_diff.jpg",)
        # Save the plot with high resolution
        plt.savefig(os.path.join("plots", "time_diff.jpg"), dpi=300, bbox_inches='tight')#
    except Exception as e:
        print(f"Error plotting benchmarks: {e}", flush=True)


def trigger_alarm(iot_id, frame):
    """
    Send an alarm to the IoT device via the alarm SQS queue.
    """
    try:
        alarm_message = {"iot_id": iot_id, "message": "Unknown person detected"}
        alarm_queue.send_message(MessageBody=json.dumps(alarm_message))
        # Save the cropped image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(ALARM_DIR, f"person_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Edge: Alarm triggered for IoT device {iot_id}.", flush=True)
    except Exception as e:
        print(f"Error triggering alarm: {e}", flush=True)

def listen_for_images():
    """
    Listen for image messages from the IoT devices via the SQS queue.
    """
    msg_count = 0
    start_time = list()
    end_time = list()
    while True:
        try:
            messages = images_queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=10)
            start_time.append(time.time())
            for message in messages:
                msg_count = msg_count + 1
                body = json.loads(message.body)
                frame_data = base64.b64decode(body["frame"])    # Decode the base64 string to bytes
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                iot_id = body["iot_id"]

                print(f"Received image from IoT {iot_id}.", flush=True)

                message.delete()
                process_yolo(frame, iot_id, end_time)
                print(f"EDGE: {msg_count} messages already processed.", flush=True)
                if msg_count == 100:
                    plot_benchmarks(start_time=start_time,end_time=end_time)
        except Exception as e:
            print(f"Error receiving image messages: {e}", flush=True)
            time.sleep(5)  # Retry delay

def start_threads():
    """
    Create 1 producer thread and 3 consumer threads.
    """
    for _ in range(3):
        t = threading.Thread(target=consumer)
        t.start()

    producer_thread = threading.Thread(target=listen_for_images)
    producer_thread.start()

def main():
    try:
        os.makedirs(ALARM_DIR, exist_ok=True)
        start_threads()
    except Exception as e:
        print(f"Error in main: {e}", flush=True)

    # No join threads because they run indefinitely


if __name__ == "__main__":
    main()
