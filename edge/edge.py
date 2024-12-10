import threading
import boto3
import json
import time
import cv2
from ultralytics import YOLO
import botocore.exceptions

"""
This programm receives images from the IoT devices, processes them with YOLO, and sends them to the cloud.
This programm requires environment variables to be set:
- CLOUD_LAMBDA_FUNCTION: the name of the cloud Lambda function
- IMAGES_QUEUE_NAME: the name of the images SQS queue
- ALARM_QUEUE_NAME: the name of the alarm SQS queue
- REGION_NAME: the AWS region name
"""


# Configuration
REGION_NAME = "us-east-1"
CLOUD_LAMBDA_FUNCTION = "processImage"  #TODO: Set the actual Lambda function name
IMAGES_QUEUE_NAME = "images"
ALARM_QUEUE_NAME = "alarm"

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
images_queue = get_queue(QueueName=IMAGES_QUEUE_NAME)
alarm_queue = get_queue(QueueName=ALARM_QUEUE_NAME)
# Initialize YOLO
model = YOLO("yolov8n.pt")  # Hope this is standart YOLO weights file


def process_yolo(frame, iot_id):
    """
    Run YOLO filtering on the frame and send valid images to the cloud Lambda function.
    Also start a thread for sending and receiving the message to the cloud.
    """
    try:
        results = model(frame)
        detections = results[0].boxes.data  # Adjust based on your YOLO implementation

        # Check if a person is detected (class ID = 0 in COCO dataset)
        person_detected = any(int(detection[-1]) == 0 for detection in detections)

        if person_detected:
            print(f"Person detected by YOLO for IoT device {iot_id}. Sending to cloud.", flush=True)
            threading.Thread(target=send_to_cloud, args=(frame, iot_id)).start()
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
            "image": encoded_image.tobytes().decode("latin1")  # Convert bytes to JSON serializable string
        }

        response = lambda_client.invoke(
            FunctionName=CLOUD_LAMBDA_FUNCTION,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )

        response_payload = json.loads(response["Payload"].read())   #TODO: Check for response format
        if response_payload.get("status") == "unknown":
            trigger_alarm(iot_id)
        else:
            print(f"Person recognized for IoT device {iot_id}.", flush=True)
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
                frame_data = body["frame"].encode("latin1")  # Decode string to bytes
                frame = cv2.imdecode(bytearray(frame_data), cv2.IMREAD_COLOR)
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
