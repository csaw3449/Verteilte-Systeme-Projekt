import boto3
import botocore
import cv2
import json
import numpy as np
import asyncio
import requests
from ultralytics import YOLO

# Load YOLO model for person detection
model_person = YOLO("yolov8n.pt")  # Person detection model (YOLOv8)


# Initialize SQS Queues
sqs = boto3.resource("sqs", region_name="us-east-1")

#@
#async def sendMessage(message):


MAX_RETRIES = 5
async def wait_for_queue(queue_name):
    """Wait for the queue to be created and available."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"Waiting for queue {queue_name} to be available...")
            load_balancer_queue = sqs.get_queue_by_name(QueueName=queue_name)
            print(f"Queue {queue_name} found.")
            return load_balancer_queue
        except botocore.exceptions.ClientError as e:
            print(f"Queue {queue_name} not found. Retrying...")
            retries += 1
            await asyncio.sleep(5)
    raise Exception(f"Queue {queue_name} not found after {MAX_RETRIES} retries.")


async def detect_person(frame):
    """Detect persons using YOLO (Ultralytics)."""
    results = model_person(frame)  # Run YOLO model for person detection
    for result in results.pred[0]:  # Results from the first image (batch size 1)
        if result[-1] == 0:  # Class 0 corresponds to 'person' in the coco dataset
            # Return the bounding box for the detected person (x1, y1, x2, y2)
            return result[:4].cpu().numpy()
    return None

async def process_frame(frame, iot_id):
    """Process a single frame."""
    try:
        # Detect person first
        person_bbox = await detect_person(frame)
        if person_bbox is not None:
            print(f"Person detected for IoT ID: {iot_id}. Cropping person frame...")

            # Crop the detected person from the frame
            x1, y1, x2, y2 = person_bbox
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the person from the frame

            # Encode the cropped person image to send to the cloud
            _, encoded_image = cv2.imencode(".jpg", person_frame)

            # Prepare the message with the cropped person image
            message = {
                "iot_id": iot_id,
                "frame": encoded_image.tobytes().decode("latin1"),  # Convert bytes to string
            }

            # Send the cropped person image to the cloud
            sendMessageToCloud(message);
            #forward_queue.send_message(MessageBody=json.dumps(message))
            print(f"Person frame sent to cloud for IoT ID: {iot_id}.")
        else:
            print(f"No person detected for IoT ID: {iot_id}.")
    except Exception as e:
        print(f"Error processing frame: {e}")

async def receive_messages():
    """Continuously receive messages from the load balancer queue."""
    load_balancer_queue = await wait_for_queue("images")  # Wait for the queue to be available

    while True:
        try:
            messages = load_balancer_queue.receive_messages(MaxNumberOfMessages=10, WaitTimeSeconds=10)
            for message in messages:
                body = json.loads(message.body)
                iot_id = body.get("iot_id")
                frame_data = body.get("frame").encode("latin1")
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                await process_frame(frame, iot_id)  # Process frame
                message.delete()
        except botocore.exceptions.ClientError as e:
            print(f"Error receiving messages: {e}. Retrying...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")
            await asyncio.sleep(5)

async def main():
    """Main function to start the Edge Layer."""
    print("Starting Edge Layer...")
    await receive_messages()

if __name__ == "__main__":
    asyncio.run(main())
