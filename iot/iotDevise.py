import os
import cv2
import time
import threading
import boto3
import base64
import json
import botocore.exceptions
import queue 
import sys
import signal

"""
This program sends images from a video dataset to the edge layer in a specific time interval.
It also waits for an alarm from the edge layer and rints it.
This program requires environment variables to be set:
- IOT_ID: the ID of the IoT device
- SET_NUMBER: the number of the video set to be used
"""
# Configuration
frame_rate = 30 # Framerate of the videos
seconds_between_images = 1 # 1 seconds between each image 
delay_images = frame_rate * seconds_between_images # 3 seconds delay between each image

# Environment Variables
id = os.environ.get("IOT_ID", "default_id")  # Default ID if not set
set_number = os.environ.get("SET_NUMBER", 1)

responses = queue.Queue(maxsize=500)

# Initialize SQS Queues with Retry Logic
MAX_RETRIES = 5

#Function to get the queue, with MAX_RETRIES attemps
def get_queue(queue_name, retries=0):
    """Retry logic to fetch the SQS queue."""
    sqs = boto3.resource("sqs", region_name="us-east-1")
    while retries < MAX_RETRIES:
        try:
            print(f"Attempting to fetch queue: {queue_name}", flush=True)
            return sqs.get_queue_by_name(QueueName=queue_name)
        except botocore.exceptions.ClientError as e:
            print(f"Failed to fetch queue {queue_name}: {e}", flush=True)
            print(f"Retrying in 5 seconds... Attempt {retries + 1} of {MAX_RETRIES}", flush=True)
            time.sleep(5)
            retries += 1

    print(f"Failed to fetch queue {queue_name} after {MAX_RETRIES} retries. Exiting.", flush=True)
    exit(1)  # Return None after reaching max retries

send_queue = get_queue("images")
receive_queue = get_queue("alarm")

# Function to send frames
def send_frame(frame):
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded_image = cv2.imencode(".jpg", frame, encode_param)
        start_time = time.time()
        message = {
            "iot_id": id,
            "frame": base64.b64encode(encoded_image).decode("utf-8"),  # Base64 -> UTF-8 string
            "iot_start": start_time
        }
        send_queue.send_message(MessageBody=json.dumps(message))
        print(f"Frame sent to queue", flush=True)
    except Exception as e:
        print(f"Error sending frame: {e}", flush=True)
        print("Retrying to send frame in 5 seconds...", flush=True)
        time.sleep(5)
        send_frame(frame)  # Retry sending the frame after 5 seconds

# Function to wait for alarms
def waiting_for_alarm():
    # make folder for csv files with timestamps
    if not os.path.exists("times"):
        os.makedirs("times", exist_ok=True)
    while True:
        try:
            response = receive_queue.receive_messages(
                MaxNumberOfMessages=1, WaitTimeSeconds=10
            )
            for message in response:
                body = json.loads(message.body)
                if body.get("iot_id") == id:
                    print(f"Alarm received: {message.body}", flush=True)
                    # add iot_end to the body
                    body["iot_end"] = time.time()
                    # for saving the time in a csv file
                    responses.put(body)
                    message.delete()
                else:
                    print(f"Ignoring alarm for {body['iot_id']}", flush=True)
        except botocore.exceptions.ClientError as e:
            print(f"Error receiving alarm: {e}", flush=True)
            print("Retrying alarm listener in 5 seconds...", flush=True)
            time.sleep(5)

# Function to send images
def send_images():
    path_to_video = f"data/wisenet_dataset/video_sets/set_{set_number}/"
    while True:
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
                        time.sleep(seconds_between_images)
                cap.release()
                cv2.destroyAllWindows()
        except botocore.exceptions.ClientError as e:
            print(f"Error sending images: {e}", flush=True)
            print("Retrying image sending in 5 seconds...", flush=True)
            time.sleep(5)

def write_times_in_csv():
    # write times in csv file
    filename = f"times/times_{id}.csv"
    with open(filename, "w") as f:
        print(f"Writing times in {filename}", flush=True)
        f.write("iot_start, iot_end, edge_start1, edge_end1, cloud_start, cloud_end, edge_start2, edge_end2\n") # Should be correct now!
        while not responses.empty():
            response = responses.get()
            f.write(f"{response['iot_start']}," + f"{response['iot_end']}," 
                    + f"{response['edge_start1']}," + f"{response['edge_end1']},"
                    + f"{response['cloud_start']}," + f"{response['cloud_end']},"
                    + f"{response['edge_start2']}," + f"{response['edge_end2']}\n") # Should be correct now!
    print("Exiting...", flush=True)

def signal_handler(sig, frame):
    print("Signal received, cleaning up...")
    write_times_in_csv()

# Main Function
def main():
    # Register signal handlers for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    thread_send = threading.Thread(target=send_images)
    thread_receive = threading.Thread(target=waiting_for_alarm)

    thread_send.start()
    thread_receive.start()

if __name__ == "__main__":
    main()