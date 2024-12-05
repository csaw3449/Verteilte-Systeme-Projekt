import random
import os
import cv2
import time 
import threading
import boto3
import json
"""
This programm sends images in from a video dataset to the edge layer in a specific time interval.
It also waits for an alarm from the edge layer and prints it.
This programm need environment variables to be set:
- IOT_ID: the id of the iot device
- SET_NUMBER: the number of the video set to be used
"""
frame_rate = 24 # TODO: find out the frame rate of the videos, not really that important. Could also make random sampling
secounds_between_images = 5
delay_images = frame_rate * secounds_between_images # 5 seconds delay between each image
#aws communicatioin
sqs = boto3.resource('sqs', region_name='us-east-1')
send_queue = sqs.get_queue_by_name(QueueName='images')  #TODO: names might change, check with edge layer
receive_queue = sqs.get_queue_by_name(QueueName='alarm')

id = os.environ.get('IOT_ID')

def send_frame(frame):
    # TODO: send the frame to the edge layer to be processed
    message = {
        "iot_id": id,
        "frame": cv2.imencode('.jpg', frame)[1].tobytes()
    }
    send_queue.send_message(MessageBody=json.dumps(message))
    print(f"Frame sent frame to queue: {message}")

# @app.route('/alarm', methods=['POST'])
def waiting_for_alarm():
    # TODO: implement logic that waits for an alarm from the AWS and prints something
    while True:
        response = receive_queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=10)
        for message in response:
            body = json.loads(message.body)
            if body.get("iot_id") == id:
                print(f"Alarm received: {message.body}")
                message.delete()

def send_images():
    set_number = os.environ.get('SET_NUMBER', 1)
    path_to_video = "data//wisenet_dataset/video_sets/set_"
    while True: 
        # selects all videos from the chosen set
        videos = os.listdir(path_to_video + str(set_number))
        # selects all videos from the chosen set 
        for video in videos:
            cap = cv2.VideoCapture(path_to_video + str(set_number) + "//" + video)
            counter = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret: 
                    break
                counter += 1
                # we only need to send the frame every 5 seconds
                if (counter > delay_images):
                    # Display the resulting frame TODO remove this line after testing
                    cv2.imshow('frame',frame)
                    counter = 0
                    # send the frame
                    send_frame(frame)
                    # waits for 5 seconds to simulate the delay
                    time.sleep(secounds_between_images)
                # just for testing purposes TODO remove this block after testing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

def main():
    threadSend = threading.Thread(target=send_images)
    threadReceive = threading.Thread(target=waiting_for_alarm)
    
    threadSend.start()
    threadReceive.start()
        
if __name__ == "__main__":
    main()