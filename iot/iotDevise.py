import random
import os
import cv2
import time 
import threading

frame_rate = 30 # Framerate of the videos
seconds_between_images = 3 # 3 seconds between each image 
delay_images = frame_rate * seconds_between_images # 3 seconds delay between each image

def send_frame(frame):
    # TODO: send the frame to the edge layer to be processed
    pass

# @app.route('/alarm', methods=['POST'])
def waiting_for_alarm():
    # TODO: implement logic that waits for an alarm from the AWS and prints something
    pass

def main():
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
                    time.sleep(seconds_between_images)
                # just for testing purposes TODO remove this block after testing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()