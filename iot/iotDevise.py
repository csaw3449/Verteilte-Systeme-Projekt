import random
import os
import cv2
import time 
import threading

frame_rate = 24 # TODO: find out the frame rate of the videos
secounds_between_images = 5
delay_images = frame_rate * secounds_between_images # 5 seconds delay between each image

def send_frame(frame):
    # TODO: send the frame to the edge layer to be processed
    pass

# @app.route('/trigger_alarm', methods=['POST'])
def waiting_for_alarm():
    # TODO: implement logic that waits for an alarm from the AWS and prints something
    pass

def main(randomGenerator):
    path_to_video = "data//wisenet_dataset/video_sets/set_"
    while True: 
        # We have 11 different video sets and it chooses one of them randomly
        randomSet = randomGenerator.randint(1, 11) 
        # selects all videos from the chosen set
        videos = os.listdir(path_to_video + str(randomSet))
        # selects all videos from the chosen set 
        for video in videos:
            cap = cv2.VideoCapture(path_to_video + str(randomSet) + "//" + video)
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
        
if __name__ == "__main__":
    # get the environment variable for the seed
    seed = os.environ.get('SEED', 111)

    randomGenerator = random.Random()
    randomGenerator.seed(seed)

    main(randomGenerator)