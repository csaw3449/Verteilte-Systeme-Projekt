'''class for handeling the iot devices'''

import json
import random
import os
import cv2

def main(randomGenerator):
    path_to_video = "data//wisenet_dataset/video_sets/set_"
    while True: 
        # We have 11 different video sets and it chooses one of them randomly
        randomSet = randomGenerator.randint(1, 11) 
        # selects all videos from the chosen set
        videos = os.listdir(path_to_video + str(randomSet))
        # selects one of the videos randomly
        for video in videos:
            cap = cv2.VideoCapture(path_to_video + str(randomSet) + "//" + video)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    # Display the resulting frame
                    cv2.imshow('frame',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()
        
        print("Video set " + str(randomSet) + " is done")
        return
        




if __name__ == "__main__":
    # get the environment variable for the seed
    seed = os.environ.get('SEED', 111)

    randomGenerator = random.Random()
    randomGenerator.seed(seed)

    main(randomGenerator)