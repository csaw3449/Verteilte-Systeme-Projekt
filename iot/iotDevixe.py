'''class for handeling the iot devices'''

import json
import random
import os
import cv2

def get_random_video():
    pass

def send_images_to_edge():
    pass

def get_images_from_video():
    pass


def main(randomGenerator):
    while True: 
        # We have 11 different video sets and it chooses one of them randomly
        randomSet = randomGenerator.randint(1, 11)



if __name__ == "__main__":
    # get the environment variable for the seed
    seed = os.environ.get('SEED', 111)

    randomGenerator = random.Random()
    randomGenerator.seed(seed)

    main(randomGenerator)