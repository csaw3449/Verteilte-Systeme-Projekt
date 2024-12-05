"""
Last modified on 11 April 2019

@author: Roberto Marroquin

@description: Script to extract detections from a set of JSON files and save then into a text file.
              The text file can be used to evaluate the detections using the code in: https://github.com/rafaelpadilla/Object-Detection-Metrics
              
@usage: python3 json_to_txt.py -dir ../automatic_annotations/HOG_SVM/HS_BS/set_1
        python3 json_to_txt.py -dir ../manual_annotations/people_detection/set_1 -gt 1

@comment: We assume that the script was executed from the script directory of the dataset.

@python version: Python 3.4
@opencv version: 3.4.2 and 4.0.0

"""
import argparse
import json
import os


def initialize():
    """
    Function that initialize the script by loading the json file and getting the path to save the text files
    :return: the json data, the path where the txt files will be saved and the ground truth flag
    """
    # Parse user input
    ap = argparse.ArgumentParser(
        description='Script to extract detections from a JSON file and save then into a text file')
    ap.add_argument("-dir", "--directory", help="Directory containing json files with detections")
    ap.add_argument("-gt", "--groundtruth", type=int,
                    help="Flag indicating if the file is ground truth or not by default [0]", default=0)
    ap.add_argument("-sk", "--skip", type=int,
                    help="Number of frames to be skipped [0]", default=0)
    args = ap.parse_args()

    # Get all files
    full_path = os.path.join(os.path.dirname(__file__), args.directory)
    files = []
    for file_to_check in os.listdir(full_path):
        if file_to_check.endswith(".json"):
            files.append(file_to_check)

    return full_path, files, args.groundtruth, args.skip


def load_json(path, file):
    """
    Funtion that loads the json file
    :param file: json file
    :param full_path: path of the file
    :return: dictionary with the json data
    """
    # Load JSON file
    json_data = []
    print('Loading JSON file: %s' % file)
    file_path = path + '/' + file
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
    except:
        print('Failed!')
        print('ERROR while opening the JSON file!!')
        exit()
    print("JSON file opened successfully!")

    return json_data


def get_saving_path(path, file, gt_flag):
    """
    Function that gets the saving path based on the JSON file name
    :param file: json file
    :param gt_flag: flag stating if the json file is a ground truth or not
    :return: path where the text file will be saved
    """

    tmpPath1 = path.split('/')
    set_name = tmpPath1[-1]
    video_name = file.split('.')[0]

    # Get path to save text files
    if gt_flag == 0:
        # Supposing the structure of the detections is XXX/Detector/Set_number
        root_path = "automatic_annotations_txt/"
        detector_name = tmpPath1[-2]
    else:
        root_path = "manual_annotations_txt/"
        detector_name = ""

    path = os.path.join(root_path, detector_name, set_name, video_name)

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def write_into_txt(file, detection, gt_flag):
    """
    Function that extracts the information from the json file and insert them into the txt file
    :param file: txt file
    :param detection: current detection data
    :param gt_flag: ground truth flag
    :return: 
    """
    # Extract class
    class_name = detection['class']
    # Extract xywh
    x = str(detection['xywh'][0])
    y = str(detection['xywh'][1])
    w = str(detection['xywh'][2])
    h = str(detection['xywh'][3])
    # Extract confidence
    if gt_flag == 0:
        conf = str(detection['confidence'])
        # Add detection info to text file
        file.write(class_name + ' ' + conf + ' ' + x + ' ' + y + ' ' + w + ' ' + h + '\n')
    else:
        # Add detection info to text file
        file.write(class_name + ' ' + x + ' ' + y + ' ' + w + ' ' + h + '\n')

    return


def create_file(frame, path):
    """
    Function that creates the text file 
    :param frame: current frame
    :param path: path of the text file
    :return: text file
    """
    # Extract frame number
    frame_number = frame['frameNumber']
    # Create text file
    file_name =os.path.join(path, str(frame_number) + '.txt')
    file = open(file_name, 'w')

    #print("Created file: %s" %file_name)

    return file


def main():
    """MAIN FUNCTION"""

    path, files, gt_flag, skip = initialize()

    # Go through all files
    for i in range(len(files)):
        print(files[i])

        json_data = load_json(path, files[i])
        path_save = get_saving_path(path, files[i], gt_flag)

        # Move through all the frames in the file
        for j, frame in enumerate(json_data['frames']):

            # Check if the frame should be skipped
            current_frame = frame['frameNumber']

            if skip != 0:
                if current_frame % skip != 0:
                    continue
            file = create_file(frame, path_save)

            # Move through all the detections in the frame
            for k, detection in enumerate(frame['detections']):
                write_into_txt(file, detection, gt_flag)

            file.close()

        print('Extraction of %s completed!\n' % files[i])

if __name__ == "__main__":
    main()
