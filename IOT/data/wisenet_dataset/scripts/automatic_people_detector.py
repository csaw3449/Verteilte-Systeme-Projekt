
"""
Last modified on 28 March 2019

@author: Roberto Marroquin

@description: Program to automatically detect people from a video file using different people detector models and save the detections on a json file.

@usage: python3 automatic_people_detector.py --input ../video_sets/set_3/video3_1.avi --show 1 --backgroundSupp 1 --feature HS --detector YOLOv3_608

@comment: We assume that the script was executed from the script directory of the dataset,

@python version: Python 3.4
@opencv version: 3.4.2 and 4.0.0

"""

import sys
import argparse
import cv2
sys.path.append('modules/')
import functions as fnct

# Date-time that would be considered as the starting point. We fix it to have the same starting point in all the videos.
STARTING_TIME = '2017-04-24T12:15:00.001'


def parse_args():
    """
    FUnction that parse input arguments
    :return: args (dict): The parsed arguments
    """
    detectors = ('HOG_SVM', 'YOLOv3_608', 'SSD_512')
    visualDescriptors = ('HS', 'HS_RGB')
    parser = argparse.ArgumentParser(
        description='Use this script to run people detector models using OpenCV2.')
    parser.add_argument(
        '--detector',
        choices=detectors,
        type=str,
        default='YOLOv3_608',
        help='Choose one of the pre-defined  model [YOLOv3_608].')
    parser.add_argument(
        '--feature',
        choices=visualDescriptors,
        type=str,
        default='HS',
        help='Choose one of the pre-defined visual descriptors, Hue-Saturation (HS) or HS+RGB+HOG (HS_RGB_HOG) [HS].')
    parser.add_argument(
        '--backgroundSupp',
        type=int,
        default=0,
        help='Flag to determine if the background subtraction (bs) should be performed [0].'
    )
    parser.add_argument(
        '--input',
        default=None,
        required=True,
        help='Path to input image or video file.')
    parser.add_argument(
        '--show',
        type=int,
        default=0,
        help='Flag to determine if the results should be shown or not [0].')
    parser.add_argument(
        '--confThr',
        type=float,
        default=0.5,
        help='Confidence threshold to consider a detection. Used only with DNN models [0.5].')
    args = parser.parse_args()

    return args


def main():
    """MAIN"""
    args = parse_args()
    detector = fnct.setup_detector(args)
    if args.show:
        window_name = 'Deep learning person detector'
        fnct.setup_window(window_name)
    video = fnct.initialize_video(args.input)
    rois_data, device_id = fnct.get_calibration_data(video["name"])
    # Set-up variables and constants
    stop_program, frame_count, frame_with_detections, \
        detections_count, micro_per_frame, json_data, time_start, \
        starting_datetime, label_to_filter = fnct.initialize_variables(
            video, STARTING_TIME)

    # video["capture"].set(cv2.CAP_PROP_POS_FRAMES, 810)  # uncomment to start in a precise frame
    while not stop_program:
        stop_program, frame = fnct.read_frame(video["capture"], stop_program)
        if not stop_program:
            frame_count += 1
            # Get detections
            detections = fnct.detect_people(frame, detector, label_to_filter)
            if detections:
                frame, data_to_save, num_detections = fnct.process_detections(
                    frame, detections, rois_data, frame_count, starting_datetime, micro_per_frame, device_id, args)
                json_data['frames'].append(data_to_save)
                detections_count += num_detections
                frame_with_detections += 1
            if args.show:
                stop_program = fnct.draw_frame(
                    frame, rois_data, detector, window_name, stop_program)
    exec_time = fnct.calculate_exec_time(time_start)
    fnct.release(video["capture"], detector["model"])
    file_to_save = fnct.get_file_to_save(
        'automatic_annotations/', args, video["set"])
    fnct.write_files(file_to_save, json_data, video["name"], detector["name"],
                     exec_time, frame_count, frame_with_detections, detections_count)


if __name__ == "__main__":
    main()
