"""
Last modified on July 3 2017

@author: Roberto Marroquin

@description: Script t œo visualize (manual or automatic) detections stored in a json file.

@usage:
python3 detections_visualizer.py -f ../manual_annotations/people_detection/set_4/video4_4.json --id 1
python3 detections_visualizer.py -f ../automatic_annotations/YOLOv3_608/HS_BS/set_4/video4_4.json --rois 0

@comment: We assume that: (1) the script was executed from the script directory of the dataset, (2) the name of the video is saved on the json file.

@python version: Python 3.4

"""
import argparse
import json
import cv2
import sys
sys.path.append('modules/')
import functions as fcnt

frame_count = 0
detection_count = 0
prev_frame = False


def parse_arguments():
    """
    Function that parse arguments.
    :return: arguments
    """
    argument_parser = argparse.ArgumentParser(description='Detection viewer')
    argument_parser.add_argument(
        "-f", "--file", help="Path to the json detection file", required=True)
    argument_parser.add_argument(
        "-r",
        "--rois",
        default=0,
        type=int,
        help="Flag to visualize the ROI data [1]")
    argument_parser.add_argument(
        "-id",
        "--id",
        help="Flag to determine if the data to process has information about the IDs, e.g., "
        " for ground truth data [0]",
        default=0,
        type=int)
    args = argument_parser.parse_args()

    return args


def initialize():
    """
    Function that loads the video and the data.
    :return: arguments, data, video capture and video information
    """
    args = parse_arguments()

    data = fcnt.load_data(args.file)

    video_name = data['video']  # e.g.: video1_3.avi
    video_set = video_name.split('_')[0].split('video')[1]
    video_path = '../video_sets/set_' + video_set + '/' + \
        video_name  # e.g.: video_sets/set_1/video1_3.avi
    video = fcnt.initialize_video(video_path)

    rois_data = []
    if args.rois:
        rois_data, device_id = fcnt.get_calibration_data(video_name)

    # Set up display window
    cv2.namedWindow(video_name)

    # Initialize variables
    stop_program = False

    return args, data, rois_data, video["capture"], video["noFrames"], stop_program, video["name"]


def read_frame(video, stop_program):
    """
    Function to read a new frame
    :param video: video capture
    :param stop_program: stop flag
    :return: new frame
    """
    global frame_count
    global prev_frame

    if prev_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        prev_frame = False

    grabbed, frame = video.read()

    # If the frame couldn't be grabbed, then we have reached the end of the video
    if not grabbed:
        print("EOF")
        stop_program = True

    return frame, stop_program


def extract_draw_data(frame, data, person_id):
    """
    Extract information from the data if detection counter is lower than the max number of detections and
    the frame number correspondence to the current frame, if not just display the frame
    :param frame: current frame
    :param data: detections data in json format
    :param person_id: flag that determines if the data is ground truth or not. If yes, the person ID will be drawn
    :return:
    """
    global detection_count

    # color when detection is around a roi
    color_around_roi = (212, 52, 72)  # blueish
    color_tshirt = (13, 184, 255)
    detections = data['frames'][detection_count]['detections']
    for i in range(0, len(detections)):
        color = (66, 76, 231)  # by default is redish
        x1 = detections[i]['xywh'][0]
        y1 = detections[i]['xywh'][1]
        w = detections[i]['xywh'][2]
        h = detections[i]['xywh'][3]
        x2 = w + x1
        y2 = h + y1

        # If the detection is around a roi
        if not detections[i]['regionOfInterest'] == "null":
            color = color_around_roi
            cv2.putText(
                frame,
                "ROI_" + str(detections[i]['regionOfInterest'].split('_')[-1]),
                (x2 - 55, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if person_id:
            cv2.putText(frame, str(detections[i]['id']), (x1 - 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw T-shirt - region from where we got the visual descritors
        tshirt_top_left, tshirt_bottom_right = fcnt.get_tshirt_region(
            x1, y1, w, h)
#        cv2.rectangle(frame, tshirt_top_left, tshirt_bottom_right,
#                      color_tshirt, 1)

    # Print time
    cv2.putText(frame, data['frames'][detection_count]['inXSDDateTime'],
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

    return frame


def draw_menu(frame, no_frames):
    """

    :param frame: current frame
    :param no_frames: maximum number of frames in the video
    :return:
    """
    global frame_count

    # Drawing menu
    cv2.putText(frame, 'SPACE: next frame', (15, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 2)
    cv2.putText(frame, 'ESC: stop program', (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 2)
    cv2.putText(frame, 'Frame #: ' + str(frame_count) + '/' + str(no_frames),
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return frame


def display_frame(frame, rois_data, display_name, no_frames, data,
                  stop_program):
    """
    
    :param frame: current frame
    :param rois_data: information about the regions of interest
    :param display_name: window name
    :param no_frames: maximum number of frames in the video
    :param data: detection data
    :param stop_program: stop flag
    :return: 
    """

    next_frame = False

    while not next_frame:
        # Draw menu
        frame = draw_menu(frame, no_frames)
        # Draw ROIS
        if rois_data:
            frame = fcnt.draw_rois(frame, rois_data)
        cv2.imshow(display_name, frame)

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        next_frame, stop_program = process_keyboard(key, next_frame, data,
                                                    stop_program)

    return stop_program


def process_keyboard(key, next_frame, data, stop_program):
    """

    :param key: keyboard key
    :param next_frame: flag to go to the next frame
    :param stop_program: stop flag
    :return:
    """
    global frame_count
    global prev_frame
    global detection_count

    # If the 'ESC' key or 'q' are pressed, then quit the program
    if key == 27 or key == ord('q'):
        print("Program stopped")
        stop_program = True
        next_frame = True

    # If 'space bar' is pressed then go to grab next frame
    elif key == 32:
        next_frame = True
        frame_count += 1

    # If the 'TAB' key is pressed then go to previous frame
    # FIXME: There is a bug while going to previous frame, somethimes the detections stop to appear
    # Therefore we comment this function
    elif key == 9:
        prev_frame = True
        temp_count = detection_count - 1
        if temp_count >= 0:
            if data['frames'][temp_count]['frameNumber'] == frame_count:
                if not temp_count == 0:
                    # FIXME: backward inconsistency with frame_count needs -1
                    detection_count = temp_count - 1

        frame_count -= 1
        if not frame_count >= 0:
            frame_count = 0

        next_frame = True

    return next_frame, stop_program


def main():
    """ MAIN """
    global frame_count
    global detection_count

    args, data, rois_data, video_capture, no_frames, stop_program, video_name = initialize(
    )

    while not stop_program:

        # Get next frame
        frame_raw, stop_program = read_frame(video_capture, stop_program)

        if not stop_program:
            frame = frame_raw.copy()
            # Check correspondence between the data and the video frame
            if (detection_count < len(
                    data['frames'])) and (data['frames'][detection_count]
                                          ['frameNumber'] == frame_count):
                frame = extract_draw_data(frame_raw, data, args.id)
                detection_count += 1

        stop_program = display_frame(frame, rois_data, video_name, no_frames,
                                     data, stop_program)

    fcnt.terminate_video(video_capture)


if __name__ == "__main__":
    main()
