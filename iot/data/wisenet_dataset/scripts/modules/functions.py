"""
Last modified on 28 March 2019

@author: Roberto Marroquin

@description: Module containing functions used in the different scripts.

@comment: To use this module add the following lines to the scripts:
            import sys
            sys.path.append('assets/')
            import functions as fnct

@python version: Python 3.4.3
@opencv version: 3.4.2

"""
import json
import cv2
import dnnFunctions as dnn
import datetime
import os
from timeit import default_timer as timer
import dateutil.parser


def get_calibration_file(video_name):
    """

    :return:
    """
    # Get camera number
    camera_number = video_name.split('_')[1].split('.')[0]  # e.g.: 1, 2 ...
    set_number = int(video_name.split('_')[0].split('o')[-1])

    if set_number < 5:
        # This works on Linux and on Windows
        calibration_file = "../network_enviroment/camera_calibration/1280_720/cam_" + \
            camera_number + ".json"
    else:
        calibration_file = "../network_enviroment/camera_calibration/640_480/cam_" + \
            camera_number + ".json"

    return calibration_file


def load_calibration_data(calibration_file):
    """
    Load the ROI coordinates for each space.
    Videos from sets_1-4 were recorded at 1280x720 resolution,
    while sets_5-11 were recorded at 640x480 resolution
    :param video_name: Name of the video from which the ROI coordinates are needed
    :return: The ROI coordinates (calibration_data), their names (calibration_name) and the name of the camera that setup the coordinates (deviceID)
    """
    try:
        print('Loading calibration Information...')
        with open(calibration_file) as data_file:
            data = json.load(data_file)
    except IOError as io_error:
        print("Failed!")
        print(io_error)
        exit()
        print('Successful!')

    calibration_data = []
    device_id = data['deviceID']
    for i in range(len(data['regionsOfInterest'])):
        xywh = data['regionsOfInterest'][i]['xywh']
        xywh = [
            xywh[0], xywh[1], xywh[2], xywh[3],
            data['regionsOfInterest'][i]['regionOfInterest']
        ]
        calibration_data.append(xywh)
    return calibration_data, device_id


def initialize_video(video_filepath):
    """
    Initilize a video and get its information
    :param video_filepath: Path to the video file
    :return: A dictionary containing the video object and its information
    """
    # Get video information
    splitted_path = video_filepath.split('/')
    video_name = splitted_path[-1]
    video_set = splitted_path[-2]

    # Initialize video stream
    print("Opening capture...")
    video_capture = cv2.VideoCapture(video_filepath)
    # Check if the Video Capture was correctly initialize
    if not video_capture.isOpened():
        print("FAILED !")
        print("ERROR: Couldn't open video %s!!." % video_filepath)
        exit()
    print("Video '%s' successfully loaded" % video_filepath)

    # Get capture information
    video_no_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_resolution = [video_capture.get(
        cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]

    print("FRAMES: {}".format(video_no_frames))
    print("FPS: {}".format(video_fps))

    # Create video dictionary
    video_keys = ["capture", "name", "set", "resolution", "fps", "noFrames"]
    video_values = [video_capture, video_name, video_set,
                    video_resolution, video_fps, video_no_frames]
    keys_and_values = zip(video_keys, video_values)
    video_dict = {}

    for keys, values in keys_and_values:
        video_dict[keys] = values

    return video_dict


def initialize_json_data(video):
    """

    :return:
    """
    json_data = {
        'resolution': [{
            'width': video["capture"].get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': video["capture"].get(cv2.CAP_PROP_FRAME_HEIGHT)
        }],
        'video':
        video["name"],
        'frames': []
    }
    return json_data


def draw_processing_data(frame, net):
    """

    :param frame:
    :return:
    """
    t, _ = net.getPerfProfile()
    processing_time = 'Inference time: %.2f ms' % (
        t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, processing_time, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0))

    return frame


def draw_frame(frame, rois_data, detector, window_name, stop_program):
    frame = draw_rois(frame, rois_data)
    if detector["name"] != "HOG_SVM":
        frame = draw_processing_data(frame, detector["model"])
    # Display frame
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    # If the 'ESC' key or 'q' are pressed, then quit the program
    if key == 27 or key == ord('q'):
        print("Program manually stopped")
        stop_program = True
    return stop_program


def read_frame(video, stop_program):
    """
    Function that reads a new frame if available
    :param video: Video capture
    :param stop_program: Stop flag
    :return:
    """
    grabbed, frame = video.read()
    # If the frame couldn't be grabbed, then we have reached the end of the video
    if not grabbed:
        print("EOF")
        stop_program = True

    return stop_program, frame


def draw_rois(img, rois_data):
    """
    Draw ROIs on an image
    :param img: Image
    :param rois_data: ROIs data
    :return: Image with ROIs
    """
    color_roi = (96, 174, 39)
    for i, _ in enumerate(rois_data):
        point1 = (rois_data[i][0], rois_data[i][1])
        point2 = (rois_data[i][2] + rois_data[i][0],
                  rois_data[i][3] + rois_data[i][1])
        cv2.rectangle(img, point1, point2, color_roi, 1)
        cv2.putText(img, "ROI_" + str(rois_data[i][4].split('_')[-1]),
                    (int(rois_data[i][2] / 2) + rois_data[i][0] - 30,
                     rois_data[i][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color_roi, 1)
    return img


def get_time_stamp(frame_count, starting_datetime, micro_per_frame):
    """

    """
    delta_micro = frame_count * micro_per_frame
    # Convert microseconds to time
    delta_time = datetime.timedelta(microseconds=delta_micro)
    # Add delta time to starting date-time
    current_datetime = starting_datetime + delta_time
    # Convert current date-time to string
    time_stamp = current_datetime.isoformat()[:-3]

    return time_stamp


def initialize_data_to_save(frame_count, device_id, time_stamp):
    """
    """
    data_to_save = []

    data_to_save = {
        'frameNumber': frame_count,
        'deviceID': device_id,
        'inXSDDateTime': time_stamp + 'Z',
        'detections': []}
    return data_to_save


def process_detections(frame, detections, rois_data, frame_count, starting_datetime, micro_per_frame, deviceID, args):
    """
    """
    # Get time stamp FOR OFFLINE!!
    time_stamp = get_time_stamp(
        frame_count, starting_datetime, micro_per_frame)
    data_to_save = initialize_data_to_save(frame_count, deviceID, time_stamp)
    num_detections = 0
    # Loop for all the detections of the frame
    for (object_class, confidence, x, y, w, h) in detections:
        num_detections += 1
        # Determine if the detection is around ROI
        around_roi, detection_color = is_around_roi(rois_data, x, y, w, h)

        # Extract detection's features
        features, x, y, w, h, tshirt_coordinates = extract_features(
            frame, x, y, w, h, args.feature, args.backgroundSubs)

        data_to_save = append_data(
            data_to_save, x, y, w, h, around_roi, features, object_class, confidence, args.detector)

        if args.show:
            # Draw detection
            frame = draw_detection(frame, x, y, w, h, detection_color,
                                   tshirt_coordinates, object_class, confidence, args.backgroundSubs)

    return frame, data_to_save, num_detections


def extract_features(frame, x, y, w, h, visual_descriptor, background_substraction):
    """
    """
    # correct coordinates if necessary
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if (x + w) > frame.shape[1]:
        w = frame.shape[1] - x
    if (y + h) > frame.shape[0]:
        h = frame.shape[0] - y

    detection_region = frame[y:y + h, x:x + w]
    tshirt_coordinates = []
    # Background substraction?
    if background_substraction:
        # Get T-shirt region
        tshirt_top_left, tshirt_bottom_right = get_tshirt_region(x, y, w, h)
        # Extract t-shirt pixels
        detection_region = frame[tshirt_top_left[1]:tshirt_bottom_right[1],
                                 tshirt_top_left[0]:tshirt_bottom_right[0]]
        tshirt_coordinates = [tshirt_top_left, tshirt_bottom_right]

    # Get features
    if visual_descriptor == 'HS':
        features = extract_HS_features(detection_region, hs_bins=8)
    elif visual_descriptor == 'HS_RGB':
        features = extract_HS_RGB_features(
            detection_region, hs_bins=8, rgb_bins=4)

    return features, x, y, w, h, tshirt_coordinates


def get_tshirt_region(x, y, w, h):
    """
    Get tshirt box from a detection
    :param x: detection's top left coordinate
    :param y: detection's top left coordinate
    :param w: detection's width
    :param h: detection's height
    :return: tshirt's top_left and bottom_right points
    """
    ratio = w / h
    if ratio >= 0.55:
        factor = 1 / 1.7
    elif 0.4 <= ratio < 0.55:
        factor = 1 / 3
    elif 0.2 <= ratio < 0.4:
        factor = 1 / 4
    else:
        factor = 1 / 5

    tshirt_x = int(x + w / 3)
    tshirt_y = int(y + h * factor)
    tshirt_w = int(w / 3)
    tshirt_h = int(h / 4)

    top_left = (tshirt_x, tshirt_y)
    bottom_right = (tshirt_x + tshirt_w, tshirt_y + tshirt_h)

    return top_left, bottom_right


def extract_HS_RGB_features(image, hs_bins, rgb_bins):
    """
    Extract a HS + RGB  histograms of an image
    :param image: Image from which the feature will be extracted
           hs_bins: Number of bins for each channel used for the HS historgrams
           rgb_bins: Number of bins for each channel used for the RGB histogram
    :return: HS+RGB+HOG features
    """
    HS = extract_HS_features(image, hs_bins)
    RGB = extract_RGB_features(image, rgb_bins)
    features = HS + RGB

    # normalization? TODO

    return features


def extract_HS_features(image, hs_bins):
    """
    Extract a 2D Hue Saturation color histogram of an image
    :param imgage: Image from which a 2D histogram will be extracted
           hs_bins: Number of bins for each channel used to compute the histrogram
    :return: 2D histrogram features
    """
    # We will compute the 2D histogram by not considering the V value (like this we are less dependent of luminosity)
    # The SATURATION goes from 0-256 (resolution of colors) and the HUE goes from 0-
    # Different BINS where tested, but 9 was found to give the best trade off between performance and accuracy

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [hs_bins] * 2,
                        [0, 180, 0, 256])
    # square-root? TODO
    # L2 normalization
    cv2.normalize(hist, hist)
    # Convert narray to list
    hist = hist.flatten().tolist()

    return hist


def extract_RGB_features(image, rgb_bins):
    """
    Extract a 2D Hue Saturation color histogram of an image
    :param imgage: Image from which a 2D histogram will be extracted
           rgb_bins: Number of bins for each channel used for to compute the histogram
    :return: 2D histrogram features
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [rgb_bins] * 3,
                        [0, 256] * 3)

    # square-root? TODO
    # L2 normalization
    cv2.normalize(hist, hist)
    # Convert narray to list
    hist = hist.flatten().tolist()

    return hist


def extract_HOG_features(image, image_size=(24, 48), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), num_bins=4):
    """

    """
    # Preprocess image_size
    processed_image = (image, image_size[0], image_size[1])

    hog = cv2.HOGDescriptor(image_size, block_size, block_stride,
                            cell_size, num_bins)
    features = hog.compute(processed_image)

    return features.flatten()


def preprocess_image(img, widht, height):
    # Resize image
    resized_img = cv2.resize(img, (int(widht), int(height)))
    new_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.equalizeHist(new_img)
    new_img = cv2.GaussianBlur(new_img, (7, 7), 0)  # Smoothing the image

    return new_img


def get_bbox_area_center(x, y, w, h):
    """
    """
    area = w * h
    center = (x + w / 2, y + h / 2)
    return area, center


def is_around_roi(rois_data, x, y, w, h):
    """
    Determine if detection is around a ROI, for one camera a detection can only be inside one ROI
    """
    around_roi = "null"
    bbox_color = (66, 76, 231)  # redish
    area, center = get_bbox_area_center(x, y, w, h)
    for k, _ in enumerate(rois_data):
        # A detection to be in a ROI first its center most be inside the ROI and also
        # the ROI and the detection should be around the same floor level, i.e their difference shouldn't be greater than 10% of the ROI height
        if (rois_data[k][0] < center[0]
            < (rois_data[k][2] + rois_data[k][0])) and (
                rois_data[k][1] < center[1]
                < (rois_data[k][3] + rois_data[k][1])):
            if (abs((y + h) - (rois_data[k][1] + rois_data[k][3])) <
                    ((rois_data[k][3])) * 0.15):
                around_roi = rois_data[k][4]
                bbox_color = (212, 52, 72)  # blueish
                break
    return around_roi, bbox_color


def detect_people(frame, detector, label_to_filter):
    """
    """
    if detector["name"] == "HOG_SVM":
        detections = hogSVM_detect_people(frame, detector, label_to_filter)
    else:
        detections = dnn.dnn_detect_people(frame, detector, label_to_filter)

    return detections


def append_data(data_to_save, x, y, w, h, around_roi, features, object_class, confidence, detector):
    """
    """
    data_to_save['detections'].append({
        'imageAlgorithm':
            detector,
            'regionOfInterest':
            around_roi,
            'xywh': [x, y, int(w), int(h)],
            'visualDescriptors':
            features,
            'class':
            object_class,
            'confidence':
            confidence
    })

    return data_to_save


def draw_detection(frame, x, y, w, h, color, tshirt_coordinates, object_class, confidence, bs_flag):
    """
    """
    cv2.rectangle(frame, (x, y), (int(x + w), int(y + h)), color, 2)

    label = '%.2f' % confidence
    label = '%s: %s' % (object_class, label)
    cv2.putText(frame, label, (x - 10, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    # Draw t-shirt
    if bs_flag:
        color_tshirt = (13, 184, 255)
        cv2.rectangle(frame, tshirt_coordinates[0], tshirt_coordinates[1],
                      color_tshirt, 1)
    return frame


def get_file_to_save(root_path, args, video_set):
    """

    :return:
    """
    if args.backgroundSubs:
        file_path = os.path.join(
            root_path, args.detector, args.feature + '_BS', video_set)
    else:
        file_path = os.path.join(
            root_path, args.detector, args.feature, video_set)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def write_info_file(file_to_save, video_name, detector, exec_time, frame_count, frame_with_detections, num_detections):
    """

    :return:
    """
    info_file = open(os.path.join(
        file_to_save, video_name.split('.')[0] + '_info.txt'), 'w')
    info_file.write("Detector: %s \n" % detector)
    info_file.write("Video file: %s \n" % video_name)
    info_file.write("Processing time (m): %.2f \n" % (exec_time / 60))
    info_file.write("Average FPS: %.2f \n" % (frame_count / exec_time))
    # We add 1 because we initialize it at -1
    info_file.write("Number of frames: %s \n" % (frame_count + 1))
    info_file.write("Number of frames with detections: %s \n" %
                    frame_with_detections)
    info_file.write("Number of detections: %s \n" % num_detections)
    info_file.close()

    return


def write_json_file(file_to_save, video_name, json_data):
    """

    :return:
    """

    data_file = os.path.join(file_to_save, video_name.split('.')[0] + '.json')

    print('Saving file: ' + data_file)

    with open(data_file, 'w') as file:
        json.dump(json_data, file, sort_keys=True, indent=4)

    print('File successfully saved')

    del json_data
    print("Program successfully FINISHED!")
    print("#######################################\n")

    return


def check_width_height(width, height):
    """
    If detections height is lower than 90% of the width then the detection shouldn't be considered
    :param width:
    :param height:
    :return:
    """
    consider = True
    if height < (width - width * 0.1):
        consider = False

    return consider


def setup_detector(args):
    """

    :param args:
    :return:
    """
    if args.detector == "HOG_SVM":
        detector = setup_hogSVM_detector(args.confThr)
    else:
        detector = dnn.setup_ddn_detector(args.detector, args.confThr)

    return detector


def setup_hogSVM_detector(confThr=0.5):
    """

    :return:
    """
    model = cv2.HOGDescriptor()
    model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Create model dictionary
    detector_keys = ["name", "model", "confThreshold"]
    detector_values = ["HOG_SVM", model, confThr]
    keys_and_values = zip(detector_keys, detector_values)
    detector_dict = {}

    for keys, values in keys_and_values:
        detector_dict[keys] = values
    return detector_dict


def hogSVM_detect_people(frame, detector, label, winStride=(8, 8), padding=(16, 16), scale=1.05):
    """
    """
    resize_width = frame.shape[1]  # 640 320
    resize_height = frame.shape[0]  # 360 240
    # If frame is of high resolution, then resize it to improve processing time
    if frame.shape[0] > 500 and frame.shape[1] > 500:
        resize_width = frame.shape[1] / 2   # 640
        resize_height = frame.shape[0] / 2  # 360

    # Preprocess image
    enhanced_img = preprocess_image(frame, resize_width, resize_height)
    # Run detector - hitThreshold filters the detection's SVM weight that are lower thant the threshold.
    # Notice that the confidence(weight) value has not upper boundary, and greater is better
    outs, confidence = detector["model"].detectMultiScale(
        enhanced_img, winStride=(8, 8), padding=(16, 16), scale=1.05)

    # Post-process detections
    detections = postprocess(
        outs, 0.65, confidence, frame.shape[1], frame.shape[0], resize_width, resize_height, label)

    return detections


def postprocess(outs, confThreshold, confidence, width, height, resize_width, resize_height, label):
    """

    :return:
    """
    # we convert to float to obtain a more accurate value from the division
    width_ratio = width / float(resize_width)
    height_ratio = height / float(resize_height)
    object_classes = []
    confidences = []
    boxes = []
    detections = []
    for i, (x, y, w, h) in enumerate(outs):
        # Shrink the boxes so they are tighter
        pad_w, pad_h = (0.3 * w * width_ratio), (0.11 * h * height_ratio)
        # Up-scale the detection's coordinates
        x = int((x * width_ratio) + pad_w)
        y = int((y * height_ratio) + pad_h)
        w = int((w * width_ratio) - 2 * pad_w)
        h = int((h * height_ratio) - 2 * pad_h)

        object_classes.append(label)
        confidences.append(float(confidence[i]))
        boxes.append([x, y, w, h])
    # Filter detections by confidence and by NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.3)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        detections.append([object_classes[i], confidences[i], x, y, w, h])

    return detections


def release(capture, model):
    """

    :return:
    """

    # Release resources
    capture.release()
    cv2.destroyAllWindows()
    del model


def calculate_exec_time(time_start):
    """

    :param start:
    :return:
    """
    time_end = timer()
    exec_time = (time_end - time_start)
    return exec_time


def write_files(file_path, json_data, video_name, detector_name, exec_time, frame_count, frame_with_detections, detections_count):
    """

    :return:
    """
    # Write data about time and FPS
    write_info_file(file_path, video_name, detector_name, exec_time,
                    frame_count, frame_with_detections, detections_count)
    # Write JSON file
    write_json_file(file_path, video_name, json_data)


def setup_window(window_name):
    """

    :return:
    """
    # Initialize window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


def get_calibration_data(video_name):
    """

    :return:
    """
    calibration_file = get_calibration_file(video_name)
    rois_data, device_id = load_calibration_data(calibration_file)
    return rois_data, device_id


def initialize_variables(video, starting_time, class_label='person'):
    """

    :return:
    """
    stop_program = False
    frame_count = -1  # OpenCV considers frames as 0-based index
    frame_with_detections = 0
    detections_count = 0

    # The videos were recorded at 30FPS, which is 1 frame per 33333.3333 microseconds
    # We use microseconds to have a finer resolution, we can have many detections in one second
    micro_per_frame = 1000000 / video["fps"]
    json_data = initialize_json_data(video)
    time_start = timer()
    starting_datetime = dateutil.parser.parse(starting_time)

    return stop_program, frame_count, frame_with_detections, detections_count, micro_per_frame, json_data, time_start, starting_datetime, class_label


def load_data(file_path):
    """

    :param file_path:
    :return:
    """
    try:
        print('Loading data from json file...')
        with open(file_path) as data_file:
            data = json.load(data_file)
    except:
        print('Failed!')
        print('PATH: %s' % file_path)
        print('Please verify the path of the json file!!')
        exit()
    print("Successful!")

    return data


def terminate_video(video_capture):
    """

    :param video_capture:
    :return:
    """
    video_capture.release()
    cv2.destroyAllWindows()

    print("Program finished")

    return None
