import cv2
import boto3
import numpy as np
import base64

# outside of the lambda function to avoid creating a new client for every invocation (might be changed)
rekognition = boto3.client('rekognition', region_name='us-east-1')

# this lambda function is triggered by the edge layer and the pictures a located in a container
def lambda_handler(event, context):
    # get the image and iot_id from the event
    image = event['image']
    iot_id = event['iot_id']
    
    # decode the image
    image_decoded = image.encode('latin1', image)
    image = cv2.imdecode(np.frombuffer(image_decoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {
            'status': 'error',
            'iot_id': iot_id
        }
    
    # Re-encode the image to base 64
    _, image_buffer = cv2.imencode('.jpg', image)
    image64 = base64.b64encode(image_buffer)

    # look for similar faces in the collection and check if there is a 90% match
    response = rekognition.search_faces_by_image(
        CollectionId='pfusch-collection',
        QualityFilter='NONE',
        Image={
            'Bytes': image64
        },
        FaceMatchThreshold=90
    )

    # if there is a match, return the status known otherwise unknown
    if len(response['FaceMatches']) > 0:
        return {
            'status': 'known',
            'iot_id': iot_id
        }
    else:
        return {
            'status': 'unknown',
            'iot_id': iot_id
        }
    