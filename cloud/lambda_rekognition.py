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
    
    image_bytes = base64.b64decode(image)
    image_base64 = base64.b64encode(image_bytes)
    
    if image_base64 is None:
        return {
            'status': 'error',
            'iot_id': iot_id,
            'error': 'image could not be decoded'
        }

    # look for similar faces in the collection and check if there is a 90% match
    response = rekognition.search_faces_by_image(
        CollectionId='pfusch-collection',
        QualityFilter='NONE',
        Image={
            'Bytes': image_base64
        },
        FaceMatchThreshold=70
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
    