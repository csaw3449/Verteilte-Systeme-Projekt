import boto3.exceptions
import boto3
import base64
import time

# outside of the lambda function to avoid creating a new client for every invocation (might be changed)
rekognition = boto3.client('rekognition', region_name='us-east-1')

# this lambda function is triggered by the edge layer and the pictures a located in a container
def lambda_handler(event, context):
    # get the image and iot_id from the event
    cloud_start = time.time()
    image = event['image']
    iot_id = event['iot_id']
    
    # decode the image from base64, if it not works remove this lines
    image_bytes = base64.b64decode(image)

    if image_bytes is None:
        return {
            'status': 'error',
            'iot_id': iot_id,
            'error': 'image could not be decoded',
            'iot_start': event['iot_start'],
            'edge_start1' : event['edge_start1'],
            'edge_end1' : event['edge_end1'],
            'cloud_start': cloud_start,
            'cloud_end': time.time()
        }

    # look for similar faces in the collection and check if there is a 90% match
    try:
        response = rekognition.search_faces_by_image(
            CollectionId='pfusch-collection',
            QualityFilter='NONE',
            Image={
                'Bytes': image_bytes
            },
            
            FaceMatchThreshold=70
        )
    except rekognition.exceptions.InvalidParameterException as e:
        return {
            'status': 'no_face',
            'iot_id': iot_id,
            'error': 'no face detected',
            'iot_start': event['iot_start'],
            'edge_start1' : event['edge_st66666art1'],
            'edge_end1' : event['edge_end1'],
            'cloud_start': cloud_start,
            'cloud_end': time.time()
        }
    except Exception as e:
        return {
            'status': 'error',
            'iot_id': iot_id,
            'error': str(e), 
            'iot_start': event['iot_start'],
            'edge_start1' : event['edge_start1'],
            'edge_end1' : event['edge_end1'],
            'cloud_start': cloud_start,
            'cloud_end': time.time()
        }

    # if there is a match, return the status known otherwise unknown
    if len(response['FaceMatches']) > 0:
        return {
            'status': 'known',
            'iot_id': iot_id,
            'iot_start': event['iot_start'],
            'edge_start1' : event['edge_start1'],
            'edge_end1' : event['edge_end1'],
            'cloud_start': cloud_start,
            'cloud_end': time.time()
        }
    else:
        return {
            'status': 'unknown',
            'iot_id': iot_id, 
            'iot_start': event['iot_start'],
            'edge_start1' : event['edge_start1'],
            'edge_end1' : event['edge_end1'],
            'cloud_start': cloud_start,
            'cloud_end': time.time()
        }
    