import boto3

# outside of the lambda function to avoid creating a new client for every invocation (might be changed)
rekognition = boto3.client('rekognition')

# this lambda function is triggered by the edge layer and the pictures a located in a container
def lambda_handler(event, context):
    # get the image and iot_id from the event
    image = event['image']
    iot_id = event['iot_id']

    # look for similar faces in the collection and check if there is a 90% match
    response = rekognition.search_faces_by_image(
        CollectionId='pfusch-collection',
        Bucket='pfusch-bucket',
        Image={
            'Bytes': image
        },
        FaceMatchThreshold=90
    )

    # if there is a match, return the iot_id
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