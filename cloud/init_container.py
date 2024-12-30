import logging as logger
from botocore.exceptions import ClientError
import os
import boto3
# Create a logger
logger = logger.getLogger(__name__)

try: 
    prof_name =os.getenv('PROFILE_NAME')
except KeyError:
    logger.error('Please set the profile name')
    exit(1)

def create_collection(bucket, collection_id, path_to_images):
    session = boto3.Session(profile_name=prof_name)
    client = session.client('rekognition')

    try:
        client.describe_collection(CollectionId=collection_id)
        logger.log('Collection already exists')
    except ClientError as e:
        # Create a collection
        print('Creating collection:' + collection_id)
        try:
            response = client.create_collection(CollectionId=collection_id)
            print('Collection ARN: ' + response['CollectionArn'])
            print('Status code: ' + str(response['StatusCode']))
            print('Done...')
        except ClientError as e:
            logger.error(e)
            return False

    # read images from a given path and add them to the collection
    with os.scandir(path_to_images) as entries:
        for entry in entries:
            if entry.is_file():
                print("Adding image: " + entry.name)
                indexed_faces_count = client.index_faces(Collection_id=collection_id,
                Image={'S3Object': {'Bucket': bucket, 'Name': entry.name}},
                                  ExternalImageId=entry.name,
                                  MaxFaces=1,
                                  QualityFilter="NONE",
                                  DetectionAttributes=['ALL'])
                print("Faces indexed: " + str(indexed_faces_count))

    print('All images are added to the collection')

def create_bucket(bucket_name):
    session = boto3.Session(profile_name=prof_name)
    s3 = session.client('s3')
    try:
        s3.head_bucket(Bucket=bucket_name)
        logger.log('Bucket already exists')
        return True
    except ClientError as e:
        try:
            s3.create_bucket(Bucket=bucket_name)
        except ClientError as e:
            logger.error(e)
            return False
        return True

if __name__ == '__main__':
    path_to_images = os.getenv('PATH_TO_IMAGES')
    if path_to_images is None:
        logger.error('Please set the path to the images')
        exit(1)
    create_bucket('pfusch-bucket')
    create_collection('pfusch-bucket', 'pfusch-collection', path_to_images)