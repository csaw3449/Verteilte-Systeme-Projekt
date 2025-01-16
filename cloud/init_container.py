from botocore.exceptions import ClientError
import os
import boto3

try: 
    prof_name =os.getenv('PROFILE_NAME')
except KeyError:
    print('Please set the profile name')
    exit(1)

def create_collection(bucket, collection_id, path_to_images):
    session = boto3.Session(profile_name=prof_name, region_name='us-east-1')
    client = session.client('rekognition')

    try:
        client.describe_collection(CollectionId=collection_id)
        print('Collection already exists')
    except ClientError as e:
        # Create a collection
        print('Creating collection:' + collection_id)
        try:
            response = client.create_collection(CollectionId=collection_id)
            print('Collection ARN: ' + response['CollectionArn'])
            print('Status code: ' + str(response['StatusCode']))
            print('Done...')
            # write the information to the CollectionConfig file
            with open('CollectionConfig.cfg', 'w') as file:
                file.write('CollectionId=' + collection_id + '\n')
                file.write('CollectionArn=' + response['CollectionArn'] + '\n')
                file.write('Bucket=' + bucket + '\n')
        except ClientError as e:
            print(e)
            return False

    # read images from a given path and add them to the collection
    indexed_faces_count = 0
    with os.scandir(path_to_images) as entries:
        for entry in entries:
            if entry.is_file():
                response = client.index_faces(CollectionId=collection_id,
                Image={'S3Object': {'Bucket': bucket, 'Name': entry.name}},
                                  ExternalImageId=entry.name,
                                  MaxFaces=1,
                                  QualityFilter="NONE",
                                  DetectionAttributes=['ALL'])
                indexed_faces_count += len(response['FaceRecords'])
                print(f"Faces indexed from {entry.name}: " + str(len(response['FaceRecords'])))
    print('Faces indexed count: ' + str(indexed_faces_count))

def create_bucket(bucket_name):
    session = boto3.Session(profile_name=prof_name, region_name='us-east-1')
    s3 = session.client('s3')
    try:
        s3.head_bucket(Bucket=bucket_name)
        print("Bucket exists and is accessible.")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ('404', 'NoSuchBucket'):
            # Create the missing bucket
            try:
                s3.create_bucket(Bucket=bucket_name)
                print(f"Bucket '{bucket_name}' created.")
                return True
            except ClientError as create_err:
                print(create_err)
                return False
        elif error_code == '403':
            print(f"Bucket '{bucket_name}' exists but access is denied.")
            return False
        else:
            print(e)
            return False

def add_images_to_bucket(bucket_name, path_to_images):
    session = boto3.Session(profile_name=prof_name, region_name='us-east-1')
    s3 = session.client('s3')
    with os.scandir(path_to_images) as entries:
        for entry in entries:
            if entry.is_file():
                s3.upload_file(path_to_images + '/' + entry.name, bucket_name, entry.name)
                print('Image uploaded to bucket: ' + entry.name)

if __name__ == '__main__':
    bucket_name = input('Enter the s3 bucket name: ')
    path_to_images = os.getenv('PATH_TO_IMAGES')
    if path_to_images is None:
        print('Please set the path to the image folder with the environment variable PATH_TO_IMAGES="path"')   
        exit(1)
    create_bucket(bucket_name)
    add_images_to_bucket(bucket_name, path_to_images)
    create_collection(bucket_name, 'pfusch-collection2', path_to_images)