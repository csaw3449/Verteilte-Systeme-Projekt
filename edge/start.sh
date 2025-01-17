sudo zip -r ./file_upload/weights.zip ./yolo4weights
terraform apply -auto-approve
host=$(terraform show | grep public_dns | sed -n 's/.*"\([^"]*\)".*/\1/p')

# Install Docker dependencies on EC2
# ssh -o "StrictHostKeyChecking=accept-new" -i "EC2-key.pem" ec2-user@$host sudo yum install python3-pip -y
ssh -i "EC2-key.pem" ec2-user@$host sudo yum install python3-pip
ssh -i "EC2-key.pem" ec2-user@$host pip3 install boto3 opencv-python-headless numpy botocore matplotlib
ssh -i "EC2-key.pem" ec2-user@$host sudo unzip weights.zip

# Connect to ssh
ssh -i "EC2-key.pem" ec2-user@$host 
echo $host

# Auto destroy EC2 on command 'exit'
#terraform destroy -auto-approve


