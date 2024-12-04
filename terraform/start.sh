terraform apply -auto-approve
host=$(terraform show | grep public_dns | sed -n 's/.*"\([^"]*\)".*/\1/p')

echo "Exportierte Host Adresse für EC2": $host

export TF_host_address = $host

# Install Docker dependencies on EC2
# ssh -o "StrictHostKeyChecking=accept-new" -i "EC2-key.pem" ec2-user@$host sudo yum install python3-pip -y
ssh -i "EC2-key.pem" ec2-user@$host sudo yum install -y docker
ssh -i "EC2-key.pem" ec2-user@$host sudo service docker start
ssh -i "EC2-key.pem" ec2-user@$host sudo usermod -a -G docker ec2-user
ssh -i "EC2-key.pem" ec2-user@$host sudo systemctl enable docker
ssh -i "EC2-key.pem" ec2-user@$host sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
ssh -i "EC2-key.pem" ec2-user@$host sudo chmod +x /usr/local/bin/docker-compose
ssh -i "EC2-key.pem" ec2-user@$host docker-compose up -d --build


# Connect to ssh
ssh -i "EC2-key.pem" ec2-user@$host 
echo $host

# Auto destroy EC2 on command 'exit'
terraform destroy -auto-approve


