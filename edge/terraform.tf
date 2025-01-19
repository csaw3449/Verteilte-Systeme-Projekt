terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }



  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-east-1"
}


resource "aws_security_group" "security" {
  name = "allow-ssh3"


  ingress {
    cidr_blocks = [
      "0.0.0.0/0"
    ]
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = -1
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app_server" {
  ami                  = "ami-0278a29a3eec44bbb"
  instance_type        = "t2.micro"
  iam_instance_profile = "LabInstanceProfile"

  key_name = "EC2-key"

  vpc_security_group_ids = [aws_security_group.security.id]

    provisioner "file" {
    source      = "${path.module}/file_upload/"
    destination = "/home/ec2-user/"

    connection {
      type        = "ssh"
      user        = "ec2-user" # Or whichever username your AMI uses
      private_key = file("${path.module}/EC2-key.pem")
      host        = self.public_ip
    }
  }


  tags = {
    Name = "Edge-Devices2"
  }
}
