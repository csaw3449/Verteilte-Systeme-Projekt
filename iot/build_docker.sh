# Build docker iot container and save iot.tar in terraform/fileupload/
sudo docker build -t iot ./docker_files
sudo docker save -o ./terraform_iot/file_upload/iot.tar iot
sudo chmod 644 ./terraform_iot/file_upload/iot.tar