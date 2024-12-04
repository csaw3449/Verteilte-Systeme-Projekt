# Build docker iot container and save iot.tar in terraform/fileupload/
sudo docker build -t iot ./iot
sudo docker save -o ./terraform/file_upload/iot.tar iot
sudo chmod 644 ./terraform/file_upload/iot.tar

