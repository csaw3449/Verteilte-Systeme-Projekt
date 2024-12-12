# Build docker iot container and save iot.tar in terraform/fileupload/
sudo docker build -t iot ./iot
sudo docker save -o ./terraform_iot/file_upload/iot.tar iot
sudo chmod 644 ./terraform_iot/file_upload/iot.tar

# Build docker iot container and save iot.tar in terraform/fileupload/
sudo docker build -t edge ./edge
sudo docker save -o ./terraform_edge/file_upload/edge.tar edge
sudo chmod 644 ./terraform_edge/file_upload/edge.tar
