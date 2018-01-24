#/bin/bash

if [ ! -d "../CamVid" ]; then
  echo "Please put your Camvid directory here."
  echo "You can download it from https://github.com/alexgkendall/SegNet-Tutorial.git by git."
  exit
fi

if [ ! -d "cudnn-9.1-linux-x64-v7.tgz" ]; then
    echo "Please put cudnn-9.1-linux-x64-v7.tgz here."
    echo "Your can download it from NVIDIA website."
    exit
fi

read -p "Please enter the ip address of your ECS: " address
read -p "Please enter the username of your ECS: " username

function upload() {
    scp -r $1 $username@$address:~
}

upload cudnn-9.1-linux-x64-v7.tgz
upload cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
upload CamVid
