# EMLOV4-Session-02 Assignment - PyTorch Docker Assignment - Docker - I

## Assignment Overview

In this assignment, you will:

1. Create a Dockerfile for a PyTorch (CPU version) environment. (Done)
2. Keep the size of your Docker image under 1GB (uncompressed). (Done)
3. Train any model on the MNIST dataset inside the Docker container. (Done)
4. Save the trained model checkpoint to the host operating system. (Done)
5. Add an option to resume model training from a checkpoint. (Done)

## Development method

- Requirement [1] and [2]

    Created docker image with pytorch cpu version under 1GB using python:3.9-slim image from docker hub and using --no-cache-dir key word during pip install. This keyword makes sure that there is no copy of package stored in docker container

- Requirement [3]
    
    Took the training script from pytorch examples

- Requirement [4]

    Since the -v keyword is used in testing script(grading.sh) it takes care of the process of saving model to host os.

- Requirement [5]

    Added "--resume" CLI parameter using argparse library and by default "model_checkpoint.pth" file is used for loading the model when training is resumed.

## Learnings

- it seems that when i tried to use python-alphine i was not able to use pytorch in it as pthreads is not avaiable in alphine version. when i py-alphine + torch it was only about 854 MB. Tried nightly torch build but it was going beyong 950 MB.

## Docker hub link
- [pytorch-docker-1-image](https://hub.docker.com/r/ajithvallabai/emlov4-pytorch-docker-1)

## Group members
1. Ajith Kumar V (myself)
2. Aakash Vardhan
3. Anvesh Vankayala
4. Manjunath Yelipeta
5. Abhijith Kumar K P
6. Sabitha Devarajulu

