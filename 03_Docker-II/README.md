### EMLOV4-Session-03 Assignment - PyTorch Docker Assignment - Docker - II

### Pytorch - Docker - II

#### Requirements:

1. You’ll need to use this model and training technique (MNIST Hogwild): https://github.com/pytorch/examples/tree/main/mnist_hogwildLinks to an external site.
2. Set Num Processes to 2 for MNIST HogWild
3. Create three services in the Docker Compose file: train, evaluate, and infer.
4. Use a shared volume called mnist for sharing data between the services.
5. The train service should:
Look for a checkpoint file in the volume. If found, resume training from that checkpoint. Train for ONLY 1 epoch and save the final checkpoint. Once done, exit.
6. The evaluate service should:
Look for the final checkpoint file in the volume. Evaluate the model using the checkpoint and save the evaluation metrics in a json file. Once done, exit.
7. Share the model code by importing the model instead of copy-pasting it in eval.pyLinks to an external site.
8. The infer service should:
9. Run inference on any 5 random MNIST images and save the results (images with file name as predicted number) in the results folder in the volume. Then exit.
10. After running all the services, ensure that the model, and results are available in the mnist volume.

### Development method

1. Since we are going to use docker compose its better to create a common `model` folder to store `model` in the root and create 
seperate folders for each service and place their files in those folders.
2. Write the `train.py`, `eval.py`, `infer.py` and test it first itself.
3. After the scripts are ready, we need to mount the shared volume properly. Here we have used `mnist` in docker compose as 
a shared volume. Make sure you `name` the volume else it will take the default path value as prefix for volume name.
4. Since we need both `model` folder and the common volume `mnist` we need to mount two volumes for each service while running.
5. Using `docker compose` run the train service with `process=2` command and then run the eval service and then the infer service.
6. if you have mounted properly the output files would have been available in the shared folder. You can verify using below command `/opt/mount` location.
``` docker run --rm -it -v mnist:/opt/mount/model alpine /bin/sh ```
7. In the default code it was generating inference images with class id's so we need to change to index numbers to get 5 images as output.

### Learnings
1. Name the volume properly and mount the volume properly.
2. We can even mount more than one volume to a service and each service docker files can be placed under seperate folder for better readabilty and management .

### Group Members
1. Ajith Kumar V (myself)
2. Aakash Vardhan
3. Anvesh Vankayala
4. Manjunath Yelipeta
5. Abhijith Kumar K P
6. Sabitha Devarajulu

