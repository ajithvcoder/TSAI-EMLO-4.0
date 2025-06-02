
Build docker images and push to aws ecr first

Install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
Executing docker install script, commit: 7cae5f8b0decc17d6571f9f52eb840fbc13b2737
<...>



docker build -t ui-server -f Dockerfile.ui-server .

docker run -it  --network my_network -v /home/ajith/mlops/course/emlo_play/emlo4-s18/E2EMLOps/K8SDeploy/src/ui-server/ui:/opt/src -p3000:3000 ui-server bash

docker build -t web-server -f Dockerfile.web-server .
docker run -it --network my_network -v /home/ajith/mlops/course/emlo_play/emlo4-s18/E2EMLOps/K8SDeploy/src/web-server:/opt/src -p9090:9090 web-server bash

uvicorn server:app --host 0.0.0.0 --port 9090

docker build -t model-server -f Dockerfile.model-server .
docker run -it --network my_network -v /home/ajith/mlops/course/emlo_play/emlo4-s18/E2EMLOps/K8SDeploy/src/model-server:/opt/src -p8080:8080 model-server bash
uvicorn server:app --host 0.0.0.0 --port 8080

docker network create my_network

First make above ones working and then proceed to kubernetes


docker tag model-server:latest 306093656765.dkr.ecr.ap-south-1.amazonaws.com/a18/model-server:latest

docker tag ui-server:latest 306093656765.dkr.ecr.ap-south-1.amazonaws.com/a18/ui-server:latest

docker tag web-server:latest 306093656765.dkr.ecr.ap-south-1.amazonaws.com/a18/web-server:latest

docker push 306093656765.dkr.ecr.ap-south-1.amazonaws.com/a18/web-server:latest

docker push 306093656765.dkr.ecr.ap-south-1.amazonaws.com/a18/ui-server:latest

aws iam get-policy-version --policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO --version-id v1


eksctl create iamserviceaccount   --name s3-list-sa   --cluster basic-cluster   --attach-policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO   --approve 	--region ap-south-1

 docker rmi -f  imageid

### Build docker image for model-onnx-server

- ```docker build -t model-onnx-server -f Dockerfile.model-onnx-server . --no-cache```
