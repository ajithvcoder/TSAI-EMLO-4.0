in torch serve just host a normal flask api that accepts a image and replies that {"dog": 0.956}
build everything and then move to g4.dxlarge

Helm charts
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh

todo:
build docker images and host and check and then push to aws then go for lunch

deploy this OFA-Sys/small-stable-diffusion-v0 model


docker build -t model-server -f Dockerfile.model-server .
docker build -t web-server -f Dockerfile.web-server .
docker build -t ui-server -f Dockerfile.ui-server .
docker network create my_network

docker run -it --network my_network -v /workspaces/codespaces-blank/src/web-server:/opt/src -p80:80 web-server bash


todo:
remove the old code for ui and web server and model server and check the docker container if its working local and push it


$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh