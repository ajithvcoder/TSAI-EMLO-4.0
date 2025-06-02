Refer the main [Readme](../../README.md) both are same

Used g4dn.xlarge 16GB mem gpu with a small model("OFA-Sys/small-stable-diffusion-v0") and small image(256x256) generation size as the cost of g6.2xlarge + sd3 is too high its 4x cost per hour.
Also even for "OFA-Sys/small-stable-diffusion-v0" model a 1024x1024 needs 24GB GPU so its better to reduce the image generation size to 256x256 or even smaller since the objective is to deploy properly and infer

```
eksctl create cluster -f eks-cluster.yaml
```

IRSA for s3 usage

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

eksctl create iamserviceaccount --name s3-list-sa   --cluster basic-cluster   --attach-policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO   --approve --region ap-south-1

verify
kubectl get sa
aws s3 ls mybucket-emlo-mumbai

EBS

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

eksctl create iamserviceaccount \
  --name ebs-csi-controller-sa \
  --namespace kube-system \
  --region ap-south-1 \
  --cluster basic-cluster \
  --attach-role-arn arn:aws:iam::306093656765:role/AmazonEKS_EBS_CSI_DriverRole


eksctl create addon --name aws-ebs-csi-driver --cluster basic-cluster --service-account-role-arn arn:aws:iam::306093656765:role/AmazonEKS_EBS_CSI_DriverRole --region ap-south-1 --force

Model file preparation

```
take a g4dn.xlarge instance

aws configure

python download_small_model.py 
mkdir -p ../model-store
sh create_mar.sh 
sh upload_to_s3.sh 
```

uncomment the " ng-gpu-spot-1" in eks-cluster.yaml and do

eksctl create nodegroup --config-file=eks-cluster.yaml

<debug>
eksctl create nodegroup --config-file=eks-cluster.yaml  --cluster basic-cluster --name ng-spot-1
eksctl delete cluster -f eks-cluster.yaml --disable-nodegroup-eviction
eksctl delete nodegroup --cluster basic-cluster --name ng-gpu-spot-1
</debug>



Verify
kubectl get nodes -L node.kubernetes.io/instance-type
kubectl get sc
kubectl get sa -n kube-system

### ISTIO
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

kubectl create namespace istio-system
helm install istio-base istio/base --version 1.20.2  --namespace istio-system --wait
helm install istiod istio/istiod  --version 1.20.2  --namespace istio-system --wait
kubectl create namespace istio-ingress

<debug>
helm uninstall istio-ingress istio/gateway \
  --version 1.20.2 \
  --namespace istio-ingress \

helm install istio-ingress istio/gateway \
  --version 1.20.2 \
  --namespace istio-ingress \
  --set labels.istio=ingressgateway \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-type"=external \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-nlb-target-type"=ip \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"=internet-facing \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-attributes"="load_balancing.cross_zone.enabled=true" 


note: install loadbalancer which is few steps away -> wait for load balancer to become action then run below command-> only then it will become active
kubectl rollout restart deployment istio-ingress -n istio-ingress
kubectl get deployment.apps/istio-ingress  -n istio-ingress


### ALB (Assumed its already created from session-15)

eksctl create iamserviceaccount \
--cluster=basic-cluster \
--namespace=kube-system \
--name=aws-load-balancer-controller \
--attach-policy-arn=arn:aws:iam::306093656765:policy/AWSLoadBalancerControllerIAMPolicy \
--override-existing-serviceaccounts \
--region ap-south-1 \
--approve

helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName=basic-cluster --set serviceAccount.create=false --set serviceAccount.name=aws-load-balancer-controller

Use
if istio-ingress external ip is not assigned even after load-balancer becomes active use below command
kubectl rollout restart deployment istio-ingress -n istio-ingress

Install Metrics
https://medium.com/@cloudspinx/fix-error-metrics-api-not-available-in-kubernetes-aa10766e1c2f
# kubectl delete -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard --create-namespace --namespace kubernetes-dashboard


### KServe
# kubectl apply -f istio-kserve-ingress.yaml - done in helm

Install Cert Manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml


Install KServe using HELM
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.14.1

helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.14.1 \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.controller.gateway.ingressGateway.className=istio


### GPU installations
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia  && helm repo update
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v24.9.1
kubectl -n gpu-operator logs -f $(kubectl -n gpu-operator get pods | grep dcgm | cut -d ' ' -f 1 | head -n 1)

todo: if u force delete resources it will be running in background so check cloud formation or ec2 instances

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve
eksctl create iamserviceaccount \
	--cluster=basic-cluster \
	--name=s3-read-only \
	--attach-policy-arn=arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
	--override-existing-serviceaccounts \
	--region ap-south-1 \
	--approve

-- not able to move to helm
kubectl apply -f s3-secret.yaml 

kubectl patch serviceaccount s3-read-only -p '{"secrets": [{"name": "s3-secret"}]}'

--- kubectl apply -f sd3-isvc.yaml --moved to helm
# -f fastapi-helm/values-prod.yaml
helm install fastapi-release-default fastapi-helm --values fastapi-helm/values.yaml 
helm uninstall fastapi-release-default
kubectl get all


kubectl label namespace default app.kubernetes.io/managed-by=Helm
kubectl annotate namespace default meta.helm.sh/release-name=fastapi-release-default
kubectl annotate namespace default meta.helm.sh/release-namespace=default
helm install fastapi-release-default fastapi-helm --values fastapi-helm/values.yaml --namespace default

Deleteion
helm uninstall fastapi-release-default  (may through an error as we have did some hacky change for helm install fastapi-release-default)
kubectl delete secret sh.helm.release.v1.fastapi-release-default.v1  -n default (this will completely uninstall helm's particular resources)
eksctl delete cluster -f eks-cluster.yaml --disable-nodegroup-eviction

Wait paitently see all deletion is successfull in cloud formation and then close the system because some times
the deletion gets failed so at backend something would be running and it may cost you high
-------------------------------------------------------------------------------------------------------------------
<docker build>
docker build -t a16/web-server -f Dockerfile.web-server  . --no-cache

kubectl exec -it ui-server-59cb8d9f96-8559p      -- /bin/bash
uvicorn server:app --host 0.0.0.0 --port 80

host three things
model, web and ui in docker and infer and see okay ?

curl -X POST https://turbo-tribble-wq6jwjr67w25vp6-9090.app.github.dev/generate_image \
-H "Content-Type: application/json" \
-d '{"text": "generate image"}'

curl -X POST http://0.0.0.0:9090/generate_image \
-H "Content-Type: application/json" \
-d '{"text": "generate image"}'

curl -X POST "http://0.0.0.0:9090/generate_image?text=horseriding"
curl -X POST "http://web-server-service/generate_image?text=horseriding"
http://web-server-service

todo: 
debugging new image with delete command
 kubectl delete pods/web-server-669c6cc8f8-zjvsd 


 Output of kubectl get all -A
Manifest Files used for deployments
Kiali Graph of the Deployment
GPU Usage from Grafana and Prometheus while on LOAD
Logs of your torchserve-predictor
5 Outputs of the SD3 Model
Make sure you copy the logs of torchserve pod while the model is inferencing

todo:
book train