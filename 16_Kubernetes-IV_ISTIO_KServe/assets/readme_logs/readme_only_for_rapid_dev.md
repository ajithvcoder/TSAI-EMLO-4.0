This file was used for developing and debugging the assignment works

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
helm uninstall istio-ingress --namespace istio-ingress

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

Just run below in linux terminal
```
for ADDON in kiali jaeger prometheus grafana
do
    ADDON_URL="https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/$ADDON.yaml"
    kubectl apply -f $ADDON_URL
done
```

port forwarding and visuvalization

kubectl port-forward svc/kiali 20001:20001 -n istio-system

# Get to the Prometheus UI
kubectl port-forward svc/prometheus 9090:9090 -n istio-system

# Visualize metrics in using Grafana
kubectl port-forward svc/grafana 3000:3000 -n istio-system

kubectl get pods,svc -n istio-system
kubectl get pods,svc -n istio-ingress

kubectl get pods,svc -n istio-system
 kubectl get pods,svc -n istio-ingress

kubectl label namespace default istio-injection=enabled

kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null || \
  { kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=v1.2.0" | kubectl apply -f -; }

Till now 
kubectl get pods,svc -n istio-ingress
above command should have a external ip assigned

### KServe
kubectl apply -f istio-kserve-ingress.yaml

Metrics API
<!-- kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml -->

Install Cert Manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml

Install KServe using HELM
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.14.1

helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.14.1 \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.controller.gateway.ingressGateway.className=istio

### Testing K-serve and istio because it could take more cost if we test direclty as mdoel loading time takes more
```
kubectl apply -f iris.yaml

Wait for 30 seconds util the service becomes true in below command

kubectl get inferenceservices sklearn-iris

SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}' | cut -d "/" -f 3)


export INGRESS_HOST=$(kubectl -n istio-ingress get service istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-ingress get service istio-ingress -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')



<debug>
kubectl delete service istio-ingress -n istio-ingress
kubectl apply -f istio-kserve-ingress.yaml
kubectl delete pod -n istio-system -l istio=ingressgateway

```
echo $INGRESS_HOST:$INGRESS_PORT
```

Above command should give something like this else check loadbalcer is active and istio-ingress is reinstalled 
a93c9a652530148ba8bf451bf39605f3-7dcc8712d84cd905.elb.us-west-2.amazonaws.com:80

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json

You should get a output like "{"predictions":[1,1]}" check the terminal right bottom corner as its not printed in a seperate line u
might miss it. if the resposne is 200 there is a high chance its present. 

```
http://k8s-istioing-istioing-85fb5f5516-1b5e523e9a7b0701.elb.ap-south-1.amazonaws.com:80/v1/models/sklearn-iris:predict
```
```
kubectl delete -f iris.yaml

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
kubectl apply -f s3-secret.yaml
kubectl patch serviceaccount s3-read-only -p '{"secrets": [{"name": "s3-secret"}]}'

kubectl apply -f sd3-isvc.yaml

<debug>
kubectl get pods,svc -n istio-ingress
kubectl get all
kubectl get torchserve-sd3-predictor-bdfbb4b7d-pwjsm   
kubectl describe replicaset.apps/torchserve-sd3-predictor-bdfbb4b7d-bdvgb  
kubectl describe deployment.apps/torchserve-sd3-predictor   
</debug>

kubectl logs pod/torchserve-sd3-predictor-659ffd97f8-rgqhs  -c storage-initializer
