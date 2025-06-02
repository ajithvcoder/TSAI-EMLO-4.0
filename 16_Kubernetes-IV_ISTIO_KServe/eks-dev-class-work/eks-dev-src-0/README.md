## Building only Sd3

### Cluster creation and attach s3 and ebs policies

eksctl create cluster -f eks-cluster.yaml

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

aws iam get-policy-version --policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO --version-id v1

eksctl create iamserviceaccount \
  --name s3-list-sa \
  --cluster basic-cluster \
  --attach-policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO \
  --approve \
	--region ap-south-1

kubectl describe sa s3-list-sa

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

---------------

eksctl create iamserviceaccount \
  --name ebs-csi-controller-sa \
  --namespace kube-system \
  --region ap-south-1 \
  --cluster basic-cluster \
  --attach-role-arn arn:aws:iam::306093656765:role/AmazonEKS_EBS_CSI_DriverRole


eksctl create addon --name aws-ebs-csi-driver --cluster basic-cluster --service-account-role-arn arn:aws:iam::306093656765:role/AmazonEKS_EBS_CSI_DriverRole --region ap-south-1 --force

### GPU Node add

comment out the gpu

eksctl create nodegroup --config-file=eks-cluster.yaml
kubectl get nodes -L node.kubernetes.io/instance-type

### ISTIO

Load balancer formation - ALB
eksctl create iamserviceaccount \
--cluster=basic-cluster \
--namespace=kube-system \
--name=aws-load-balancer-controller \
--attach-policy-arn=arn:aws:iam::306093656765:policy/AWSLoadBalancerControllerIAMPolicy \
--override-existing-serviceaccounts \
--region ap-south-1 \
--approve

helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName=basic-cluster --set serviceAccount.create=false --set serviceAccount.name=aws-load-balancer-controller

helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

kubectl create namespace istio-system
helm install istio-base istio/base \
  --version 1.20.2 \
  --namespace istio-system --wait
helm install istiod istio/istiod \
  --version 1.20.2 \
  --namespace istio-system --wait
kubectl create namespace istio-ingress


helm install istio-ingress istio/gateway \
  --version 1.20.2 \
  --namespace istio-ingress \
  --set labels.istio=ingressgateway \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-type"=external \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-nlb-target-type"=ip \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"=internet-facing \
  --set service.annotations."service\\.beta\\.kubernetes\\.io/aws-load-balancer-attributes"="load_balancing.cross_zone.enabled=true" 


helm ls -n istio-system
helm ls -n istio-ingress
helm ls -A

kubectl rollout restart deployment istio-ingress -n istio-ingress

check for active and url in ingress
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
for ADDON in kiali jaeger prometheus grafana
do
    ADDON_URL="https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/$ADDON.yaml"
    kubectl apply -f $ADDON_URL
done

Visualize Istio Mesh console using Kiali
kubectl port-forward svc/kiali 20001:20001 -n istio-system

Get to the Prometheus UI
kubectl port-forward svc/prometheus 9090:9090 -n istio-system

Visualize metrics in using Grafana
kubectl port-forward svc/grafana 3000:3000 -n istio-system
kubectl get pods,svc -n istio-system
kubectl get pods,svc -n istio-ingress
❯ kubectl get pods,svc -n istio-system

kubectl label namespace default istio-injection=enabled

kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null || \
  { kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=v1.2.0" | kubectl apply -f -; }

### Testing out Istio Installation

kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/bookinfo/platform/kube/bookinfo.yaml

<debug>
kubectl delete -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/bookinfo/platform/kube/bookinfo.yaml

kubectl exec "$(kubectl get pod -l app=ratings -o jsonpath='{.items[0].metadata.name}')" -c ratings -- curl -sS productpage:9080/productpage | grep -o "<title>.*</title>"

kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/bookinfo/gateway-api/bookinfo-gateway.yaml
<debug>
kubectl delete -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/bookinfo/gateway-api/bookinfo-gateway.yaml

export INGRESS_HOST=$(kubectl get gtw bookinfo-gateway -o jsonpath='{.status.addresses[0].value}')
export INGRESS_PORT=$(kubectl get gtw bookinfo-gateway -o jsonpath='{.spec.listeners[?(@.name=="http")].port}')
export GATEWAY_URL=$INGRESS_HOST:$INGRESS_PORT
kubectl get gtw bookinfo-gateway

You can get the LoadBalancer URL from service as well

kubectl get svc
curl -s "http://${GATEWAY_URL}/productpage" | grep -o "<title>.*</title>"
echo "http://${GATEWAY_URL}/productpage"

<debug>
Above url was accessible only at some times not sure even when running and ALB is active why its not loading in broswer or curling


### KServe
kubectl apply -f istio-kserve-ingress.yaml

Metrics API
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

Install Cert Manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml

Install KServe using HELM
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.14.1

helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.14.1 \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.controller.gateway.ingressGateway.className=istio

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

### S3 integration S3 Access for KServe

eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

Create IRSA for S3 Read Only Access

eksctl create iamserviceaccount \
	--cluster=basic-cluster \
	--name=s3-read-only \
	--attach-policy-arn=arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
	--override-existing-serviceaccounts \
	--region ap-south-1 \
	--approve

kubectl apply -f s3-secret.yaml
kubectl patch serviceaccount s3-read-only -p '{"secrets": [{"name": "s3-secret"}]}'

### Deploy SD3 on EKS with KServe


helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v24.9.1

Descibe the gpu pod
kubectl get node
Use ip from above command to check
kubectl describe node ip-192-168-33-217.ap-south-1.compute.internal 

kubectl -n gpu-operator logs -f $(kubectl -n gpu-operator get pods | grep dcgm | cut -d ' ' -f 1 | head -n 1)

kubectl apply -f sd3-isvc.yaml

Note: Better to do below in g6.2xlarge(24GB) gpu spot instance because the download size is too high and installationa and cpu also needs to be more memory and gpu also need more memory(16-17 GB)

pip install torch-model-archiver
pip install -r requirements.txt
huggingface-cli login
(give hugging face read token above)

python download_model.py

sh create_mar.sh
sh upload_to_s3.sh

kubectl logs pods/torchserve-sd3-predictor-54cf4b7f6f-v48zp    -c storage-initializer


kubectl logs pods/torchserve-sd3-predictor-54cf4b7f6f-dkdts

kubectl get isvc


export INGRESS_HOST=$(kubectl -n istio-ingress get service istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-ingress get service istio-ingress -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

Exporting names
MODEL_NAME=sd3
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve-sd3 -o jsonpath='{.status.url}' | cut -d "/" -f 3)

echo http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict
http://aee0d8fb7d6514f87a04ac80d194f1b3-c98fee301a70a638.elb.us-west-2.amazonaws.com:80/v1/models/sdxl:predict

kubectl get svc -n istio-ingress

kubectl get isvc

k8s-istioing-istioing-85fb5f5516-1b5e523e9a7b0701.elb.ap-south-1.amazonaws.com


<debugging for nodegroup handling (only for ajith)>
eksctl delete nodegroup --cluster basic-cluster --name ng-gpu-spot-1 --disable-nodegroup-eviction
eksctl create nodegroup --config-file=eks-cluster.yaml  --cluster basic-cluster --name ng-spot-1
g6.2xlarge
eksctl delete cluster -f cluster.yaml --disable-nodegroup-eviction

check if it works else go for small model

python test_sd3.py

### GPU

kubectl get ds -n kube-system
kubectl -n kube-system delete daemonset nvidia-device-plugin-daemonset

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia  && helm repo update

kubectl -n gpu-operator logs -f $(kubectl -n gpu-operator get pods | grep dcgm | cut -d ' ' -f 1 | head -n 1)

Prometheus

curl https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/prometheus.yaml -o prometheus.yaml

make the changes as in canvas and then do below

kubectl apply -f prometheus.yaml

Forwarding ports

kubectl port-forward svc/prometheus 9090:9090 -n istio-system
kubectl port-forward svc/grafana 3000:3000 -n istio-system
kubectl port-forward svc/kiali 20001:20001 -n istio-system