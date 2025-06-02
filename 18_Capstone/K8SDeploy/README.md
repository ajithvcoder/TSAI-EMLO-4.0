### Deploy Kubernetes app - Manual - Clean version

Mostly followed [assignment-17](https://github.com/ajithvcoder/emlo4-session-17-ajithvcoder) prcedure and ALB installation and helm config files from [assignment-18](https://github.com/ajithvcoder/emlo4-session-16-ajithvcoder)

#### Configurations

Check aws configure

aws configure

EKSCTL install

```
# for ARM systems, set ARCH to: `arm64`, `armv6` or `armv7`
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH

curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

# (Optional) Verify checksum
curl -sL "<https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt>" | grep $PLATFORM | sha256sum --check

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz

sudo mv /tmp/eksctl /usr/local/bin
```

Use ssh for login

### Do Torchserve setup 

**Take a g4dn.xlarge instance.**

aws configure

```
python download_small_model.py 
mkdir -p ../model-store
sh create_mar.sh
sh upload_to_s3.sh
```

#### EKS configuration

Go into `src/eks-cluster-config` folder. It takes 7-15 minutes based on number of resources you have listed in `.yaml` file for cluster creation

```
eksctl create cluster -f eks-cluster.yaml
```

``` 
<debug-facts>
# only usefull during debugging

# Create a new nodegroup
# You can comment out a resource and later uncomment it and create the new nodegroup in same cluster
# This can save some cost of gpu nodes. But make sure to create before doing gpu related installations
eksctl create nodegroup --config-file=eks-cluster.yaml

# Delete nodegroup
eksctl delete nodegroup --cluster basic-cluster --name ng-gpu-spot-1

# Delete cluster, Also check in "AWS Cloud Formation" as sometimes even in CLI if its success
in "AWS Cloud Formation" you will get deletion failed.
eksctl delete cluster -f eks-cluster.yaml --disable-nodegroup-eviction

</debug-facts>
```

Check instances which is in EC2

For this to work the defualt ssh should have been configured and it helps in establishing connection with EC2

```
ssh ec2-user@43.204.212.5
kubectl config view
kubectl get all
```

Install metric server properly

[Metric server installation follow this](https://medium.com/@cloudspinx/fix-error-metrics-api-not-available-in-kubernetes-aa10766e1c2f)

#### KNative

```
 -- ignore -kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-core.yaml
```

#### ISTIO

```
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/net-istio.yaml
```

#### Knative serving

```
kubectl patch configmap/config-domain \
      --namespace knative-serving \
      --type merge \
      --patch '{"data":{"emlo.tsai":""}}'

kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-hpa.yaml
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml

# verify
kubectl get all -n cert-manager
```

Wait for cert manager pods to be ready
```
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve.yaml
```

Wait for KServe Controller Manager to be ready
```
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve.yaml
kubectl get all -n kserve

# Wait and check if all pods are running in kserve
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve-cluster-resources.yaml
```

### S3 Setup
Create S3 Service Account and Create IRSA for S3 Read Only Access Note: These are already done in previous assignments i.e the policy creation here we are only attaching the policy

# TODO: Check AmazonS3ReadOnlyAccess

```
eksctl create iamserviceaccount \
--cluster=basic-cluster \
--name=s3-read-only \
--attach-policy-arn=arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
--override-existing-serviceaccounts \
--region ap-south-1 \
--approve
```

```
kubectl apply -f s3-secret.yaml
kubectl patch serviceaccount s3-read-only -p '{"secrets": [{"name": "s3-secret"}]}'
```

Test if `imagenet-classifier` works fine with all m1, m2 and m3 models

- kubectl apply -f imagenet-classifier-1.yml

Ingress details

- `kubectl get isvc`
- `kubectl get svc -n istio-system`

Take the url from above command and use it in `test_kserve_imagenet.py` file

Check if everything works and delete it, we need to setup prometheus and grafana

- `python test_kserve_imagenet.py`

Delete classifier after testing

- `kubectl delete -f imagenet-classifier.yml`

### Kubernetes Dashboard

Kubernetes Dashboard

helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/

helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard --create-namespace --namespace kubernetes-dashboard

kubectl label namespace default istio-injection=enabled

**ALB**

Assumed policy(AWSLoadBalancerControllerIAMPolicy) is already created from session-15 or else refer it

```
eksctl create iamserviceaccount \
--cluster=basic-cluster \
--namespace=kube-system \
--name=aws-load-balancer-controller \
--attach-policy-arn=arn:aws:iam::306093656765:policy/AWSLoadBalancerControllerIAMPolicy \
--override-existing-serviceaccounts \
--region ap-south-1 \
--approve
```

- helm repo add eks https://aws.github.io/eks-charts
- helm repo update
- helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName=basic-cluster --set serviceAccount.create=false --set serviceAccount.name=aws-load-balancer-controller

Verify
- `kubectl get pods,svc -n istio-system`

**Prometheus**

```
mkdir other-setup
cd other-setup
git clone --branch release-0.14 https://github.com/kserve/kserve.git
cd kserve
kubectl apply -k docs/samples/metrics-and-monitoring/prometheus-operator
kubectl wait --for condition=established --timeout=120s crd/prometheuses.monitoring.coreos.com
kubectl wait --for condition=established --timeout=120s crd/servicemonitors.monitoring.coreos.com
kubectl apply -k docs/samples/metrics-and-monitoring/prometheus
```

```
kubectl patch configmaps -n knative-serving config-deployment --patch-file qpext_image_patch.yaml
```

Set max nodes because if you give more request and max is not set it may scale more

```
eksctl scale nodegroup --cluster=basic-cluster --nodes=6 ng-spot-3 --nodes-max=6
eksctl get nodegroup --cluster basic-cluster --region ap-south-1 --name ng-spot-3
```

```
kubectl port-forward service/prometheus-operated -n kfserving-monitoring 9090:9090
```

**Grafana**

```
kubectl create namespace grafana

helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm install grafana grafana/grafana --namespace grafana --version 8.8.4
```

```
kubectl get secret --namespace grafana grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
kubectl port-forward svc/grafana 3000:80 -n grafana
```

Setup dashboard for grafana

- Go to Connections-> Add data source -> Prometheus -> Add this prometheus url `http://prometheus-operated.kfserving-monitoring.svc.cluster.local:9090` -> save and test

- Go to Dashboards -> New -> import -> download the json file from here `https://grafana.com/grafana/dashboards/18032-knative-serving-revision-http-requests/` -> upload the json to the specified place


*For Helm Debugging commands refer emlo assignment 16

### Helm, ArgoCD and Canary Deployment 

ArgoCD setup

`kubectl create namespace argocd`
`kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml`

ArgoCD executable file setup

```
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64
```

Get Argocd password for login

- `argocd admin initial-password -n argocd`

Check if you are able to access the UI, in codespaces i am not able to forward and access for argocd UI but in local its working

- `kubectl port-forward svc/argocd-server -n argocd 8080:443`

**Argo CD deployment**

- Have s3-secret.yaml file in argo-apps/s3-secret.yaml folder and update it with your AWS credentails i.e AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

```
apiVersion: v1
kind: Secret
metadata:
  name: s3creds
  annotations:
     serving.kserve.io/s3-endpoint: s3.ap-south-1.amazonaws.com # replace with your s3 endpoint e.g minio-service.kubeflow:9000
     serving.kserve.io/s3-usehttps: "1" # by default 1, if testing with minio you can set to 0
     serving.kserve.io/s3-region: "ap-south-1"
     serving.kserve.io/s3-useanoncredential: "false" # omitting this is the same as false, if true will ignore provided credential and use anonymous credentials
type: Opaque
stringData: # use `stringData` for raw credential string or `data` for base64 encoded string
  AWS_ACCESS_KEY_ID: AKXXXXXXXXXXXXXXXXXXXXX
  AWS_SECRET_ACCESS_KEY: "RQHBUNBSJNINQONUKNUKXXXXXX+XQIWOW"

---

apiVersion: v1
kind: ServiceAccount
metadata:
  name: s3-read-only
secrets:
- name: s3creds
```

Argo CD deployments

```
kubectl apply -f argo-apps
```

Create the repo before you start and update the repo url in argo-apps/models.yaml file

When everything is running and pods are up

**Load testing**

`python3 test_load_vegfruits_2.py --url "http://a250c292ee6114b8e9d6f23f6d8690bc-1957734331.ap-south-1.elb.amazonaws.com/v1/models/vegfruits-classifier:predict" --requests 10 --workers 2`

UI URL will be available in loadbalancer in EC2 

`http://k8s-default-uiserver-3620c3afba-764943795.ap-south-1.elb.amazonaws.com/`

```
<debug>
Delete argocd deployments
kubectl get app -n argocd
kubectl patch app fastapi-helm -p '{"metadata": {"finalizers": ["resources-finalizer.argocd.argoproj.io"]}}' --type merge -n argocd
kubectl delete app fastapi-helm -n argocd

# if not deleted properly
kubectl delete app fastapi-helm
kubectl patch app fastapi-helm -p '{"metadata": {"finalizers": null}}' --type merge -n argocd
</debug>
```

```
<debug>
kubectl delete deployment ui-server -n default
kubectl delete deployment web-server -n default
kubectl delete deployment fastapi-mamba-model-2-predictor-00001-deployment -n default
</debug>
argocd app actions run fastapi-helm  restart --kind Deployment --all
```

Deployment

Update this ingress value in `model-server.cm.yml`

![model-server](../assets/deployment/snap_model_server_url.png)

UI server url

![ui-server](../assets/deployment/snap_ui-server.png)
