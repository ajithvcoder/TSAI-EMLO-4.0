### EMLO - Session - 17 Class work
 
Create Cluster

- `eksctl create cluster -f eks-cluster.yaml`

```
<debug>
Deletion
- `eksctl delete cluster -f eks-cluster.yaml --disable-nodegroup-eviction`
</debug>
```

Install metric server properly
- [Metric server installation follow this](https://medium.com/@cloudspinx/fix-error-metrics-api-not-available-in-kubernetes-aa10766e1c2f)

**KNative**

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-core.yaml
```

**ISTIO** 

```
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/net-istio.yaml
```

**Knative serving**

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

**S3 Setup**

Create S3 Service Account and Create IRSA for S3 Read Only Access
Note: These are already done in previous assignments i.e the policy creation here we are only attaching the policy

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

Test if `vit-classifier` works fine with all m1, m2 and m3 models

- `kubectl apply -f vit-classifier.yaml`

or you can use `cat-classifier.yaml` or any

Check if everything works and delete it, we need to setup prometheus and grafana

- `python test_kserve_imagenet.py`

Ingress details

```
kubectl get isvc
kubectl get svc -n istio-system
````

Delete classifier after testing

- `kubectl delete -f vit-classifier.yaml`


**Prometheus**

```
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


verify

- `kubectl get isvc`

ArgoCD setup

- `kubectl create namespace argocd`
- `kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml`

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

Note: I have used [canary-argocd-kserve-repo](https://github.com/ajithvcoder/emlo4-session-17-ajithvcoder-canary-argocd-kserve) repo for deployment of argoCD apps. Please refer to `https://github.com/ajithvcoder/emlo4-session-17-ajithvcoder-canary-argocd-kserve` for argocd repo structure

- Have s3-secret.yaml file in argo-apps/s3-secret.yaml folder and update it with your AWS credentails i.e `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

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

- Create the repo before you start and update the repo url in `argo-apps/models.yaml` file

```
<debug>
Delete argocd deployments
kubectl get app -n argocd
kubectl patch app model-deployments  -p '{"metadata": {"finalizers": ["resources-finalizer.argocd.argoproj.io"]}}' --type merge -n argocd
kubectl delete app model-deployments -n argocd
</debug>
```


**Argocd deployment**

- Also have s3-secret.yaml file in argo-apps folder

Have only food-classifer.yaml in model-deployments folder
```
    minReplicas: 1
    maxReplicas: 1
    containerConcurrency: 1
    # canaryTrafficPercent: 0
```

- Git push and Go to argocd and check sync status by refershing
- kubectl apply -f argo-apps
- Deploy argocd app - `kubectl apply -f argo-apps`
- Go to argocd and check sync status by refershing
kubectl apply -f argo-apps
Update your loadbalancer URL in `load_kserve_in.py` file. For sample test use `test_kserve_imagenet.py`

Do load test
- `python load_kserve_in.py -c 2`
- Check grafana dashboard

Add cat-classifier.yaml
```
    minReplicas: 1
    maxReplicas: 1
    containerConcurrency: 1
    canaryTrafficPercent: 30
```
- Git push and Go to argocd and check sync status by refershing
- check `kubectl get isvc` percentage of canary
- Go to argocd and check sync status by refershing

Do load test
- `python load_kserve_in.py -c 2`
- Check grafana dashboard

increase cat-classifier.yaml

```
    minReplicas: 2
    maxReplicas: 2
    containerConcurrency: 2
    canaryTrafficPercent: 100
```
- Git push and Go to argocd and check sync status by refershing
- check `kubectl get isvc` percentage of canary
- Go to argocd and check sync status by refershing

Do load test
- `python load_kserve_in.py -c 2`
- Check grafana dashboard

<-->
Add dog-classifier.yaml
```
    minReplicas: 1
    maxReplicas: 1
    containerConcurrency: 1
    canaryTrafficPercent: 30
```
decrease cat-classifier.yaml
```
    minReplicas: 2
    maxReplicas: 2
    containerConcurrency: 2
    canaryTrafficPercent: 70
```
- Git push and Go to argocd and check sync status by refershing
- check `kubectl get isvc` percentage of canary
- Go to argocd and check sync status by refershing

Do load test
- `python load_kserve_in.py -c 2`
- Check grafana dashboard

increase dog-classifier.yaml
```
    minReplicas: 2
    maxReplicas: 2
    containerConcurrency: 2
    canaryTrafficPercent: 100
```
decrease cat-classifier.yaml
```
    minReplicas: 2
    maxReplicas: 2
    containerConcurrency: 2
    canaryTrafficPercent: 0
```
- Git push and Go to argocd and check sync status by refershing
- check `kubectl get isvc` percentage of canary
- Go to argocd and check sync status by refershing

Do load test
- `python load_kserve_in.py -c 2`
- Check grafana dashboard


**Delete argocd deployments**

Verify app name
- `kubectl get app -n argocd`

Delete cascade
- `kubectl patch app model-deployments  -p '{"metadata": {"finalizers": ["resources-finalizer.argocd.argoproj.io"]}}' --type merge -n argocd`
- `kubectl delete app model-deployments -n argocd`

**Deletion of cluster**
- `eksctl delete cluster -f eks-cluster.yaml --disable-nodegroup-eviction`

**Wait paitently see all deletion is successfull in aws cloud formation stack page and then close the system because some times
the deletion gets failed so at backend something would be running and it may cost you high**

**If you are triggering a spot instance manually with `peresistent` type ensure that both the spot request is cancelled manually
and the AWS instance is terminated finally**
