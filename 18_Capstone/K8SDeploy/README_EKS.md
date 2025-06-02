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

EKS configuration -

eksctl create cluster -f eks-cluster.yaml

create new nodegroup
eksctl create nodegroup --config-file=eks-cluster.yaml

IRSA 

# Associates an IAM OIDC (OpenID Connect) provider with your EKS cluster. This association allows you to enable Kubernetes
# service accounts in your cluster to use IAM roles for fine-grained permissions.
eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve

# Create Iam policy
aws iam create-policy --policy-name S3ListTestEMLO --policy-document file://iam-s3-test-policy.json

# Verification
aws iam get-policy-version --policy-arn arn:aws:iam::ACCOUNT_ID:policy/S3ListTestEMLO --version-id v1

# Attach policy to cluster name
eksctl create iamserviceaccount --name s3-list-sa   --cluster basic-cluster   --attach-policy-arn arn:aws:iam::306093656765:policy/S3ListTestEMLO   --approve --region ap-south-1


KIND=deployment
NAME=my-app-staging
RELEASE=fastapi-release-default
NAMESPACE=default
kubectl annotate $KIND $NAME meta.helm.sh/release-name=$RELEASE
kubectl annotate $KIND $NAME meta.helm.sh/release-namespace=$NAMESPACE
kubectl label $KIND $NAME app.kubernetes.io/managed-by=Helm

go with torchserve itself + kserve

Fix-Helm- issue 
kubectl label namespace default app.kubernetes.io/managed-by=Helm
kubectl annotate namespace default meta.helm.sh/release-name=fastapi-release-default
kubectl annotate namespace default meta.helm.sh/release-namespace=default

helm install fastapi-release-default fastapi-helm --values fastapi-helm/values.yaml --namespace default

<debug-facts>
# Uninstallation
helm uninstall fastapi-release-default

# if uninstallation fails do below
kubectl delete secret sh.helm.release.v1.fastapi-release-default.v1  -n default

todo --
change web-server python files
handle requests between each other between model-server, webserver, ui-server



Ingress for UI-server - to get a url

INSTALL ALB
Thinsg under ALB in https://github.com/ajithvcoder/emlo4-session-16-ajithvcoder

kubectl describe ingress ui-server-ingress

