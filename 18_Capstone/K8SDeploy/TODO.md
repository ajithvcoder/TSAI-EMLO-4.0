2024-04-14

1. Fix Kubernetes app storage urls with vegfruits and sports location, check if we need to use two MODEL_URLS - done
2. Make sure when train.py is triggered, it stores the best model in bin format.
3. Then another script transfer_mar.py takes that from S3 location and transfers as .mar files

Github actions
1. Once pull request is triggered github actions is triggered, Open an EC2 instance
2. Use train.py, transfer_mar.py and get the final models after training in S3
3. Install eksctl, kubectl, argocd , then do deployment via actions
4. Check scaling/load test and note the results
5. Comment it in pull request

```
<debug>

kubectl exec pods/imagenet-classifier-1-predictor-00001-deployment-6f997448fpxqzl   -it --namespace default /bin/bash

kubectl exec --stdin --tty shell-demo -- /bin/bash

kubectl exec mypod -c ruby-container -i -t -- bash -il
</debug>
```

### Pending works

2025-05-02

**Project 1: Train.py and Kubernetes deployment with ArgoCD**

Pending works

1. Demo video
2. Github actions
    - Github actions -> triggering a ec2, training the mode, saving to s3, using k8deploy to scale with argocd and do load testing
    - For argocd get the accesskey, secretkey and create a yaml file for this repo https://github.com/ajithvcoder/emlo4-session-18-ajithvcoder-canary-argocd-kserve
    - The model end point also should be changed in `model-server.cm.yml` file in this repo https://github.com/ajithvcoder/emlo4-session-18-ajithvcoder-canary-argocd-kserve . Use this command `kubectl get pods,svc -n istio-system`
3. Kubernetes
    - K8SDeploy make sport-classifer to work with K8Serve -> Facing issue with K8Server + Torchserve + loading onnx model - Fixed - Done
    - Develop script to take onnx model -> generate .mar file -> move to s3 location - Done
    - Add prometheus, grafana dashboards and show K8SDeploy load testing
        - Automate scale testing with GitHub cicd - do load testing -> GitHub actions - check how many request per second the backend is able to handle, how many total request sent- Show a graph and how much it can handle, do a comment with cml to commit. Scale with HPA and show scaling - [Performance metrics, including latency and stress test results.] 
    - A comprehensive architecture diagram illustrating the entire pipeline and deployment process.
4. Training Part
    - All done


**Project 2: Lambda service -(No need to train just use s3: Optional)**

Pending:
- Complete on - live 
- github actions
- Readme.md - Deployment method

**Project 3: Hugginng face + gradio - (No need to train just use s3: Optional)**

Pending:
- Complete on - live 
- github action
- Readme.md - Deployment method

Overall README
