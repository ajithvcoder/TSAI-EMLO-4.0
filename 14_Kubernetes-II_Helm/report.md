### Report for EMLOV4 - Assignment 14

- Output of the following in a .md file in your repository

- `kubectl describe <your_deployment>`
- `kubectl describe <your_pod>`
- `kubectl describe <your_ingress>`
- `kubectl top pod`
- `kubectl top node`
- `kubectl get all -A -o yaml`

#### Describe Deployment

**Model-server pod**

- `kubectl describe deployment.apps/model-server -n prod`

  ```
  Name:                   model-server
  Namespace:              prod
  CreationTimestamp:      Tue, 31 Dec 2024 10:45:51 +0000
  Labels:                 app.kubernetes.io/managed-by=Helm
                          app.kubernetes.io/name=model-server
                          app.kubernetes.io/part-of=fastapi-app
  Annotations:            deployment.kubernetes.io/revision: 1
                          meta.helm.sh/release-name: fastapi-release-prod
                          meta.helm.sh/release-namespace: production
  Selector:               app.kubernetes.io/name=model-server
  Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
  StrategyType:           RollingUpdate
  MinReadySeconds:        0
  RollingUpdateStrategy:  25% max unavailable, 25% max surge
  Pod Template:
    Labels:  app.kubernetes.io/name=model-server
    Containers:
    model-server:
      Image:      model-server:latest
      Port:       80/TCP
      Host Port:  0/TCP
      Limits:
        cpu:     1
        memory:  2Gi
      Environment:
        REDIS_HOST:      <set to the key 'hostname' of config map 'redis-config-v1'>           Optional: false
        REDIS_PORT:      <set to the key 'port' of config map 'redis-config-v1'>               Optional: false
        REDIS_PASSWORD:  <set to the key 'db_password' in secret 'redis-secret-v1'>            Optional: false
        MODEL_NAME:      <set to the key 'model_name' of config map 'model-server-config-v1'>  Optional: false
      Mounts:            <none>
    Volumes:             <none>
    Node-Selectors:      <none>
    Tolerations:         <none>
  Conditions:
    Type           Status  Reason
    ----           ------  ------
    Available      True    MinimumReplicasAvailable
    Progressing    True    NewReplicaSetAvailable
  OldReplicaSets:  <none>
  NewReplicaSet:   model-server-664885b79f (2/2 replicas created)
  Events:
    Type    Reason             Age   From                   Message
    ----    ------             ----  ----                   -------
    Normal  ScalingReplicaSet  27m   deployment-controller  Scaled up replica set model-server-664885b79f to 2
  ```

**Redis-server pod**

- `kubectl describe deployment.apps/redis  -n prod `

  ```
  Name:                   redis
  Namespace:              prod
  CreationTimestamp:      Tue, 31 Dec 2024 10:45:51 +0000
  Labels:                 app.kubernetes.io/managed-by=Helm
                          app.kubernetes.io/name=redis
                          app.kubernetes.io/part-of=fastapi-app
  Annotations:            deployment.kubernetes.io/revision: 1
                          meta.helm.sh/release-name: fastapi-release-prod
                          meta.helm.sh/release-namespace: production
  Selector:               app.kubernetes.io/name=redis,role=master
  Replicas:               1 desired | 1 updated | 1 total | 1 available | 0 unavailable
  StrategyType:           RollingUpdate
  MinReadySeconds:        0
  RollingUpdateStrategy:  25% max unavailable, 25% max surge
  Pod Template:
    Labels:  app.kubernetes.io/name=redis
            role=master
    Containers:
    redis:
      Image:      redis:7.4.1
      Port:       6379/TCP
      Host Port:  0/TCP
      Command:
        redis-server
      Args:
        --requirepass
        $(REDIS_PASSWORD)
      Limits:
        cpu:     200m
        memory:  200Mi
      Environment:
        REDIS_PASSWORD:  <set to the key 'db_password' in secret 'redis-secret-v1'>  Optional: false
      Mounts:
        /data from redis-storage (rw)
    Volumes:
    redis-storage:
      Type:          PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
      ClaimName:     redis-pvc
      ReadOnly:      false
    Node-Selectors:  <none>
    Tolerations:     <none>
  Conditions:
    Type           Status  Reason
    ----           ------  ------
    Available      True    MinimumReplicasAvailable
    Progressing    True    NewReplicaSetAvailable
  OldReplicaSets:  <none>
  NewReplicaSet:   redis-69cf9b5676 (1/1 replicas created)
  Events:
    Type    Reason             Age   From                   Message
    ----    ------             ----  ----                   -------
    Normal  ScalingReplicaSet  28m   deployment-controller  Scaled up replica set redis-69cf9b5676 to 1
  ```

**UI-server pod - NextJS**

- `kubectl describe deployment.apps/ui-server -n prod`

  ```
  Name:                   ui-server
  Namespace:              prod
  CreationTimestamp:      Tue, 31 Dec 2024 10:45:51 +0000
  Labels:                 app.kubernetes.io/managed-by=Helm
                          app.kubernetes.io/name=ui-server
                          app.kubernetes.io/part-of=fastapi-app
  Annotations:            deployment.kubernetes.io/revision: 1
                          meta.helm.sh/release-name: fastapi-release-prod
                          meta.helm.sh/release-namespace: production
  Selector:               app.kubernetes.io/name=ui-server
  Replicas:               1 desired | 1 updated | 1 total | 1 available | 0 unavailable
  StrategyType:           RollingUpdate
  MinReadySeconds:        0
  RollingUpdateStrategy:  25% max unavailable, 25% max surge
  Pod Template:
    Labels:  app.kubernetes.io/name=ui-server
    Containers:
    ui-server:
      Image:      ui-server:latest
      Port:       80/TCP
      Host Port:  0/TCP
      Limits:
        cpu:     500m
        memory:  1536Mi
      Requests:
        cpu:     250m
        memory:  768Mi
      Environment:
        WEB_SERVER_URL:  <set to the key 'web_server_url' of config map 'web-server-config-v1'>  Optional: false
      Mounts:            <none>
    Volumes:             <none>
    Node-Selectors:      <none>
    Tolerations:         <none>
  Conditions:
    Type           Status  Reason
    ----           ------  ------
    Available      True    MinimumReplicasAvailable
    Progressing    True    NewReplicaSetAvailable
  OldReplicaSets:  <none>
  NewReplicaSet:   ui-server-76dc4d59d8 (1/1 replicas created)
  Events:
    Type    Reason             Age   From                   Message
    ----    ------             ----  ----                   -------
    Normal  ScalingReplicaSet  28m   deployment-controller  Scaled up replica set ui-server-76dc4d59d8 to 1
  ```

**Web-server pod**

- `kubectl describe deployment.apps/web-server -n prod`

  ```
  Name:                   web-server
  Namespace:              prod
  CreationTimestamp:      Tue, 31 Dec 2024 10:45:51 +0000
  Labels:                 app.kubernetes.io/managed-by=Helm
                          app.kubernetes.io/name=web-server
                          app.kubernetes.io/part-of=fastapi-app
  Annotations:            deployment.kubernetes.io/revision: 1
                          meta.helm.sh/release-name: fastapi-release-prod
                          meta.helm.sh/release-namespace: production
  Selector:               app.kubernetes.io/name=web-server
  Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
  StrategyType:           RollingUpdate
  MinReadySeconds:        0
  RollingUpdateStrategy:  25% max unavailable, 25% max surge
  Pod Template:
    Labels:  app.kubernetes.io/name=web-server
    Containers:
    web-server:
      Image:      web-server:latest
      Port:       80/TCP
      Host Port:  0/TCP
      Environment:
        REDIS_HOST:        <set to the key 'hostname' of config map 'redis-config-v1'>                 Optional: false
        REDIS_PORT:        <set to the key 'port' of config map 'redis-config-v1'>                     Optional: false
        REDIS_PASSWORD:    <set to the key 'db_password' in secret 'redis-secret-v1'>                  Optional: false
        MODEL_SERVER_URL:  <set to the key 'model_server_url' of config map 'model-server-config-v1'>  Optional: false
      Mounts:              <none>
    Volumes:               <none>
    Node-Selectors:        <none>
    Tolerations:           <none>
  Conditions:
    Type           Status  Reason
    ----           ------  ------
    Available      True    MinimumReplicasAvailable
    Progressing    True    NewReplicaSetAvailable
  OldReplicaSets:  <none>
  NewReplicaSet:   web-server-5c89b7fd87 (2/2 replicas created)
  Events:
    Type    Reason             Age   From                   Message
    ----    ------             ----  ----                   -------
    Normal  ScalingReplicaSet  29m   deployment-controller  Scaled up replica set web-server-5c89b7fd87 to 2
  ```
#### Describe Pod

**Model-server pod**

- `kubectl describe pod/model-server-664885b79f-zgs5h -n prod` 

  ```
  Name:             model-server-664885b79f-zgs5h
  Namespace:        prod
  Priority:         0
  Service Account:  default
  Node:             minikube/192.168.49.2
  Start Time:       Tue, 31 Dec 2024 10:45:51 +0000
  Labels:           app.kubernetes.io/name=model-server
                    pod-template-hash=664885b79f
  Annotations:      <none>
  Status:           Running
  IP:               10.244.0.43
  IPs:
    IP:           10.244.0.43
  Controlled By:  ReplicaSet/model-server-664885b79f
  Containers:
    model-server:
      Container ID:   docker://82aeb91384660a0d832b5c034b92d4a0b8c12b3014c74281e1d2b26d0ca3037f
      Image:          model-server:latest
      Image ID:       docker://sha256:56299850d16cdc2cb3a3408869773ecbf17ff41380435f09fdcba2564477517e
      Port:           80/TCP
      Host Port:      0/TCP
      State:          Running
        Started:      Tue, 31 Dec 2024 10:45:52 +0000
      Ready:          True
      Restart Count:  0
      Limits:
        cpu:     1
        memory:  2Gi
      Requests:
        cpu:     1
        memory:  2Gi
      Environment:
        REDIS_HOST:      <set to the key 'hostname' of config map 'redis-config-v1'>           Optional: false
        REDIS_PORT:      <set to the key 'port' of config map 'redis-config-v1'>               Optional: false
        REDIS_PASSWORD:  <set to the key 'db_password' in secret 'redis-secret-v1'>            Optional: false
        MODEL_NAME:      <set to the key 'model_name' of config map 'model-server-config-v1'>  Optional: false
      Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-n8cp5 (ro)
  Conditions:
    Type                        Status
    PodReadyToStartContainers   True 
    Initialized                 True 
    Ready                       True 
    ContainersReady             True 
    PodScheduled                True 
  Volumes:
    kube-api-access-n8cp5:
      Type:                    Projected (a volume that contains injected data from multiple sources)
      TokenExpirationSeconds:  3607
      ConfigMapName:           kube-root-ca.crt
      ConfigMapOptional:       <nil>
      DownwardAPI:             true
  QoS Class:                   Guaranteed
  Node-Selectors:              <none>
  Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                              node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
  Events:
    Type    Reason     Age   From               Message
    ----    ------     ----  ----               -------
    Normal  Scheduled  30m   default-scheduler  Successfully assigned prod/model-server-664885b79f-zgs5h to minikube
    Normal  Pulled     30m   kubelet            Container image "model-server:latest" already present on machine
    Normal  Created    30m   kubelet            Created container model-server
    Normal  Started    30m   kubelet            Started container model-server
  ```

**Redis-server pod**

- `kubectl describe pod/redis-69cf9b5676-xnbrl -n prod` 

  ```
  Name:             redis-69cf9b5676-xnbrl
  Namespace:        prod
  Priority:         0
  Service Account:  default
  Node:             minikube/192.168.49.2
  Start Time:       Tue, 31 Dec 2024 10:46:04 +0000
  Labels:           app.kubernetes.io/name=redis
                    pod-template-hash=69cf9b5676
                    role=master
  Annotations:      <none>
  Status:           Running
  IP:               10.244.0.46
  IPs:
    IP:           10.244.0.46
  Controlled By:  ReplicaSet/redis-69cf9b5676
  Containers:
    redis:
      Container ID:  docker://0d95289ab079571e1c1c20f5ceeee90b55fbaff602f1feddc0a616c48e8e7e31
      Image:         redis:7.4.1
      Image ID:      docker-pullable://redis@sha256:bb142a9c18ac18a16713c1491d779697b4e107c22a97266616099d288237ef47
      Port:          6379/TCP
      Host Port:     0/TCP
      Command:
        redis-server
      Args:
        --requirepass
        $(REDIS_PASSWORD)
      State:          Running
        Started:      Tue, 31 Dec 2024 10:46:05 +0000
      Ready:          True
      Restart Count:  0
      Limits:
        cpu:     200m
        memory:  200Mi
      Requests:
        cpu:     200m
        memory:  200Mi
      Environment:
        REDIS_PASSWORD:  <set to the key 'db_password' in secret 'redis-secret-v1'>  Optional: false
      Mounts:
        /data from redis-storage (rw)
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-k2z8z (ro)
  Conditions:
    Type                        Status
    PodReadyToStartContainers   True 
    Initialized                 True 
    Ready                       True 
    ContainersReady             True 
    PodScheduled                True 
  Volumes:
    redis-storage:
      Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
      ClaimName:  redis-pvc
      ReadOnly:   false
    kube-api-access-k2z8z:
      Type:                    Projected (a volume that contains injected data from multiple sources)
      TokenExpirationSeconds:  3607
      ConfigMapName:           kube-root-ca.crt
      ConfigMapOptional:       <nil>
      DownwardAPI:             true
  QoS Class:                   Guaranteed
  Node-Selectors:              <none>
  Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                              node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
  Events:
    Type     Reason            Age                From               Message
    ----     ------            ----               ----               -------
    Warning  FailedScheduling  30m (x2 over 30m)  default-scheduler  0/1 nodes are available: pod has unbound immediate PersistentVolumeClaims. preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling.
    Normal   Scheduled         30m                default-scheduler  Successfully assigned prod/redis-69cf9b5676-xnbrl to minikube
    Normal   Pulled            30m                kubelet            Container image "redis:7.4.1" already present on machine
    Normal   Created           30m                kubelet            Created container redis
    Normal   Started           30m                kubelet            Started container redis
  ```

**UI-server pod**

- `kubectl describe pod/ui-server-76dc4d59d8-4gdc9 -n prod`

  ```
  Name:             ui-server-76dc4d59d8-4gdc9
  Namespace:        prod
  Priority:         0
  Service Account:  default
  Node:             minikube/192.168.49.2
  Start Time:       Tue, 31 Dec 2024 10:45:51 +0000
  Labels:           app.kubernetes.io/name=ui-server
                    pod-template-hash=76dc4d59d8
  Annotations:      <none>
  Status:           Running
  IP:               10.244.0.42
  IPs:
    IP:           10.244.0.42
  Controlled By:  ReplicaSet/ui-server-76dc4d59d8
  Containers:
    ui-server:
      Container ID:   docker://1be9141abfa1df41795b684358b110764dbe4be11d8bcf8bb53e080adba79f23
      Image:          ui-server:latest
      Image ID:       docker://sha256:49607624ad85e97b9d4ae77f9c9cdb0bab33eb252ec1ef969f6375b584a823cd
      Port:           80/TCP
      Host Port:      0/TCP
      State:          Running
        Started:      Tue, 31 Dec 2024 10:45:52 +0000
      Ready:          True
      Restart Count:  0
      Limits:
        cpu:     500m
        memory:  1536Mi
      Requests:
        cpu:     250m
        memory:  768Mi
      Environment:
        WEB_SERVER_URL:  <set to the key 'web_server_url' of config map 'web-server-config-v1'>  Optional: false
      Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-bvzp8 (ro)
  Conditions:
    Type                        Status
    PodReadyToStartContainers   True 
    Initialized                 True 
    Ready                       True 
    ContainersReady             True 
    PodScheduled                True 
  Volumes:
    kube-api-access-bvzp8:
      Type:                    Projected (a volume that contains injected data from multiple sources)
      TokenExpirationSeconds:  3607
      ConfigMapName:           kube-root-ca.crt
      ConfigMapOptional:       <nil>
      DownwardAPI:             true
  QoS Class:                   Burstable
  Node-Selectors:              <none>
  Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                              node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
  Events:
    Type    Reason     Age   From               Message
    ----    ------     ----  ----               -------
    Normal  Scheduled  31m   default-scheduler  Successfully assigned prod/ui-server-76dc4d59d8-4gdc9 to minikube
    Normal  Pulled     31m   kubelet            Container image "ui-server:latest" already present on machine
    Normal  Created    31m   kubelet            Created container ui-server
    Normal  Started    31m   kubelet            Started container ui-server
  ```

**Web-server pod**

- `kubectl describe pod/web-server-5c89b7fd87-t6vls -n prod` 

  ```
  Name:             web-server-5c89b7fd87-t6vls
  Namespace:        prod
  Priority:         0
  Service Account:  default
  Node:             minikube/192.168.49.2
  Start Time:       Tue, 31 Dec 2024 10:45:51 +0000
  Labels:           app.kubernetes.io/name=web-server
                    pod-template-hash=5c89b7fd87
  Annotations:      <none>
  Status:           Running
  IP:               10.244.0.41
  IPs:
    IP:           10.244.0.41
  Controlled By:  ReplicaSet/web-server-5c89b7fd87
  Containers:
    web-server:
      Container ID:   docker://3c181efaf7e8e153bfb0721e055722aeaf932fa6460ab1653b41963bd92cb737
      Image:          web-server:latest
      Image ID:       docker://sha256:5102c64cf8a11b48b82716fbd8591f7f611344f8d30235d8ba192aadf13aae62
      Port:           80/TCP
      Host Port:      0/TCP
      State:          Running
        Started:      Tue, 31 Dec 2024 10:45:52 +0000
      Ready:          True
      Restart Count:  0
      Environment:
        REDIS_HOST:        <set to the key 'hostname' of config map 'redis-config-v1'>                 Optional: false
        REDIS_PORT:        <set to the key 'port' of config map 'redis-config-v1'>                     Optional: false
        REDIS_PASSWORD:    <set to the key 'db_password' in secret 'redis-secret-v1'>                  Optional: false
        MODEL_SERVER_URL:  <set to the key 'model_server_url' of config map 'model-server-config-v1'>  Optional: false
      Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-zjl78 (ro)
  Conditions:
    Type                        Status
    PodReadyToStartContainers   True 
    Initialized                 True 
    Ready                       True 
    ContainersReady             True 
    PodScheduled                True 
  Volumes:
    kube-api-access-zjl78:
      Type:                    Projected (a volume that contains injected data from multiple sources)
      TokenExpirationSeconds:  3607
      ConfigMapName:           kube-root-ca.crt
      ConfigMapOptional:       <nil>
      DownwardAPI:             true
  QoS Class:                   BestEffort
  Node-Selectors:              <none>
  Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                              node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
  Events:
    Type    Reason     Age   From               Message
    ----    ------     ----  ----               -------
    Normal  Scheduled  31m   default-scheduler  Successfully assigned prod/web-server-5c89b7fd87-t6vls to minikube
    Normal  Pulled     31m   kubelet            Container image "web-server:latest" already present on machine
    Normal  Created    31m   kubelet            Created container web-server
    Normal  Started    31m   kubelet            Started container web-server
  ```

#### Describe Ingress

`kubectl describe ingress -n prod`

  ```
  Name:             ui-server-ingress
  Labels:           app.kubernetes.io/managed-by=Helm
                    app.kubernetes.io/name=ui-server
                    app.kubernetes.io/part-of=fastapi-app
  Namespace:        prod
  Address:          192.168.49.2
  Ingress Class:    nginx
  Default backend:  <default>
  Rules:
    Host          Path  Backends
    ----          ----  --------
    fastapi.prod  
                  /   ui-server-service:80 (10.244.0.42:80)
  Annotations:    meta.helm.sh/release-name: fastapi-release-prod
                  meta.helm.sh/release-namespace: production
  Events:
    Type    Reason  Age                From                      Message
    ----    ------  ----               ----                      -------
    Normal  Sync    31m (x2 over 32m)  nginx-ingress-controller  Scheduled for sync
  ```

#### Top Pod

`kubectl top pod -n prod`

  ```
  NAME                            CPU(cores)   MEMORY(bytes)   
  model-server-664885b79f-9b4zn   2m           86Mi            
  model-server-664885b79f-zgs5h   2m           115Mi           
  redis-69cf9b5676-xnbrl          4m           15Mi            
  ui-server-76dc4d59d8-4gdc9      1m           522Mi           
  web-server-5c89b7fd87-t6vls     2m           52Mi            
  web-server-5c89b7fd87-xhs8r     2m           43Mi            
  ```

#### Top Node

- `kubectl top node -n prod` 

  ```
  NAME       CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%   
  minikube   164m         0%     2083Mi          26%       
  ```

#### Get all info

- `kubectl get all -A -o yaml`

  ```
  apiVersion: v1
  items:
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:31:31Z"
      generateName: ingress-nginx-admission-create-
      labels:
        app.kubernetes.io/component: admission-webhook
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
        batch.kubernetes.io/controller-uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
        batch.kubernetes.io/job-name: ingress-nginx-admission-create
        controller-uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
        job-name: ingress-nginx-admission-create
      name: ingress-nginx-admission-create-dn7h9
      namespace: ingress-nginx
      ownerReferences:
      - apiVersion: batch/v1
        blockOwnerDeletion: true
        controller: true
        kind: Job
        name: ingress-nginx-admission-create
        uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
      resourceVersion: "1403"
      uid: a720aa91-4a7e-4137-a106-55c5fa19aa15
    spec:
      containers:
      - args:
        - create
        - --host=ingress-nginx-controller-admission,ingress-nginx-controller-admission.$(POD_NAMESPACE).svc
        - --namespace=$(POD_NAMESPACE)
        - --secret-name=ingress-nginx-admission
        env:
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.namespace
        image: registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        imagePullPolicy: IfNotPresent
        name: create
        resources: {}
        securityContext:
          allowPrivilegeEscalation: false
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-qmtvb
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
        minikube.k8s.io/primary: "true"
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: OnFailure
      schedulerName: default-scheduler
      securityContext:
        runAsNonRoot: true
        runAsUser: 2000
      serviceAccount: ingress-nginx-admission
      serviceAccountName: ingress-nginx-admission
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-qmtvb
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:45Z"
        status: "False"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "False"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "False"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://30ec83ce230cce9cd48b74acfde3279f972b5e293840e9105051e9dcc4e6bfa1
        image: registry.k8s.io/ingress-nginx/kube-webhook-certgen@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        imageID: docker-pullable://registry.k8s.io/ingress-nginx/kube-webhook-certgen@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        lastState: {}
        name: create
        ready: false
        restartCount: 0
        started: false
        state:
          terminated:
            containerID: docker://30ec83ce230cce9cd48b74acfde3279f972b5e293840e9105051e9dcc4e6bfa1
            exitCode: 0
            finishedAt: "2024-12-31T08:31:43Z"
            reason: Completed
            startedAt: "2024-12-31T08:31:43Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-qmtvb
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Succeeded
      podIP: 10.244.0.7
      podIPs:
      - ip: 10.244.0.7
      qosClass: BestEffort
      startTime: "2024-12-31T08:31:31Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:31:31Z"
      generateName: ingress-nginx-admission-patch-
      labels:
        app.kubernetes.io/component: admission-webhook
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
        batch.kubernetes.io/controller-uid: 572547be-9593-4e67-83c8-b9c09600be77
        batch.kubernetes.io/job-name: ingress-nginx-admission-patch
        controller-uid: 572547be-9593-4e67-83c8-b9c09600be77
        job-name: ingress-nginx-admission-patch
      name: ingress-nginx-admission-patch-zctnt
      namespace: ingress-nginx
      ownerReferences:
      - apiVersion: batch/v1
        blockOwnerDeletion: true
        controller: true
        kind: Job
        name: ingress-nginx-admission-patch
        uid: 572547be-9593-4e67-83c8-b9c09600be77
      resourceVersion: "1399"
      uid: 6071e5c1-6ba2-4453-b5ce-77afe18f2ff2
    spec:
      containers:
      - args:
        - patch
        - --webhook-name=ingress-nginx-admission
        - --namespace=$(POD_NAMESPACE)
        - --patch-mutating=false
        - --secret-name=ingress-nginx-admission
        - --patch-failure-policy=Fail
        env:
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.namespace
        image: registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        imagePullPolicy: IfNotPresent
        name: patch
        resources: {}
        securityContext:
          allowPrivilegeEscalation: false
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-vt2fs
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
        minikube.k8s.io/primary: "true"
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: OnFailure
      schedulerName: default-scheduler
      securityContext:
        runAsNonRoot: true
        runAsUser: 2000
      serviceAccount: ingress-nginx-admission
      serviceAccountName: ingress-nginx-admission
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-vt2fs
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:45Z"
        status: "False"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "False"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        reason: PodCompleted
        status: "False"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://67b0f5737904b985fd872cb2c4985faadfbfd1d4c760b4b4d371a0d615e63c86
        image: registry.k8s.io/ingress-nginx/kube-webhook-certgen@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        imageID: docker-pullable://registry.k8s.io/ingress-nginx/kube-webhook-certgen@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
        lastState: {}
        name: patch
        ready: false
        restartCount: 0
        started: false
        state:
          terminated:
            containerID: docker://67b0f5737904b985fd872cb2c4985faadfbfd1d4c760b4b4d371a0d615e63c86
            exitCode: 0
            finishedAt: "2024-12-31T08:31:43Z"
            reason: Completed
            startedAt: "2024-12-31T08:31:43Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-vt2fs
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Succeeded
      podIP: 10.244.0.8
      podIPs:
      - ip: 10.244.0.8
      qosClass: BestEffort
      startTime: "2024-12-31T08:31:31Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:31:31Z"
      generateName: ingress-nginx-controller-bc57996ff-
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
        gcp-auth-skip-secret: "true"
        pod-template-hash: bc57996ff
      name: ingress-nginx-controller-bc57996ff-q8flx
      namespace: ingress-nginx
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: ingress-nginx-controller-bc57996ff
        uid: f8ffdf94-1ef8-4048-b820-4858c0c6d57d
      resourceVersion: "1554"
      uid: b675e2b8-4050-4d71-9502-97bcefed7fa3
    spec:
      containers:
      - args:
        - /nginx-ingress-controller
        - --election-id=ingress-nginx-leader
        - --controller-class=k8s.io/ingress-nginx
        - --watch-ingress-without-class=true
        - --configmap=$(POD_NAMESPACE)/ingress-nginx-controller
        - --tcp-services-configmap=$(POD_NAMESPACE)/tcp-services
        - --udp-services-configmap=$(POD_NAMESPACE)/udp-services
        - --validating-webhook=:8443
        - --validating-webhook-certificate=/usr/local/certificates/cert
        - --validating-webhook-key=/usr/local/certificates/key
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.namespace
        - name: LD_PRELOAD
          value: /usr/local/lib/libmimalloc.so
        image: registry.k8s.io/ingress-nginx/controller:v1.11.2@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce
        imagePullPolicy: IfNotPresent
        lifecycle:
          preStop:
            exec:
              command:
              - /wait-shutdown
        livenessProbe:
          failureThreshold: 5
          httpGet:
            path: /healthz
            port: 10254
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        name: controller
        ports:
        - containerPort: 80
          hostPort: 80
          name: http
          protocol: TCP
        - containerPort: 443
          hostPort: 443
          name: https
          protocol: TCP
        - containerPort: 8443
          name: webhook
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /healthz
            port: 10254
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        resources:
          requests:
            cpu: 100m
            memory: 90Mi
        securityContext:
          allowPrivilegeEscalation: true
          capabilities:
            add:
            - NET_BIND_SERVICE
            drop:
            - ALL
          runAsUser: 101
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /usr/local/certificates/
          name: webhook-cert
          readOnly: true
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-wrq9l
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
        minikube.k8s.io/primary: "true"
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: ingress-nginx
      serviceAccountName: ingress-nginx
      terminationGracePeriodSeconds: 0
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
        operator: Equal
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: webhook-cert
        secret:
          defaultMode: 420
          secretName: ingress-nginx-admission
      - name: kube-api-access-wrq9l
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:28Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:42Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:42Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:31:31Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://b2833ab7ba23b9fec64b8efce4a5e84e4bc07657d1969d533b9733484ca5bdf4
        image: registry.k8s.io/ingress-nginx/controller@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce
        imageID: docker-pullable://registry.k8s.io/ingress-nginx/controller@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce
        lastState: {}
        name: controller
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:32:28Z"
        volumeMounts:
        - mountPath: /usr/local/certificates/
          name: webhook-cert
          readOnly: true
          recursiveReadOnly: Disabled
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-wrq9l
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.9
      podIPs:
      - ip: 10.244.0.9
      qosClass: Burstable
      startTime: "2024-12-31T08:31:31Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:13:28Z"
      generateName: coredns-6f6b679f8f-
      labels:
        k8s-app: kube-dns
        pod-template-hash: 6f6b679f8f
      name: coredns-6f6b679f8f-zcjzp
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: coredns-6f6b679f8f
        uid: a04cc8c6-fdc8-43b8-b167-be7dbbab4644
      resourceVersion: "411"
      uid: 0b678424-687d-48b8-a771-297724432d38
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: k8s-app
                  operator: In
                  values:
                  - kube-dns
              topologyKey: kubernetes.io/hostname
            weight: 100
      containers:
      - args:
        - -conf
        - /etc/coredns/Corefile
        image: registry.k8s.io/coredns/coredns:v1.11.1
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 5
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 5
        name: coredns
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        - containerPort: 9153
          name: metrics
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /ready
            port: 8181
            scheme: HTTP
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        resources:
          limits:
            memory: 170Mi
          requests:
            cpu: 100m
            memory: 70Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            add:
            - NET_BIND_SERVICE
            drop:
            - ALL
          readOnlyRootFilesystem: true
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /etc/coredns
          name: config-volume
          readOnly: true
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-b9blg
          readOnly: true
      dnsPolicy: Default
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
      preemptionPolicy: PreemptLowerPriority
      priority: 2000000000
      priorityClassName: system-cluster-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: coredns
      serviceAccountName: coredns
      terminationGracePeriodSeconds: 30
      tolerations:
      - key: CriticalAddonsOnly
        operator: Exists
      - effect: NoSchedule
        key: node-role.kubernetes.io/control-plane
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - configMap:
          defaultMode: 420
          items:
          - key: Corefile
            path: Corefile
          name: coredns
        name: config-volume
      - name: kube-api-access-b9blg
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:28Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:14:09Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:14:09Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:28Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://630aecaf3affa88fe72a01cc1d3352e1145cba75cea5bf4e245539915cdc3e36
        image: registry.k8s.io/coredns/coredns:v1.11.1
        imageID: docker-pullable://registry.k8s.io/coredns/coredns@sha256:1eeb4c7316bacb1d4c8ead65571cd92dd21e27359f0d4917f1a5822a73b75db1
        lastState: {}
        name: coredns
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:29Z"
        volumeMounts:
        - mountPath: /etc/coredns
          name: config-volume
          readOnly: true
          recursiveReadOnly: Disabled
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-b9blg
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.2
      podIPs:
      - ip: 10.244.0.2
      qosClass: Burstable
      startTime: "2024-12-31T08:13:28Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        kubeadm.kubernetes.io/etcd.advertise-client-urls: https://192.168.49.2:2379
        kubernetes.io/config.hash: a5363f4f31e043bdae3c93aca4991903
        kubernetes.io/config.mirror: a5363f4f31e043bdae3c93aca4991903
        kubernetes.io/config.seen: "2024-12-31T08:13:23.218580566Z"
        kubernetes.io/config.source: file
      creationTimestamp: "2024-12-31T08:13:23Z"
      labels:
        component: etcd
        tier: control-plane
      name: etcd-minikube
      namespace: kube-system
      ownerReferences:
      - apiVersion: v1
        controller: true
        kind: Node
        name: minikube
        uid: 28e96d07-ad32-446b-b87d-1e83a64f562b
      resourceVersion: "375"
      uid: 73dfe804-4fcc-490f-a3ee-503ed5cf72f4
    spec:
      containers:
      - command:
        - etcd
        - --advertise-client-urls=https://192.168.49.2:2379
        - --cert-file=/var/lib/minikube/certs/etcd/server.crt
        - --client-cert-auth=true
        - --data-dir=/var/lib/minikube/etcd
        - --experimental-initial-corrupt-check=true
        - --experimental-watch-progress-notify-interval=5s
        - --initial-advertise-peer-urls=https://192.168.49.2:2380
        - --initial-cluster=minikube=https://192.168.49.2:2380
        - --key-file=/var/lib/minikube/certs/etcd/server.key
        - --listen-client-urls=https://127.0.0.1:2379,https://192.168.49.2:2379
        - --listen-metrics-urls=http://127.0.0.1:2381
        - --listen-peer-urls=https://192.168.49.2:2380
        - --name=minikube
        - --peer-cert-file=/var/lib/minikube/certs/etcd/peer.crt
        - --peer-client-cert-auth=true
        - --peer-key-file=/var/lib/minikube/certs/etcd/peer.key
        - --peer-trusted-ca-file=/var/lib/minikube/certs/etcd/ca.crt
        - --proxy-refresh-interval=70000
        - --snapshot-count=10000
        - --trusted-ca-file=/var/lib/minikube/certs/etcd/ca.crt
        image: registry.k8s.io/etcd:3.5.15-0
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /livez
            port: 2381
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        name: etcd
        readinessProbe:
          failureThreshold: 3
          httpGet:
            host: 127.0.0.1
            path: /readyz
            port: 2381
            scheme: HTTP
          periodSeconds: 1
          successThreshold: 1
          timeoutSeconds: 15
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        startupProbe:
          failureThreshold: 24
          httpGet:
            host: 127.0.0.1
            path: /readyz
            port: 2381
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/lib/minikube/etcd
          name: etcd-data
        - mountPath: /var/lib/minikube/certs/etcd
          name: etcd-certs
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 2000001000
      priorityClassName: system-node-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext:
        seccompProfile:
          type: RuntimeDefault
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        operator: Exists
      volumes:
      - hostPath:
          path: /var/lib/minikube/certs/etcd
          type: DirectoryOrCreate
        name: etcd-certs
      - hostPath:
          path: /var/lib/minikube/etcd
          type: DirectoryOrCreate
        name: etcd-data
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:32Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:32Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://a1a9e05fecf01908446ef5e1b962f993dc81facea7a65188483d4c85d0dac8f8
        image: registry.k8s.io/etcd:3.5.15-0
        imageID: docker-pullable://registry.k8s.io/etcd@sha256:a6dc63e6e8cfa0307d7851762fa6b629afb18f28d8aa3fab5a6e91b4af60026a
        lastState: {}
        name: etcd
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:17Z"
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: Burstable
      startTime: "2024-12-31T08:13:23Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint: 192.168.49.2:8443
        kubernetes.io/config.hash: 9e315b3a91fa9f6f7463439d9dac1a56
        kubernetes.io/config.mirror: 9e315b3a91fa9f6f7463439d9dac1a56
        kubernetes.io/config.seen: "2024-12-31T08:13:23.218585913Z"
        kubernetes.io/config.source: file
      creationTimestamp: "2024-12-31T08:13:23Z"
      labels:
        component: kube-apiserver
        tier: control-plane
      name: kube-apiserver-minikube
      namespace: kube-system
      ownerReferences:
      - apiVersion: v1
        controller: true
        kind: Node
        name: minikube
        uid: 28e96d07-ad32-446b-b87d-1e83a64f562b
      resourceVersion: "376"
      uid: 7d18eb08-8005-40d7-bf1c-a16f657529b6
    spec:
      containers:
      - command:
        - kube-apiserver
        - --advertise-address=192.168.49.2
        - --allow-privileged=true
        - --authorization-mode=Node,RBAC
        - --client-ca-file=/var/lib/minikube/certs/ca.crt
        - --enable-admission-plugins=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota
        - --enable-bootstrap-token-auth=true
        - --etcd-cafile=/var/lib/minikube/certs/etcd/ca.crt
        - --etcd-certfile=/var/lib/minikube/certs/apiserver-etcd-client.crt
        - --etcd-keyfile=/var/lib/minikube/certs/apiserver-etcd-client.key
        - --etcd-servers=https://127.0.0.1:2379
        - --kubelet-client-certificate=/var/lib/minikube/certs/apiserver-kubelet-client.crt
        - --kubelet-client-key=/var/lib/minikube/certs/apiserver-kubelet-client.key
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --proxy-client-cert-file=/var/lib/minikube/certs/front-proxy-client.crt
        - --proxy-client-key-file=/var/lib/minikube/certs/front-proxy-client.key
        - --requestheader-allowed-names=front-proxy-client
        - --requestheader-client-ca-file=/var/lib/minikube/certs/front-proxy-ca.crt
        - --requestheader-extra-headers-prefix=X-Remote-Extra-
        - --requestheader-group-headers=X-Remote-Group
        - --requestheader-username-headers=X-Remote-User
        - --secure-port=8443
        - --service-account-issuer=https://kubernetes.default.svc.cluster.local
        - --service-account-key-file=/var/lib/minikube/certs/sa.pub
        - --service-account-signing-key-file=/var/lib/minikube/certs/sa.key
        - --service-cluster-ip-range=10.96.0.0/12
        - --tls-cert-file=/var/lib/minikube/certs/apiserver.crt
        - --tls-private-key-file=/var/lib/minikube/certs/apiserver.key
        image: registry.k8s.io/kube-apiserver:v1.31.0
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 192.168.49.2
            path: /livez
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        name: kube-apiserver
        readinessProbe:
          failureThreshold: 3
          httpGet:
            host: 192.168.49.2
            path: /readyz
            port: 8443
            scheme: HTTPS
          periodSeconds: 1
          successThreshold: 1
          timeoutSeconds: 15
        resources:
          requests:
            cpu: 250m
        startupProbe:
          failureThreshold: 24
          httpGet:
            host: 192.168.49.2
            path: /livez
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /etc/ssl/certs
          name: ca-certs
          readOnly: true
        - mountPath: /etc/ca-certificates
          name: etc-ca-certificates
          readOnly: true
        - mountPath: /var/lib/minikube/certs
          name: k8s-certs
          readOnly: true
        - mountPath: /usr/local/share/ca-certificates
          name: usr-local-share-ca-certificates
          readOnly: true
        - mountPath: /usr/share/ca-certificates
          name: usr-share-ca-certificates
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 2000001000
      priorityClassName: system-node-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext:
        seccompProfile:
          type: RuntimeDefault
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        operator: Exists
      volumes:
      - hostPath:
          path: /etc/ssl/certs
          type: DirectoryOrCreate
        name: ca-certs
      - hostPath:
          path: /etc/ca-certificates
          type: DirectoryOrCreate
        name: etc-ca-certificates
      - hostPath:
          path: /var/lib/minikube/certs
          type: DirectoryOrCreate
        name: k8s-certs
      - hostPath:
          path: /usr/local/share/ca-certificates
          type: DirectoryOrCreate
        name: usr-local-share-ca-certificates
      - hostPath:
          path: /usr/share/ca-certificates
          type: DirectoryOrCreate
        name: usr-share-ca-certificates
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:32Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:32Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://02d47181cadbf21b22f337bcb7ea13f50d9317fe4d9c2428b7f970c4f62757b6
        image: registry.k8s.io/kube-apiserver:v1.31.0
        imageID: docker-pullable://registry.k8s.io/kube-apiserver@sha256:470179274deb9dc3a81df55cfc24823ce153147d4ebf2ed649a4f271f51eaddf
        lastState: {}
        name: kube-apiserver
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:17Z"
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: Burstable
      startTime: "2024-12-31T08:13:23Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        kubernetes.io/config.hash: 40f5f661ab65f2e4bfe41ac2993c01de
        kubernetes.io/config.mirror: 40f5f661ab65f2e4bfe41ac2993c01de
        kubernetes.io/config.seen: "2024-12-31T08:13:23.218587181Z"
        kubernetes.io/config.source: file
      creationTimestamp: "2024-12-31T08:13:23Z"
      labels:
        component: kube-controller-manager
        tier: control-plane
      name: kube-controller-manager-minikube
      namespace: kube-system
      ownerReferences:
      - apiVersion: v1
        controller: true
        kind: Node
        name: minikube
        uid: 28e96d07-ad32-446b-b87d-1e83a64f562b
      resourceVersion: "369"
      uid: a881921c-21b1-416d-81c7-be1e5e2d209b
    spec:
      containers:
      - command:
        - kube-controller-manager
        - --allocate-node-cidrs=true
        - --authentication-kubeconfig=/etc/kubernetes/controller-manager.conf
        - --authorization-kubeconfig=/etc/kubernetes/controller-manager.conf
        - --bind-address=127.0.0.1
        - --client-ca-file=/var/lib/minikube/certs/ca.crt
        - --cluster-cidr=10.244.0.0/16
        - --cluster-name=mk
        - --cluster-signing-cert-file=/var/lib/minikube/certs/ca.crt
        - --cluster-signing-key-file=/var/lib/minikube/certs/ca.key
        - --controllers=*,bootstrapsigner,tokencleaner
        - --kubeconfig=/etc/kubernetes/controller-manager.conf
        - --leader-elect=false
        - --requestheader-client-ca-file=/var/lib/minikube/certs/front-proxy-ca.crt
        - --root-ca-file=/var/lib/minikube/certs/ca.crt
        - --service-account-private-key-file=/var/lib/minikube/certs/sa.key
        - --service-cluster-ip-range=10.96.0.0/12
        - --use-service-account-credentials=true
        image: registry.k8s.io/kube-controller-manager:v1.31.0
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10257
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        name: kube-controller-manager
        resources:
          requests:
            cpu: 200m
        startupProbe:
          failureThreshold: 24
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10257
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /etc/ssl/certs
          name: ca-certs
          readOnly: true
        - mountPath: /etc/ca-certificates
          name: etc-ca-certificates
          readOnly: true
        - mountPath: /usr/libexec/kubernetes/kubelet-plugins/volume/exec
          name: flexvolume-dir
        - mountPath: /var/lib/minikube/certs
          name: k8s-certs
          readOnly: true
        - mountPath: /etc/kubernetes/controller-manager.conf
          name: kubeconfig
          readOnly: true
        - mountPath: /usr/local/share/ca-certificates
          name: usr-local-share-ca-certificates
          readOnly: true
        - mountPath: /usr/share/ca-certificates
          name: usr-share-ca-certificates
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 2000001000
      priorityClassName: system-node-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext:
        seccompProfile:
          type: RuntimeDefault
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        operator: Exists
      volumes:
      - hostPath:
          path: /etc/ssl/certs
          type: DirectoryOrCreate
        name: ca-certs
      - hostPath:
          path: /etc/ca-certificates
          type: DirectoryOrCreate
        name: etc-ca-certificates
      - hostPath:
          path: /usr/libexec/kubernetes/kubelet-plugins/volume/exec
          type: DirectoryOrCreate
        name: flexvolume-dir
      - hostPath:
          path: /var/lib/minikube/certs
          type: DirectoryOrCreate
        name: k8s-certs
      - hostPath:
          path: /etc/kubernetes/controller-manager.conf
          type: FileOrCreate
        name: kubeconfig
      - hostPath:
          path: /usr/local/share/ca-certificates
          type: DirectoryOrCreate
        name: usr-local-share-ca-certificates
      - hostPath:
          path: /usr/share/ca-certificates
          type: DirectoryOrCreate
        name: usr-share-ca-certificates
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://f27e2c0d1db54ae72b7020d398894eec7be8273b7da6fc2d8ecd7ee6f0b6f580
        image: registry.k8s.io/kube-controller-manager:v1.31.0
        imageID: docker-pullable://registry.k8s.io/kube-controller-manager@sha256:f6f3c33dda209e8434b83dacf5244c03b59b0018d93325ff21296a142b68497d
        lastState: {}
        name: kube-controller-manager
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:17Z"
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: Burstable
      startTime: "2024-12-31T08:13:23Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:13:28Z"
      generateName: kube-proxy-
      labels:
        controller-revision-hash: 5976bc5f75
        k8s-app: kube-proxy
        pod-template-generation: "1"
      name: kube-proxy-rtc5p
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: DaemonSet
        name: kube-proxy
        uid: 3a06cdae-3f0c-4247-a662-370d4a742ead
      resourceVersion: "357"
      uid: 947b6b9a-3455-4e01-a879-90cfb1f48cd5
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchFields:
              - key: metadata.name
                operator: In
                values:
                - minikube
      containers:
      - command:
        - /usr/local/bin/kube-proxy
        - --config=/var/lib/kube-proxy/config.conf
        - --hostname-override=$(NODE_NAME)
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: spec.nodeName
        image: registry.k8s.io/kube-proxy:v1.31.0
        imagePullPolicy: IfNotPresent
        name: kube-proxy
        resources: {}
        securityContext:
          privileged: true
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/lib/kube-proxy
          name: kube-proxy
        - mountPath: /run/xtables.lock
          name: xtables-lock
        - mountPath: /lib/modules
          name: lib-modules
          readOnly: true
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-pksxk
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
      preemptionPolicy: PreemptLowerPriority
      priority: 2000001000
      priorityClassName: system-node-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: kube-proxy
      serviceAccountName: kube-proxy
      terminationGracePeriodSeconds: 30
      tolerations:
      - operator: Exists
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
      - effect: NoSchedule
        key: node.kubernetes.io/disk-pressure
        operator: Exists
      - effect: NoSchedule
        key: node.kubernetes.io/memory-pressure
        operator: Exists
      - effect: NoSchedule
        key: node.kubernetes.io/pid-pressure
        operator: Exists
      - effect: NoSchedule
        key: node.kubernetes.io/unschedulable
        operator: Exists
      - effect: NoSchedule
        key: node.kubernetes.io/network-unavailable
        operator: Exists
      volumes:
      - configMap:
          defaultMode: 420
          name: kube-proxy
        name: kube-proxy
      - hostPath:
          path: /run/xtables.lock
          type: FileOrCreate
        name: xtables-lock
      - hostPath:
          path: /lib/modules
          type: ""
        name: lib-modules
      - name: kube-api-access-pksxk
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:29Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:28Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:29Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:29Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:28Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://5cab44322eeb9ed1381148b6654d5f525260655d86bf311646d463892e242c70
        image: registry.k8s.io/kube-proxy:v1.31.0
        imageID: docker-pullable://registry.k8s.io/kube-proxy@sha256:c727efb1c6f15a68060bf7f207f5c7a765355b7e3340c513e582ec819c5cd2fe
        lastState: {}
        name: kube-proxy
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:28Z"
        volumeMounts:
        - mountPath: /var/lib/kube-proxy
          name: kube-proxy
        - mountPath: /run/xtables.lock
          name: xtables-lock
        - mountPath: /lib/modules
          name: lib-modules
          readOnly: true
          recursiveReadOnly: Disabled
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-pksxk
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: BestEffort
      startTime: "2024-12-31T08:13:28Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        kubernetes.io/config.hash: e039200acb850c82bb901653cc38ff6e
        kubernetes.io/config.mirror: e039200acb850c82bb901653cc38ff6e
        kubernetes.io/config.seen: "2024-12-31T08:13:23.218588080Z"
        kubernetes.io/config.source: file
      creationTimestamp: "2024-12-31T08:13:23Z"
      labels:
        component: kube-scheduler
        tier: control-plane
      name: kube-scheduler-minikube
      namespace: kube-system
      ownerReferences:
      - apiVersion: v1
        controller: true
        kind: Node
        name: minikube
        uid: 28e96d07-ad32-446b-b87d-1e83a64f562b
      resourceVersion: "381"
      uid: 7745d240-8758-47b9-941b-95458b2d782b
    spec:
      containers:
      - command:
        - kube-scheduler
        - --authentication-kubeconfig=/etc/kubernetes/scheduler.conf
        - --authorization-kubeconfig=/etc/kubernetes/scheduler.conf
        - --bind-address=127.0.0.1
        - --kubeconfig=/etc/kubernetes/scheduler.conf
        - --leader-elect=false
        image: registry.k8s.io/kube-scheduler:v1.31.0
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10259
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        name: kube-scheduler
        resources:
          requests:
            cpu: 100m
        startupProbe:
          failureThreshold: 24
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10259
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 15
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /etc/kubernetes/scheduler.conf
          name: kubeconfig
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 2000001000
      priorityClassName: system-node-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext:
        seccompProfile:
          type: RuntimeDefault
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        operator: Exists
      volumes:
      - hostPath:
          path: /etc/kubernetes/scheduler.conf
          type: FileOrCreate
        name: kubeconfig
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:37Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:37Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:23Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://9857d83f8b3f5c24a3c7eae47a5cd9661951e26a5b824b772456b61a45e69cd3
        image: registry.k8s.io/kube-scheduler:v1.31.0
        imageID: docker-pullable://registry.k8s.io/kube-scheduler@sha256:96ddae9c9b2e79342e0551e2d2ec422c0c02629a74d928924aaa069706619808
        lastState: {}
        name: kube-scheduler
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:17Z"
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: Burstable
      startTime: "2024-12-31T08:13:23Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:34:29Z"
      generateName: metrics-server-54bf7cdd6-
      labels:
        k8s-app: metrics-server
        pod-template-hash: 54bf7cdd6
      name: metrics-server-54bf7cdd6-pb6lr
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: metrics-server-54bf7cdd6
        uid: 436bbc45-d3c8-4dcc-aa95-d4f0f5903c50
      resourceVersion: "1719"
      uid: 63c07016-6c3e-4423-a7ed-467d8fb4feb2
    spec:
      containers:
      - args:
        - --cert-dir=/tmp
        - --secure-port=10250
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --kubelet-use-node-status-port
        - --metric-resolution=15s
        image: registry.k8s.io/metrics-server/metrics-server:v0.7.2
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 3
          httpGet:
            path: /livez
            port: https
            scheme: HTTPS
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        name: metrics-server
        ports:
        - containerPort: 10250
          name: https
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /readyz
            port: https
            scheme: HTTPS
          initialDelaySeconds: 20
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        resources:
          requests:
            cpu: 100m
            memory: 200Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          seccompProfile:
            type: RuntimeDefault
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /tmp
          name: tmp-dir
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-s789l
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
      preemptionPolicy: PreemptLowerPriority
      priority: 2000000000
      priorityClassName: system-cluster-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: metrics-server
      serviceAccountName: metrics-server
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - emptyDir: {}
        name: tmp-dir
      - name: kube-api-access-s789l
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:34:31Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:34:29Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:34:29Z"
        message: 'containers with unready status: [metrics-server]'
        reason: ContainersNotReady
        status: "False"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:34:29Z"
        message: 'containers with unready status: [metrics-server]'
        reason: ContainersNotReady
        status: "False"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:34:29Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://e2648153ca25a89154ec1513bbacff84723ce838cea762364f68d71459c77ab2
        image: registry.k8s.io/metrics-server/metrics-server:v0.7.2
        imageID: docker-pullable://registry.k8s.io/metrics-server/metrics-server@sha256:ffcb2bf004d6aa0a17d90e0247cf94f2865c8901dcab4427034c341951c239f9
        lastState: {}
        name: metrics-server
        ready: false
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:34:30Z"
        volumeMounts:
        - mountPath: /tmp
          name: tmp-dir
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-s789l
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.13
      podIPs:
      - ip: 10.244.0.13
      qosClass: Burstable
      startTime: "2024-12-31T08:34:29Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:32:40Z"
      generateName: metrics-server-84c5f94fbc-
      labels:
        k8s-app: metrics-server
        pod-template-hash: 84c5f94fbc
      name: metrics-server-84c5f94fbc-vrzzn
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: metrics-server-84c5f94fbc
        uid: 6a8906b7-5782-4506-af7f-5b89ff4d1a1c
      resourceVersion: "1630"
      uid: f0a4c9c8-7866-4e4d-b2ed-43b07422b11c
    spec:
      containers:
      - args:
        - --cert-dir=/tmp
        - --secure-port=4443
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --kubelet-use-node-status-port
        - --metric-resolution=60s
        - --kubelet-insecure-tls
        image: registry.k8s.io/metrics-server/metrics-server:v0.7.2@sha256:ffcb2bf004d6aa0a17d90e0247cf94f2865c8901dcab4427034c341951c239f9
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 3
          httpGet:
            path: /livez
            port: https
            scheme: HTTPS
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        name: metrics-server
        ports:
        - containerPort: 4443
          name: https
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /readyz
            port: https
            scheme: HTTPS
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 1
        resources:
          requests:
            cpu: 100m
            memory: 200Mi
        securityContext:
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /tmp
          name: tmp-dir
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-4qth6
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 2000000000
      priorityClassName: system-cluster-critical
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: metrics-server
      serviceAccountName: metrics-server
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - emptyDir: {}
        name: tmp-dir
      - name: kube-api-access-4qth6
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:26Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:40Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:28Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:28Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:40Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://356a50b717eef915abd6d03976a19f89b480365e597577adb7487f003f26c58a
        image: registry.k8s.io/metrics-server/metrics-server@sha256:ffcb2bf004d6aa0a17d90e0247cf94f2865c8901dcab4427034c341951c239f9
        imageID: docker-pullable://registry.k8s.io/metrics-server/metrics-server@sha256:ffcb2bf004d6aa0a17d90e0247cf94f2865c8901dcab4427034c341951c239f9
        lastState: {}
        name: metrics-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:33:26Z"
        volumeMounts:
        - mountPath: /tmp
          name: tmp-dir
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-4qth6
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.12
      podIPs:
      - ip: 10.244.0.12
      qosClass: Burstable
      startTime: "2024-12-31T08:32:40Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Pod","metadata":{"annotations":{},"labels":{"addonmanager.kubernetes.io/mode":"Reconcile","integration-test":"storage-provisioner"},"name":"storage-provisioner","namespace":"kube-system"},"spec":{"containers":[{"command":["/storage-provisioner"],"image":"gcr.io/k8s-minikube/storage-provisioner:v5","imagePullPolicy":"IfNotPresent","name":"storage-provisioner","volumeMounts":[{"mountPath":"/tmp","name":"tmp"}]}],"hostNetwork":true,"serviceAccountName":"storage-provisioner","volumes":[{"hostPath":{"path":"/tmp","type":"Directory"},"name":"tmp"}]}}
      creationTimestamp: "2024-12-31T08:13:24Z"
      labels:
        addonmanager.kubernetes.io/mode: Reconcile
        integration-test: storage-provisioner
      name: storage-provisioner
      namespace: kube-system
      resourceVersion: "365"
      uid: d20ee962-a0c6-4e98-8805-fa6a44d4deba
    spec:
      containers:
      - command:
        - /storage-provisioner
        image: gcr.io/k8s-minikube/storage-provisioner:v5
        imagePullPolicy: IfNotPresent
        name: storage-provisioner
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /tmp
          name: tmp
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-6jv4q
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      hostNetwork: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: storage-provisioner
      serviceAccountName: storage-provisioner
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - hostPath:
          path: /tmp
          type: Directory
        name: tmp
      - name: kube-api-access-6jv4q
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:27Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:30Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:13:27Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://cbfa57616ce74bea6dee29cba97871b9dd9b812bfb493c063ab1805545f67bb9
        image: gcr.io/k8s-minikube/storage-provisioner:v5
        imageID: docker-pullable://gcr.io/k8s-minikube/storage-provisioner@sha256:18eb69d1418e854ad5a19e399310e52808a8321e4c441c1dddad8977a0d7a944
        lastState: {}
        name: storage-provisioner
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:13:29Z"
        volumeMounts:
        - mountPath: /tmp
          name: tmp
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-6jv4q
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 192.168.49.2
      podIPs:
      - ip: 192.168.49.2
      qosClass: BestEffort
      startTime: "2024-12-31T08:13:27Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      annotations:
        seccomp.security.alpha.kubernetes.io/pod: runtime/default
      creationTimestamp: "2024-12-31T08:32:36Z"
      generateName: dashboard-metrics-scraper-c5db448b4-
      labels:
        k8s-app: dashboard-metrics-scraper
        pod-template-hash: c5db448b4
      name: dashboard-metrics-scraper-c5db448b4-bgpww
      namespace: kubernetes-dashboard
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: dashboard-metrics-scraper-c5db448b4
        uid: f07ba36d-bf66-42f1-8869-e6047187883d
      resourceVersion: "1605"
      uid: b8150afb-d49d-4499-a0d6-f106cad05b52
    spec:
      containers:
      - image: docker.io/kubernetesui/metrics-scraper:v1.0.8@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 3
          httpGet:
            path: /
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 30
        name: dashboard-metrics-scraper
        ports:
        - containerPort: 8000
          protocol: TCP
        resources: {}
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsGroup: 2001
          runAsUser: 1001
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /tmp
          name: tmp-volume
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-7vcbf
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: kubernetes-dashboard
      serviceAccountName: kubernetes-dashboard
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - emptyDir: {}
        name: tmp-volume
      - name: kube-api-access-7vcbf
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:16Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:36Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:16Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:16Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:36Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://e80fa53da21c6d34469a8bdb3bdf6f21d50bca0114d793ee52632441cdceb4e6
        image: kubernetesui/metrics-scraper@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c
        imageID: docker-pullable://kubernetesui/metrics-scraper@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c
        lastState: {}
        name: dashboard-metrics-scraper
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:33:16Z"
        volumeMounts:
        - mountPath: /tmp
          name: tmp-volume
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-7vcbf
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.11
      podIPs:
      - ip: 10.244.0.11
      qosClass: BestEffort
      startTime: "2024-12-31T08:32:36Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T08:32:36Z"
      generateName: kubernetes-dashboard-695b96c756-
      labels:
        gcp-auth-skip-secret: "true"
        k8s-app: kubernetes-dashboard
        pod-template-hash: 695b96c756
      name: kubernetes-dashboard-695b96c756-m9lrb
      namespace: kubernetes-dashboard
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: kubernetes-dashboard-695b96c756
        uid: d272a318-2a13-4319-870e-ccfb4a95a999
      resourceVersion: "1588"
      uid: e6c5b6f8-cff9-43c1-b999-80198fe51477
    spec:
      containers:
      - args:
        - --namespace=kubernetes-dashboard
        - --enable-skip-login
        - --disable-settings-authorizer
        image: docker.io/kubernetesui/dashboard:v2.7.0@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 3
          httpGet:
            path: /
            port: 9090
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 30
        name: kubernetes-dashboard
        ports:
        - containerPort: 9090
          protocol: TCP
        resources: {}
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsGroup: 2001
          runAsUser: 1001
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /tmp
          name: tmp-volume
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-k9gwx
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      nodeSelector:
        kubernetes.io/os: linux
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: kubernetes-dashboard
      serviceAccountName: kubernetes-dashboard
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - emptyDir: {}
        name: tmp-volume
      - name: kube-api-access-k9gwx
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:06Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:36Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:06Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:33:06Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T08:32:36Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://f2105cf531e5f4ec66a020e5d119f47a78e0182850c91fe0484c6e9bc5d151f7
        image: kubernetesui/dashboard@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93
        imageID: docker-pullable://kubernetesui/dashboard@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93
        lastState: {}
        name: kubernetes-dashboard
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T08:33:03Z"
        volumeMounts:
        - mountPath: /tmp
          name: tmp-volume
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-k9gwx
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.10
      podIPs:
      - ip: 10.244.0.10
      qosClass: BestEffort
      startTime: "2024-12-31T08:32:36Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: model-server-664885b79f-
      labels:
        app.kubernetes.io/name: model-server
        pod-template-hash: 664885b79f
      name: model-server-664885b79f-9b4zn
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: model-server-664885b79f
        uid: a0ac1e23-5dc3-4896-9e95-cc90b8f61806
      resourceVersion: "9866"
      uid: 6f7495ab-5584-4478-ab6d-8739cc218edf
    spec:
      containers:
      - env:
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              key: hostname
              name: redis-config-v1
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              key: port
              name: redis-config-v1
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              key: db_password
              name: redis-secret-v1
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              key: model_name
              name: model-server-config-v1
        image: model-server:latest
        imagePullPolicy: Never
        name: model-server
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources:
          limits:
            cpu: "1"
            memory: 2Gi
          requests:
            cpu: "1"
            memory: 2Gi
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-vbrfn
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-vbrfn
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://1583e5fecddfa8be1db669555819a9df81301e9d20e46066bdfebb6ca42f94ce
        image: model-server:latest
        imageID: docker://sha256:56299850d16cdc2cb3a3408869773ecbf17ff41380435f09fdcba2564477517e
        lastState: {}
        name: model-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:45:52Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-vbrfn
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.44
      podIPs:
      - ip: 10.244.0.44
      qosClass: Guaranteed
      startTime: "2024-12-31T10:45:51Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: model-server-664885b79f-
      labels:
        app.kubernetes.io/name: model-server
        pod-template-hash: 664885b79f
      name: model-server-664885b79f-zgs5h
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: model-server-664885b79f
        uid: a0ac1e23-5dc3-4896-9e95-cc90b8f61806
      resourceVersion: "9871"
      uid: 829a8b15-1067-48a7-95c0-3c1f2412a613
    spec:
      containers:
      - env:
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              key: hostname
              name: redis-config-v1
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              key: port
              name: redis-config-v1
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              key: db_password
              name: redis-secret-v1
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              key: model_name
              name: model-server-config-v1
        image: model-server:latest
        imagePullPolicy: Never
        name: model-server
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources:
          limits:
            cpu: "1"
            memory: 2Gi
          requests:
            cpu: "1"
            memory: 2Gi
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-n8cp5
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-n8cp5
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://82aeb91384660a0d832b5c034b92d4a0b8c12b3014c74281e1d2b26d0ca3037f
        image: model-server:latest
        imageID: docker://sha256:56299850d16cdc2cb3a3408869773ecbf17ff41380435f09fdcba2564477517e
        lastState: {}
        name: model-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:45:52Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-n8cp5
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.43
      podIPs:
      - ip: 10.244.0.43
      qosClass: Guaranteed
      startTime: "2024-12-31T10:45:51Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: redis-69cf9b5676-
      labels:
        app.kubernetes.io/name: redis
        pod-template-hash: 69cf9b5676
        role: master
      name: redis-69cf9b5676-xnbrl
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: redis-69cf9b5676
        uid: e19f7630-f1e9-4091-bb32-bc61a0573fe6
      resourceVersion: "9910"
      uid: 28c8888e-62f7-4c72-a451-f5aafc281723
    spec:
      containers:
      - args:
        - --requirepass
        - $(REDIS_PASSWORD)
        command:
        - redis-server
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              key: db_password
              name: redis-secret-v1
        image: redis:7.4.1
        imagePullPolicy: IfNotPresent
        name: redis
        ports:
        - containerPort: 6379
          name: redis
          protocol: TCP
        resources:
          limits:
            cpu: 200m
            memory: 200Mi
          requests:
            cpu: 200m
            memory: 200Mi
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /data
          name: redis-storage
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-k2z8z
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
      - name: kube-api-access-k2z8z
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:46:05Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:46:04Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:46:05Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:46:05Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:46:04Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://0d95289ab079571e1c1c20f5ceeee90b55fbaff602f1feddc0a616c48e8e7e31
        image: redis:7.4.1
        imageID: docker-pullable://redis@sha256:bb142a9c18ac18a16713c1491d779697b4e107c22a97266616099d288237ef47
        lastState: {}
        name: redis
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:46:05Z"
        volumeMounts:
        - mountPath: /data
          name: redis-storage
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-k2z8z
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.46
      podIPs:
      - ip: 10.244.0.46
      qosClass: Guaranteed
      startTime: "2024-12-31T10:46:04Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: ui-server-76dc4d59d8-
      labels:
        app.kubernetes.io/name: ui-server
        pod-template-hash: 76dc4d59d8
      name: ui-server-76dc4d59d8-4gdc9
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: ui-server-76dc4d59d8
        uid: d293a584-ef0b-4420-9e1f-5896d07b3d7c
      resourceVersion: "9878"
      uid: ff964623-cc62-4daa-a670-fd2b10a86e17
    spec:
      containers:
      - env:
        - name: WEB_SERVER_URL
          valueFrom:
            configMapKeyRef:
              key: web_server_url
              name: web-server-config-v1
        image: ui-server:latest
        imagePullPolicy: Never
        name: ui-server
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources:
          limits:
            cpu: 500m
            memory: 1536Mi
          requests:
            cpu: 250m
            memory: 768Mi
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-bvzp8
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-bvzp8
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://1be9141abfa1df41795b684358b110764dbe4be11d8bcf8bb53e080adba79f23
        image: ui-server:latest
        imageID: docker://sha256:49607624ad85e97b9d4ae77f9c9cdb0bab33eb252ec1ef969f6375b584a823cd
        lastState: {}
        name: ui-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:45:52Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-bvzp8
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.42
      podIPs:
      - ip: 10.244.0.42
      qosClass: Burstable
      startTime: "2024-12-31T10:45:51Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: web-server-5c89b7fd87-
      labels:
        app.kubernetes.io/name: web-server
        pod-template-hash: 5c89b7fd87
      name: web-server-5c89b7fd87-t6vls
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: web-server-5c89b7fd87
        uid: 3eb4d83c-384e-49f5-ae40-5b99238cb221
      resourceVersion: "9862"
      uid: 42e7bfa0-7fe2-44af-b174-b57b9dbc1316
    spec:
      containers:
      - env:
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              key: hostname
              name: redis-config-v1
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              key: port
              name: redis-config-v1
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              key: db_password
              name: redis-secret-v1
        - name: MODEL_SERVER_URL
          valueFrom:
            configMapKeyRef:
              key: model_server_url
              name: model-server-config-v1
        image: web-server:latest
        imagePullPolicy: Never
        name: web-server
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-zjl78
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-zjl78
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://3c181efaf7e8e153bfb0721e055722aeaf932fa6460ab1653b41963bd92cb737
        image: web-server:latest
        imageID: docker://sha256:5102c64cf8a11b48b82716fbd8591f7f611344f8d30235d8ba192aadf13aae62
        lastState: {}
        name: web-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:45:52Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-zjl78
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.41
      podIPs:
      - ip: 10.244.0.41
      qosClass: BestEffort
      startTime: "2024-12-31T10:45:51Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-31T10:45:51Z"
      generateName: web-server-5c89b7fd87-
      labels:
        app.kubernetes.io/name: web-server
        pod-template-hash: 5c89b7fd87
      name: web-server-5c89b7fd87-xhs8r
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: web-server-5c89b7fd87
        uid: 3eb4d83c-384e-49f5-ae40-5b99238cb221
      resourceVersion: "9883"
      uid: d4777314-7051-4a4e-9b06-4b13e87da3db
    spec:
      containers:
      - env:
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              key: hostname
              name: redis-config-v1
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              key: port
              name: redis-config-v1
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              key: db_password
              name: redis-secret-v1
        - name: MODEL_SERVER_URL
          valueFrom:
            configMapKeyRef:
              key: model_server_url
              name: model-server-config-v1
        image: web-server:latest
        imagePullPolicy: Never
        name: web-server
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-s86fr
          readOnly: true
      dnsPolicy: ClusterFirst
      enableServiceLinks: true
      nodeName: minikube
      preemptionPolicy: PreemptLowerPriority
      priority: 0
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      serviceAccount: default
      serviceAccountName: default
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoExecute
        key: node.kubernetes.io/not-ready
        operator: Exists
        tolerationSeconds: 300
      - effect: NoExecute
        key: node.kubernetes.io/unreachable
        operator: Exists
        tolerationSeconds: 300
      volumes:
      - name: kube-api-access-s86fr
        projected:
          defaultMode: 420
          sources:
          - serviceAccountToken:
              expirationSeconds: 3607
              path: token
          - configMap:
              items:
              - key: ca.crt
                path: ca.crt
              name: kube-root-ca.crt
          - downwardAPI:
              items:
              - fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
                path: namespace
    status:
      conditions:
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:53Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-31T10:45:51Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://f69fa777908aecb285b838a6fca95b38a461843488420501bdfd34796c97679d
        image: web-server:latest
        imageID: docker://sha256:5102c64cf8a11b48b82716fbd8591f7f611344f8d30235d8ba192aadf13aae62
        lastState: {}
        name: web-server
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-31T10:45:52Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-s86fr
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.45
      podIPs:
      - ip: 10.244.0.45
      qosClass: BestEffort
      startTime: "2024-12-31T10:45:51Z"
  - apiVersion: v1
    kind: Service
    metadata:
      creationTimestamp: "2024-12-31T08:13:21Z"
      labels:
        component: apiserver
        provider: kubernetes
      name: kubernetes
      namespace: default
      resourceVersion: "198"
      uid: 168e2cf1-c2d8-4049-98c6-9367c71999b2
    spec:
      clusterIP: 10.96.0.1
      clusterIPs:
      - 10.96.0.1
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: https
        port: 443
        protocol: TCP
        targetPort: 8443
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-controller","namespace":"ingress-nginx"},"spec":{"ipFamilies":["IPv4"],"ipFamilyPolicy":"SingleStack","ports":[{"appProtocol":"http","name":"http","port":80,"protocol":"TCP","targetPort":"http"},{"appProtocol":"https","name":"https","port":443,"protocol":"TCP","targetPort":"https"}],"selector":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"type":"NodePort"}}
      creationTimestamp: "2024-12-31T08:31:31Z"
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      name: ingress-nginx-controller
      namespace: ingress-nginx
      resourceVersion: "1330"
      uid: b16dc56a-e727-44a3-9dec-f18eba92adac
    spec:
      clusterIP: 10.101.27.127
      clusterIPs:
      - 10.101.27.127
      externalTrafficPolicy: Cluster
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - appProtocol: http
        name: http
        nodePort: 32470
        port: 80
        protocol: TCP
        targetPort: http
      - appProtocol: https
        name: https
        nodePort: 30194
        port: 443
        protocol: TCP
        targetPort: https
      selector:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      sessionAffinity: None
      type: NodePort
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-controller-admission","namespace":"ingress-nginx"},"spec":{"ports":[{"appProtocol":"https","name":"https-webhook","port":443,"targetPort":"webhook"}],"selector":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"type":"ClusterIP"}}
      creationTimestamp: "2024-12-31T08:31:31Z"
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      name: ingress-nginx-controller-admission
      namespace: ingress-nginx
      resourceVersion: "1334"
      uid: c7c5f403-0b29-4971-8349-5051b2f6410b
    spec:
      clusterIP: 10.96.62.132
      clusterIPs:
      - 10.96.62.132
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - appProtocol: https
        name: https-webhook
        port: 443
        protocol: TCP
        targetPort: webhook
      selector:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        prometheus.io/port: "9153"
        prometheus.io/scrape: "true"
      creationTimestamp: "2024-12-31T08:13:23Z"
      labels:
        k8s-app: kube-dns
        kubernetes.io/cluster-service: "true"
        kubernetes.io/name: CoreDNS
      name: kube-dns
      namespace: kube-system
      resourceVersion: "241"
      uid: 640f245a-2128-4a71-be0a-a8c2f419d692
    spec:
      clusterIP: 10.96.0.10
      clusterIPs:
      - 10.96.0.10
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: dns
        port: 53
        protocol: UDP
        targetPort: 53
      - name: dns-tcp
        port: 53
        protocol: TCP
        targetPort: 53
      - name: metrics
        port: 9153
        protocol: TCP
        targetPort: 9153
      selector:
        k8s-app: kube-dns
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"k8s-app":"metrics-server"},"name":"metrics-server","namespace":"kube-system"},"spec":{"ports":[{"name":"https","port":443,"protocol":"TCP","targetPort":"https"}],"selector":{"k8s-app":"metrics-server"}}}
      creationTimestamp: "2024-12-31T08:32:40Z"
      labels:
        k8s-app: metrics-server
      name: metrics-server
      namespace: kube-system
      resourceVersion: "1697"
      uid: f1fba3fd-9da3-481e-af15-1da71d3573b8
    spec:
      clusterIP: 10.103.12.76
      clusterIPs:
      - 10.103.12.76
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: https
        port: 443
        protocol: TCP
        targetPort: https
      selector:
        k8s-app: metrics-server
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"addonmanager.kubernetes.io/mode":"Reconcile","k8s-app":"dashboard-metrics-scraper","kubernetes.io/minikube-addons":"dashboard"},"name":"dashboard-metrics-scraper","namespace":"kubernetes-dashboard"},"spec":{"ports":[{"port":8000,"targetPort":8000}],"selector":{"k8s-app":"dashboard-metrics-scraper"}}}
      creationTimestamp: "2024-12-31T08:32:36Z"
      labels:
        addonmanager.kubernetes.io/mode: Reconcile
        k8s-app: dashboard-metrics-scraper
        kubernetes.io/minikube-addons: dashboard
      name: dashboard-metrics-scraper
      namespace: kubernetes-dashboard
      resourceVersion: "1513"
      uid: cf7f028a-076d-4e6a-b13d-580c8c11ce13
    spec:
      clusterIP: 10.96.168.182
      clusterIPs:
      - 10.96.168.182
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - port: 8000
        protocol: TCP
        targetPort: 8000
      selector:
        k8s-app: dashboard-metrics-scraper
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"addonmanager.kubernetes.io/mode":"Reconcile","k8s-app":"kubernetes-dashboard","kubernetes.io/minikube-addons":"dashboard"},"name":"kubernetes-dashboard","namespace":"kubernetes-dashboard"},"spec":{"ports":[{"port":80,"targetPort":9090}],"selector":{"k8s-app":"kubernetes-dashboard"}}}
      creationTimestamp: "2024-12-31T08:32:36Z"
      labels:
        addonmanager.kubernetes.io/mode: Reconcile
        k8s-app: kubernetes-dashboard
        kubernetes.io/minikube-addons: dashboard
      name: kubernetes-dashboard
      namespace: kubernetes-dashboard
      resourceVersion: "1509"
      uid: c6a41a1a-2f27-4c31-aabd-83037aeda3e5
    spec:
      clusterIP: 10.108.205.197
      clusterIPs:
      - 10.108.205.197
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - port: 80
        protocol: TCP
        targetPort: 9090
      selector:
        k8s-app: kubernetes-dashboard
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: model-server
        app.kubernetes.io/part-of: fastapi-app
      name: model-server-service
      namespace: prod
      resourceVersion: "9773"
      uid: 8d2e73c1-e052-4883-af0f-475294f60d43
    spec:
      clusterIP: 10.104.77.73
      clusterIPs:
      - 10.104.77.73
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: http
        port: 80
        protocol: TCP
        targetPort: 80
      selector:
        app.kubernetes.io/name: model-server
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: redis
        app.kubernetes.io/part-of: fastapi-app
        role: master
      name: redis-service
      namespace: prod
      resourceVersion: "9777"
      uid: 9d907b8d-08f5-44ac-afbd-e24e1e6a0447
    spec:
      clusterIP: 10.96.54.213
      clusterIPs:
      - 10.96.54.213
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: redis
        port: 6379
        protocol: TCP
        targetPort: 6379
      selector:
        app.kubernetes.io/name: redis
        role: master
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      labels:
        app.kubernetes.io/instance: fastapi-release-prod
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: ui-server
        app.kubernetes.io/part-of: fastapi-app
      name: ui-server-service
      namespace: prod
      resourceVersion: "9765"
      uid: 25e883b7-6325-4585-a3c8-0f35482217d4
    spec:
      clusterIP: 10.107.19.225
      clusterIPs:
      - 10.107.19.225
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: http
        port: 80
        protocol: TCP
        targetPort: 80
      selector:
        app.kubernetes.io/name: ui-server
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      labels:
        app.kubernetes.io/instance: fastapi-release-prod
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: web-server
        app.kubernetes.io/part-of: fastapi-app
      name: web-server-service
      namespace: prod
      resourceVersion: "9768"
      uid: a40e8412-f24a-4f67-ad8f-27f9463f0441
    spec:
      clusterIP: 10.105.1.197
      clusterIPs:
      - 10.105.1.197
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - name: http
        port: 80
        protocol: TCP
        targetPort: 80
      selector:
        app.kubernetes.io/name: web-server
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: apps/v1
    kind: DaemonSet
    metadata:
      annotations:
        deprecated.daemonset.template.generation: "1"
      creationTimestamp: "2024-12-31T08:13:23Z"
      generation: 1
      labels:
        k8s-app: kube-proxy
      name: kube-proxy
      namespace: kube-system
      resourceVersion: "358"
      uid: 3a06cdae-3f0c-4247-a662-370d4a742ead
    spec:
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          k8s-app: kube-proxy
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: kube-proxy
        spec:
          containers:
          - command:
            - /usr/local/bin/kube-proxy
            - --config=/var/lib/kube-proxy/config.conf
            - --hostname-override=$(NODE_NAME)
            env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: spec.nodeName
            image: registry.k8s.io/kube-proxy:v1.31.0
            imagePullPolicy: IfNotPresent
            name: kube-proxy
            resources: {}
            securityContext:
              privileged: true
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /var/lib/kube-proxy
              name: kube-proxy
            - mountPath: /run/xtables.lock
              name: xtables-lock
            - mountPath: /lib/modules
              name: lib-modules
              readOnly: true
          dnsPolicy: ClusterFirst
          hostNetwork: true
          nodeSelector:
            kubernetes.io/os: linux
          priorityClassName: system-node-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: kube-proxy
          serviceAccountName: kube-proxy
          terminationGracePeriodSeconds: 30
          tolerations:
          - operator: Exists
          volumes:
          - configMap:
              defaultMode: 420
              name: kube-proxy
            name: kube-proxy
          - hostPath:
              path: /run/xtables.lock
              type: FileOrCreate
            name: xtables-lock
          - hostPath:
              path: /lib/modules
              type: ""
            name: lib-modules
      updateStrategy:
        rollingUpdate:
          maxSurge: 0
          maxUnavailable: 1
        type: RollingUpdate
    status:
      currentNumberScheduled: 1
      desiredNumberScheduled: 1
      numberAvailable: 1
      numberMisscheduled: 0
      numberReady: 1
      observedGeneration: 1
      updatedNumberScheduled: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-controller","namespace":"ingress-nginx"},"spec":{"minReadySeconds":0,"revisionHistoryLimit":10,"selector":{"matchLabels":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"}},"strategy":{"rollingUpdate":{"maxUnavailable":1},"type":"RollingUpdate"},"template":{"metadata":{"labels":{"app.kubernetes.io/component":"controller","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx","gcp-auth-skip-secret":"true"}},"spec":{"containers":[{"args":["/nginx-ingress-controller","--election-id=ingress-nginx-leader","--controller-class=k8s.io/ingress-nginx","--watch-ingress-without-class=true","--configmap=$(POD_NAMESPACE)/ingress-nginx-controller","--tcp-services-configmap=$(POD_NAMESPACE)/tcp-services","--udp-services-configmap=$(POD_NAMESPACE)/udp-services","--validating-webhook=:8443","--validating-webhook-certificate=/usr/local/certificates/cert","--validating-webhook-key=/usr/local/certificates/key"],"env":[{"name":"POD_NAME","valueFrom":{"fieldRef":{"fieldPath":"metadata.name"}}},{"name":"POD_NAMESPACE","valueFrom":{"fieldRef":{"fieldPath":"metadata.namespace"}}},{"name":"LD_PRELOAD","value":"/usr/local/lib/libmimalloc.so"}],"image":"registry.k8s.io/ingress-nginx/controller:v1.11.2@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce","imagePullPolicy":"IfNotPresent","lifecycle":{"preStop":{"exec":{"command":["/wait-shutdown"]}}},"livenessProbe":{"failureThreshold":5,"httpGet":{"path":"/healthz","port":10254,"scheme":"HTTP"},"initialDelaySeconds":10,"periodSeconds":10,"successThreshold":1,"timeoutSeconds":1},"name":"controller","ports":[{"containerPort":80,"hostPort":80,"name":"http","protocol":"TCP"},{"containerPort":443,"hostPort":443,"name":"https","protocol":"TCP"},{"containerPort":8443,"name":"webhook","protocol":"TCP"}],"readinessProbe":{"failureThreshold":3,"httpGet":{"path":"/healthz","port":10254,"scheme":"HTTP"},"initialDelaySeconds":10,"periodSeconds":10,"successThreshold":1,"timeoutSeconds":1},"resources":{"requests":{"cpu":"100m","memory":"90Mi"}},"securityContext":{"allowPrivilegeEscalation":true,"capabilities":{"add":["NET_BIND_SERVICE"],"drop":["ALL"]},"runAsUser":101},"volumeMounts":[{"mountPath":"/usr/local/certificates/","name":"webhook-cert","readOnly":true}]}],"dnsPolicy":"ClusterFirst","nodeSelector":{"kubernetes.io/os":"linux","minikube.k8s.io/primary":"true"},"serviceAccountName":"ingress-nginx","terminationGracePeriodSeconds":0,"tolerations":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master","operator":"Equal"}],"volumes":[{"name":"webhook-cert","secret":{"secretName":"ingress-nginx-admission"}}]}}}}
      creationTimestamp: "2024-12-31T08:31:31Z"
      generation: 1
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      name: ingress-nginx-controller
      namespace: ingress-nginx
      resourceVersion: "1560"
      uid: 70f186ed-c28f-44ca-8d9b-601ef3d23296
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app.kubernetes.io/component: controller
          app.kubernetes.io/instance: ingress-nginx
          app.kubernetes.io/name: ingress-nginx
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 1
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/component: controller
            app.kubernetes.io/instance: ingress-nginx
            app.kubernetes.io/name: ingress-nginx
            gcp-auth-skip-secret: "true"
        spec:
          containers:
          - args:
            - /nginx-ingress-controller
            - --election-id=ingress-nginx-leader
            - --controller-class=k8s.io/ingress-nginx
            - --watch-ingress-without-class=true
            - --configmap=$(POD_NAMESPACE)/ingress-nginx-controller
            - --tcp-services-configmap=$(POD_NAMESPACE)/tcp-services
            - --udp-services-configmap=$(POD_NAMESPACE)/udp-services
            - --validating-webhook=:8443
            - --validating-webhook-certificate=/usr/local/certificates/cert
            - --validating-webhook-key=/usr/local/certificates/key
            env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
            - name: LD_PRELOAD
              value: /usr/local/lib/libmimalloc.so
            image: registry.k8s.io/ingress-nginx/controller:v1.11.2@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce
            imagePullPolicy: IfNotPresent
            lifecycle:
              preStop:
                exec:
                  command:
                  - /wait-shutdown
            livenessProbe:
              failureThreshold: 5
              httpGet:
                path: /healthz
                port: 10254
                scheme: HTTP
              initialDelaySeconds: 10
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            name: controller
            ports:
            - containerPort: 80
              hostPort: 80
              name: http
              protocol: TCP
            - containerPort: 443
              hostPort: 443
              name: https
              protocol: TCP
            - containerPort: 8443
              name: webhook
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /healthz
                port: 10254
                scheme: HTTP
              initialDelaySeconds: 10
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              requests:
                cpu: 100m
                memory: 90Mi
            securityContext:
              allowPrivilegeEscalation: true
              capabilities:
                add:
                - NET_BIND_SERVICE
                drop:
                - ALL
              runAsUser: 101
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /usr/local/certificates/
              name: webhook-cert
              readOnly: true
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
            minikube.k8s.io/primary: "true"
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: ingress-nginx
          serviceAccountName: ingress-nginx
          terminationGracePeriodSeconds: 0
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
            operator: Equal
          volumes:
          - name: webhook-cert
            secret:
              defaultMode: 420
              secretName: ingress-nginx-admission
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T08:31:31Z"
        lastUpdateTime: "2024-12-31T08:31:31Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T08:31:31Z"
        lastUpdateTime: "2024-12-31T08:32:42Z"
        message: ReplicaSet "ingress-nginx-controller-bc57996ff" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:13:23Z"
      generation: 2
      labels:
        k8s-app: kube-dns
      name: coredns
      namespace: kube-system
      resourceVersion: "415"
      uid: da78862e-b740-4fc6-9320-72fbc6021295
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          k8s-app: kube-dns
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 1
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: kube-dns
        spec:
          affinity:
            podAntiAffinity:
              preferredDuringSchedulingIgnoredDuringExecution:
              - podAffinityTerm:
                  labelSelector:
                    matchExpressions:
                    - key: k8s-app
                      operator: In
                      values:
                      - kube-dns
                  topologyKey: kubernetes.io/hostname
                weight: 100
          containers:
          - args:
            - -conf
            - /etc/coredns/Corefile
            image: registry.k8s.io/coredns/coredns:v1.11.1
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 5
              httpGet:
                path: /health
                port: 8080
                scheme: HTTP
              initialDelaySeconds: 60
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 5
            name: coredns
            ports:
            - containerPort: 53
              name: dns
              protocol: UDP
            - containerPort: 53
              name: dns-tcp
              protocol: TCP
            - containerPort: 9153
              name: metrics
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /ready
                port: 8181
                scheme: HTTP
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              limits:
                memory: 170Mi
              requests:
                cpu: 100m
                memory: 70Mi
            securityContext:
              allowPrivilegeEscalation: false
              capabilities:
                add:
                - NET_BIND_SERVICE
                drop:
                - ALL
              readOnlyRootFilesystem: true
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /etc/coredns
              name: config-volume
              readOnly: true
          dnsPolicy: Default
          nodeSelector:
            kubernetes.io/os: linux
          priorityClassName: system-cluster-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: coredns
          serviceAccountName: coredns
          terminationGracePeriodSeconds: 30
          tolerations:
          - key: CriticalAddonsOnly
            operator: Exists
          - effect: NoSchedule
            key: node-role.kubernetes.io/control-plane
          volumes:
          - configMap:
              defaultMode: 420
              items:
              - key: Corefile
                path: Corefile
              name: coredns
            name: config-volume
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T08:13:28Z"
        lastUpdateTime: "2024-12-31T08:13:28Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T08:13:28Z"
        lastUpdateTime: "2024-12-31T08:14:09Z"
        message: ReplicaSet "coredns-6f6b679f8f" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 2
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "2"
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"k8s-app":"metrics-server"},"name":"metrics-server","namespace":"kube-system"},"spec":{"selector":{"matchLabels":{"k8s-app":"metrics-server"}},"strategy":{"rollingUpdate":{"maxUnavailable":0}},"template":{"metadata":{"labels":{"k8s-app":"metrics-server"}},"spec":{"containers":[{"args":["--cert-dir=/tmp","--secure-port=10250","--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname","--kubelet-use-node-status-port","--metric-resolution=15s"],"image":"registry.k8s.io/metrics-server/metrics-server:v0.7.2","imagePullPolicy":"IfNotPresent","livenessProbe":{"failureThreshold":3,"httpGet":{"path":"/livez","port":"https","scheme":"HTTPS"},"periodSeconds":10},"name":"metrics-server","ports":[{"containerPort":10250,"name":"https","protocol":"TCP"}],"readinessProbe":{"failureThreshold":3,"httpGet":{"path":"/readyz","port":"https","scheme":"HTTPS"},"initialDelaySeconds":20,"periodSeconds":10},"resources":{"requests":{"cpu":"100m","memory":"200Mi"}},"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]},"readOnlyRootFilesystem":true,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}},"volumeMounts":[{"mountPath":"/tmp","name":"tmp-dir"}]}],"nodeSelector":{"kubernetes.io/os":"linux"},"priorityClassName":"system-cluster-critical","serviceAccountName":"metrics-server","volumes":[{"emptyDir":{},"name":"tmp-dir"}]}}}}
      creationTimestamp: "2024-12-31T08:32:40Z"
      generation: 2
      labels:
        k8s-app: metrics-server
      name: metrics-server
      namespace: kube-system
      resourceVersion: "2480"
      uid: 42fe7c85-6395-4f9e-af22-2d5958454fb1
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          k8s-app: metrics-server
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 0
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: metrics-server
        spec:
          containers:
          - args:
            - --cert-dir=/tmp
            - --secure-port=10250
            - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
            - --kubelet-use-node-status-port
            - --metric-resolution=15s
            image: registry.k8s.io/metrics-server/metrics-server:v0.7.2
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /livez
                port: https
                scheme: HTTPS
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            name: metrics-server
            ports:
            - containerPort: 10250
              name: https
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /readyz
                port: https
                scheme: HTTPS
              initialDelaySeconds: 20
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              requests:
                cpu: 100m
                memory: 200Mi
            securityContext:
              allowPrivilegeEscalation: false
              capabilities:
                drop:
                - ALL
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              seccompProfile:
                type: RuntimeDefault
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-dir
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          priorityClassName: system-cluster-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: metrics-server
          serviceAccountName: metrics-server
          terminationGracePeriodSeconds: 30
          volumes:
          - emptyDir: {}
            name: tmp-dir
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T08:33:28Z"
        lastUpdateTime: "2024-12-31T08:33:28Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T08:45:17Z"
        lastUpdateTime: "2024-12-31T08:45:17Z"
        message: ReplicaSet "metrics-server-54bf7cdd6" has timed out progressing.
        reason: ProgressDeadlineExceeded
        status: "False"
        type: Progressing
      observedGeneration: 2
      readyReplicas: 1
      replicas: 2
      unavailableReplicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"addonmanager.kubernetes.io/mode":"Reconcile","k8s-app":"dashboard-metrics-scraper","kubernetes.io/minikube-addons":"dashboard"},"name":"dashboard-metrics-scraper","namespace":"kubernetes-dashboard"},"spec":{"replicas":1,"revisionHistoryLimit":10,"selector":{"matchLabels":{"k8s-app":"dashboard-metrics-scraper"}},"template":{"metadata":{"annotations":{"seccomp.security.alpha.kubernetes.io/pod":"runtime/default"},"labels":{"k8s-app":"dashboard-metrics-scraper"}},"spec":{"containers":[{"image":"docker.io/kubernetesui/metrics-scraper:v1.0.8@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c","livenessProbe":{"httpGet":{"path":"/","port":8000,"scheme":"HTTP"},"initialDelaySeconds":30,"timeoutSeconds":30},"name":"dashboard-metrics-scraper","ports":[{"containerPort":8000,"protocol":"TCP"}],"securityContext":{"allowPrivilegeEscalation":false,"readOnlyRootFilesystem":true,"runAsGroup":2001,"runAsUser":1001},"volumeMounts":[{"mountPath":"/tmp","name":"tmp-volume"}]}],"nodeSelector":{"kubernetes.io/os":"linux"},"serviceAccountName":"kubernetes-dashboard","tolerations":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master"}],"volumes":[{"emptyDir":{},"name":"tmp-volume"}]}}}}
      creationTimestamp: "2024-12-31T08:32:36Z"
      generation: 1
      labels:
        addonmanager.kubernetes.io/mode: Reconcile
        k8s-app: dashboard-metrics-scraper
        kubernetes.io/minikube-addons: dashboard
      name: dashboard-metrics-scraper
      namespace: kubernetes-dashboard
      resourceVersion: "1609"
      uid: 2c41240a-19e2-4adf-98bd-5cc14ad6bb1c
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          k8s-app: dashboard-metrics-scraper
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          annotations:
            seccomp.security.alpha.kubernetes.io/pod: runtime/default
          creationTimestamp: null
          labels:
            k8s-app: dashboard-metrics-scraper
        spec:
          containers:
          - image: docker.io/kubernetesui/metrics-scraper:v1.0.8@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /
                port: 8000
                scheme: HTTP
              initialDelaySeconds: 30
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 30
            name: dashboard-metrics-scraper
            ports:
            - containerPort: 8000
              protocol: TCP
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsGroup: 2001
              runAsUser: 1001
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-volume
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: kubernetes-dashboard
          serviceAccountName: kubernetes-dashboard
          terminationGracePeriodSeconds: 30
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
          volumes:
          - emptyDir: {}
            name: tmp-volume
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T08:33:16Z"
        lastUpdateTime: "2024-12-31T08:33:16Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T08:32:36Z"
        lastUpdateTime: "2024-12-31T08:33:16Z"
        message: ReplicaSet "dashboard-metrics-scraper-c5db448b4" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"addonmanager.kubernetes.io/mode":"Reconcile","k8s-app":"kubernetes-dashboard","kubernetes.io/minikube-addons":"dashboard"},"name":"kubernetes-dashboard","namespace":"kubernetes-dashboard"},"spec":{"replicas":1,"revisionHistoryLimit":10,"selector":{"matchLabels":{"k8s-app":"kubernetes-dashboard"}},"template":{"metadata":{"labels":{"gcp-auth-skip-secret":"true","k8s-app":"kubernetes-dashboard"}},"spec":{"containers":[{"args":["--namespace=kubernetes-dashboard","--enable-skip-login","--disable-settings-authorizer"],"image":"docker.io/kubernetesui/dashboard:v2.7.0@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93","livenessProbe":{"httpGet":{"path":"/","port":9090},"initialDelaySeconds":30,"timeoutSeconds":30},"name":"kubernetes-dashboard","ports":[{"containerPort":9090,"protocol":"TCP"}],"securityContext":{"allowPrivilegeEscalation":false,"readOnlyRootFilesystem":true,"runAsGroup":2001,"runAsUser":1001},"volumeMounts":[{"mountPath":"/tmp","name":"tmp-volume"}]}],"nodeSelector":{"kubernetes.io/os":"linux"},"serviceAccountName":"kubernetes-dashboard","tolerations":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master"}],"volumes":[{"emptyDir":{},"name":"tmp-volume"}]}}}}
      creationTimestamp: "2024-12-31T08:32:36Z"
      generation: 1
      labels:
        addonmanager.kubernetes.io/mode: Reconcile
        k8s-app: kubernetes-dashboard
        kubernetes.io/minikube-addons: dashboard
      name: kubernetes-dashboard
      namespace: kubernetes-dashboard
      resourceVersion: "1592"
      uid: d96a0a26-d0b1-4ef1-ae5e-c01a32114593
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          k8s-app: kubernetes-dashboard
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            gcp-auth-skip-secret: "true"
            k8s-app: kubernetes-dashboard
        spec:
          containers:
          - args:
            - --namespace=kubernetes-dashboard
            - --enable-skip-login
            - --disable-settings-authorizer
            image: docker.io/kubernetesui/dashboard:v2.7.0@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /
                port: 9090
                scheme: HTTP
              initialDelaySeconds: 30
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 30
            name: kubernetes-dashboard
            ports:
            - containerPort: 9090
              protocol: TCP
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsGroup: 2001
              runAsUser: 1001
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-volume
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: kubernetes-dashboard
          serviceAccountName: kubernetes-dashboard
          terminationGracePeriodSeconds: 30
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
          volumes:
          - emptyDir: {}
            name: tmp-volume
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T08:33:06Z"
        lastUpdateTime: "2024-12-31T08:33:06Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T08:32:36Z"
        lastUpdateTime: "2024-12-31T08:33:06Z"
        message: ReplicaSet "kubernetes-dashboard-695b96c756" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: model-server
        app.kubernetes.io/part-of: fastapi-app
      name: model-server
      namespace: prod
      resourceVersion: "9877"
      uid: 9486f26c-8d5b-4ff8-8119-7aa4fefc14ef
    spec:
      progressDeadlineSeconds: 600
      replicas: 2
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app.kubernetes.io/name: model-server
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: model-server
        spec:
          containers:
          - env:
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  key: hostname
                  name: redis-config-v1
            - name: REDIS_PORT
              valueFrom:
                configMapKeyRef:
                  key: port
                  name: redis-config-v1
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            - name: MODEL_NAME
              valueFrom:
                configMapKeyRef:
                  key: model_name
                  name: model-server-config-v1
            image: model-server:latest
            imagePullPolicy: Never
            name: model-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources:
              limits:
                cpu: "1"
                memory: 2Gi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 2
      conditions:
      - lastTransitionTime: "2024-12-31T10:45:53Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T10:45:51Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: ReplicaSet "model-server-664885b79f" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 2
      replicas: 2
      updatedReplicas: 2
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: redis
        app.kubernetes.io/part-of: fastapi-app
      name: redis
      namespace: prod
      resourceVersion: "9914"
      uid: 17fddc35-7ff5-4b50-a3ac-70d3029d154b
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app.kubernetes.io/name: redis
          role: master
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: redis
            role: master
        spec:
          containers:
          - args:
            - --requirepass
            - $(REDIS_PASSWORD)
            command:
            - redis-server
            env:
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            image: redis:7.4.1
            imagePullPolicy: IfNotPresent
            name: redis
            ports:
            - containerPort: 6379
              name: redis
              protocol: TCP
            resources:
              limits:
                cpu: 200m
                memory: 200Mi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /data
              name: redis-storage
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
          volumes:
          - name: redis-storage
            persistentVolumeClaim:
              claimName: redis-pvc
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T10:46:05Z"
        lastUpdateTime: "2024-12-31T10:46:05Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T10:45:51Z"
        lastUpdateTime: "2024-12-31T10:46:05Z"
        message: ReplicaSet "redis-69cf9b5676" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: ui-server
        app.kubernetes.io/part-of: fastapi-app
      name: ui-server
      namespace: prod
      resourceVersion: "9882"
      uid: e71679b4-71f0-4bc5-9864-0b83b44c1202
    spec:
      progressDeadlineSeconds: 600
      replicas: 1
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app.kubernetes.io/name: ui-server
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: ui-server
        spec:
          containers:
          - env:
            - name: WEB_SERVER_URL
              valueFrom:
                configMapKeyRef:
                  key: web_server_url
                  name: web-server-config-v1
            image: ui-server:latest
            imagePullPolicy: Never
            name: ui-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources:
              limits:
                cpu: 500m
                memory: 1536Mi
              requests:
                cpu: 250m
                memory: 768Mi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 1
      conditions:
      - lastTransitionTime: "2024-12-31T10:45:53Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T10:45:51Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: ReplicaSet "ui-server-76dc4d59d8" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
      updatedReplicas: 1
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/managed-by: Helm
        app.kubernetes.io/name: web-server
        app.kubernetes.io/part-of: fastapi-app
      name: web-server
      namespace: prod
      resourceVersion: "9887"
      uid: 5b161fe0-3bd6-4748-b937-5956d10ee439
    spec:
      progressDeadlineSeconds: 600
      replicas: 2
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app.kubernetes.io/name: web-server
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: web-server
        spec:
          containers:
          - env:
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  key: hostname
                  name: redis-config-v1
            - name: REDIS_PORT
              valueFrom:
                configMapKeyRef:
                  key: port
                  name: redis-config-v1
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            - name: MODEL_SERVER_URL
              valueFrom:
                configMapKeyRef:
                  key: model_server_url
                  name: model-server-config-v1
            image: web-server:latest
            imagePullPolicy: Never
            name: web-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources: {}
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 2
      conditions:
      - lastTransitionTime: "2024-12-31T10:45:53Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-31T10:45:51Z"
        lastUpdateTime: "2024-12-31T10:45:53Z"
        message: ReplicaSet "web-server-5c89b7fd87" has successfully progressed.
        reason: NewReplicaSetAvailable
        status: "True"
        type: Progressing
      observedGeneration: 1
      readyReplicas: 2
      replicas: 2
      updatedReplicas: 2
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:31:31Z"
      generation: 1
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
        gcp-auth-skip-secret: "true"
        pod-template-hash: bc57996ff
      name: ingress-nginx-controller-bc57996ff
      namespace: ingress-nginx
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: ingress-nginx-controller
        uid: 70f186ed-c28f-44ca-8d9b-601ef3d23296
      resourceVersion: "1559"
      uid: f8ffdf94-1ef8-4048-b820-4858c0c6d57d
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/component: controller
          app.kubernetes.io/instance: ingress-nginx
          app.kubernetes.io/name: ingress-nginx
          pod-template-hash: bc57996ff
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/component: controller
            app.kubernetes.io/instance: ingress-nginx
            app.kubernetes.io/name: ingress-nginx
            gcp-auth-skip-secret: "true"
            pod-template-hash: bc57996ff
        spec:
          containers:
          - args:
            - /nginx-ingress-controller
            - --election-id=ingress-nginx-leader
            - --controller-class=k8s.io/ingress-nginx
            - --watch-ingress-without-class=true
            - --configmap=$(POD_NAMESPACE)/ingress-nginx-controller
            - --tcp-services-configmap=$(POD_NAMESPACE)/tcp-services
            - --udp-services-configmap=$(POD_NAMESPACE)/udp-services
            - --validating-webhook=:8443
            - --validating-webhook-certificate=/usr/local/certificates/cert
            - --validating-webhook-key=/usr/local/certificates/key
            env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
            - name: LD_PRELOAD
              value: /usr/local/lib/libmimalloc.so
            image: registry.k8s.io/ingress-nginx/controller:v1.11.2@sha256:d5f8217feeac4887cb1ed21f27c2674e58be06bd8f5184cacea2a69abaf78dce
            imagePullPolicy: IfNotPresent
            lifecycle:
              preStop:
                exec:
                  command:
                  - /wait-shutdown
            livenessProbe:
              failureThreshold: 5
              httpGet:
                path: /healthz
                port: 10254
                scheme: HTTP
              initialDelaySeconds: 10
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            name: controller
            ports:
            - containerPort: 80
              hostPort: 80
              name: http
              protocol: TCP
            - containerPort: 443
              hostPort: 443
              name: https
              protocol: TCP
            - containerPort: 8443
              name: webhook
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /healthz
                port: 10254
                scheme: HTTP
              initialDelaySeconds: 10
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              requests:
                cpu: 100m
                memory: 90Mi
            securityContext:
              allowPrivilegeEscalation: true
              capabilities:
                add:
                - NET_BIND_SERVICE
                drop:
                - ALL
              runAsUser: 101
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /usr/local/certificates/
              name: webhook-cert
              readOnly: true
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
            minikube.k8s.io/primary: "true"
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: ingress-nginx
          serviceAccountName: ingress-nginx
          terminationGracePeriodSeconds: 0
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
            operator: Equal
          volumes:
          - name: webhook-cert
            secret:
              defaultMode: 420
              secretName: ingress-nginx-admission
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:13:28Z"
      generation: 1
      labels:
        k8s-app: kube-dns
        pod-template-hash: 6f6b679f8f
      name: coredns-6f6b679f8f
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: coredns
        uid: da78862e-b740-4fc6-9320-72fbc6021295
      resourceVersion: "414"
      uid: a04cc8c6-fdc8-43b8-b167-be7dbbab4644
    spec:
      replicas: 1
      selector:
        matchLabels:
          k8s-app: kube-dns
          pod-template-hash: 6f6b679f8f
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: kube-dns
            pod-template-hash: 6f6b679f8f
        spec:
          affinity:
            podAntiAffinity:
              preferredDuringSchedulingIgnoredDuringExecution:
              - podAffinityTerm:
                  labelSelector:
                    matchExpressions:
                    - key: k8s-app
                      operator: In
                      values:
                      - kube-dns
                  topologyKey: kubernetes.io/hostname
                weight: 100
          containers:
          - args:
            - -conf
            - /etc/coredns/Corefile
            image: registry.k8s.io/coredns/coredns:v1.11.1
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 5
              httpGet:
                path: /health
                port: 8080
                scheme: HTTP
              initialDelaySeconds: 60
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 5
            name: coredns
            ports:
            - containerPort: 53
              name: dns
              protocol: UDP
            - containerPort: 53
              name: dns-tcp
              protocol: TCP
            - containerPort: 9153
              name: metrics
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /ready
                port: 8181
                scheme: HTTP
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              limits:
                memory: 170Mi
              requests:
                cpu: 100m
                memory: 70Mi
            securityContext:
              allowPrivilegeEscalation: false
              capabilities:
                add:
                - NET_BIND_SERVICE
                drop:
                - ALL
              readOnlyRootFilesystem: true
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /etc/coredns
              name: config-volume
              readOnly: true
          dnsPolicy: Default
          nodeSelector:
            kubernetes.io/os: linux
          priorityClassName: system-cluster-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: coredns
          serviceAccountName: coredns
          terminationGracePeriodSeconds: 30
          tolerations:
          - key: CriticalAddonsOnly
            operator: Exists
          - effect: NoSchedule
            key: node-role.kubernetes.io/control-plane
          volumes:
          - configMap:
              defaultMode: 420
              items:
              - key: Corefile
                path: Corefile
              name: coredns
            name: config-volume
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "2"
      creationTimestamp: "2024-12-31T08:34:29Z"
      generation: 1
      labels:
        k8s-app: metrics-server
        pod-template-hash: 54bf7cdd6
      name: metrics-server-54bf7cdd6
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: metrics-server
        uid: 42fe7c85-6395-4f9e-af22-2d5958454fb1
      resourceVersion: "1710"
      uid: 436bbc45-d3c8-4dcc-aa95-d4f0f5903c50
    spec:
      replicas: 1
      selector:
        matchLabels:
          k8s-app: metrics-server
          pod-template-hash: 54bf7cdd6
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: metrics-server
            pod-template-hash: 54bf7cdd6
        spec:
          containers:
          - args:
            - --cert-dir=/tmp
            - --secure-port=10250
            - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
            - --kubelet-use-node-status-port
            - --metric-resolution=15s
            image: registry.k8s.io/metrics-server/metrics-server:v0.7.2
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /livez
                port: https
                scheme: HTTPS
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            name: metrics-server
            ports:
            - containerPort: 10250
              name: https
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /readyz
                port: https
                scheme: HTTPS
              initialDelaySeconds: 20
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              requests:
                cpu: 100m
                memory: 200Mi
            securityContext:
              allowPrivilegeEscalation: false
              capabilities:
                drop:
                - ALL
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              seccompProfile:
                type: RuntimeDefault
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-dir
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          priorityClassName: system-cluster-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: metrics-server
          serviceAccountName: metrics-server
          terminationGracePeriodSeconds: 30
          volumes:
          - emptyDir: {}
            name: tmp-dir
    status:
      fullyLabeledReplicas: 1
      observedGeneration: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:32:40Z"
      generation: 1
      labels:
        k8s-app: metrics-server
        pod-template-hash: 84c5f94fbc
      name: metrics-server-84c5f94fbc
      namespace: kube-system
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: metrics-server
        uid: 42fe7c85-6395-4f9e-af22-2d5958454fb1
      resourceVersion: "1633"
      uid: 6a8906b7-5782-4506-af7f-5b89ff4d1a1c
    spec:
      replicas: 1
      selector:
        matchLabels:
          k8s-app: metrics-server
          pod-template-hash: 84c5f94fbc
      template:
        metadata:
          creationTimestamp: null
          labels:
            k8s-app: metrics-server
            pod-template-hash: 84c5f94fbc
          name: metrics-server
        spec:
          containers:
          - args:
            - --cert-dir=/tmp
            - --secure-port=4443
            - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
            - --kubelet-use-node-status-port
            - --metric-resolution=60s
            - --kubelet-insecure-tls
            image: registry.k8s.io/metrics-server/metrics-server:v0.7.2@sha256:ffcb2bf004d6aa0a17d90e0247cf94f2865c8901dcab4427034c341951c239f9
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /livez
                port: https
                scheme: HTTPS
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            name: metrics-server
            ports:
            - containerPort: 4443
              name: https
              protocol: TCP
            readinessProbe:
              failureThreshold: 3
              httpGet:
                path: /readyz
                port: https
                scheme: HTTPS
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 1
            resources:
              requests:
                cpu: 100m
                memory: 200Mi
            securityContext:
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-dir
          dnsPolicy: ClusterFirst
          priorityClassName: system-cluster-critical
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: metrics-server
          serviceAccountName: metrics-server
          terminationGracePeriodSeconds: 30
          volumes:
          - emptyDir: {}
            name: tmp-dir
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:32:36Z"
      generation: 1
      labels:
        k8s-app: dashboard-metrics-scraper
        pod-template-hash: c5db448b4
      name: dashboard-metrics-scraper-c5db448b4
      namespace: kubernetes-dashboard
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: dashboard-metrics-scraper
        uid: 2c41240a-19e2-4adf-98bd-5cc14ad6bb1c
      resourceVersion: "1608"
      uid: f07ba36d-bf66-42f1-8869-e6047187883d
    spec:
      replicas: 1
      selector:
        matchLabels:
          k8s-app: dashboard-metrics-scraper
          pod-template-hash: c5db448b4
      template:
        metadata:
          annotations:
            seccomp.security.alpha.kubernetes.io/pod: runtime/default
          creationTimestamp: null
          labels:
            k8s-app: dashboard-metrics-scraper
            pod-template-hash: c5db448b4
        spec:
          containers:
          - image: docker.io/kubernetesui/metrics-scraper:v1.0.8@sha256:76049887f07a0476dc93efc2d3569b9529bf982b22d29f356092ce206e98765c
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /
                port: 8000
                scheme: HTTP
              initialDelaySeconds: 30
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 30
            name: dashboard-metrics-scraper
            ports:
            - containerPort: 8000
              protocol: TCP
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsGroup: 2001
              runAsUser: 1001
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-volume
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: kubernetes-dashboard
          serviceAccountName: kubernetes-dashboard
          terminationGracePeriodSeconds: 30
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
          volumes:
          - emptyDir: {}
            name: tmp-volume
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-31T08:32:36Z"
      generation: 1
      labels:
        gcp-auth-skip-secret: "true"
        k8s-app: kubernetes-dashboard
        pod-template-hash: 695b96c756
      name: kubernetes-dashboard-695b96c756
      namespace: kubernetes-dashboard
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: kubernetes-dashboard
        uid: d96a0a26-d0b1-4ef1-ae5e-c01a32114593
      resourceVersion: "1591"
      uid: d272a318-2a13-4319-870e-ccfb4a95a999
    spec:
      replicas: 1
      selector:
        matchLabels:
          k8s-app: kubernetes-dashboard
          pod-template-hash: 695b96c756
      template:
        metadata:
          creationTimestamp: null
          labels:
            gcp-auth-skip-secret: "true"
            k8s-app: kubernetes-dashboard
            pod-template-hash: 695b96c756
        spec:
          containers:
          - args:
            - --namespace=kubernetes-dashboard
            - --enable-skip-login
            - --disable-settings-authorizer
            image: docker.io/kubernetesui/dashboard:v2.7.0@sha256:2e500d29e9d5f4a086b908eb8dfe7ecac57d2ab09d65b24f588b1d449841ef93
            imagePullPolicy: IfNotPresent
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /
                port: 9090
                scheme: HTTP
              initialDelaySeconds: 30
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 30
            name: kubernetes-dashboard
            ports:
            - containerPort: 9090
              protocol: TCP
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsGroup: 2001
              runAsUser: 1001
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /tmp
              name: tmp-volume
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          serviceAccount: kubernetes-dashboard
          serviceAccountName: kubernetes-dashboard
          terminationGracePeriodSeconds: 30
          tolerations:
          - effect: NoSchedule
            key: node-role.kubernetes.io/master
          volumes:
          - emptyDir: {}
            name: tmp-volume
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "2"
        deployment.kubernetes.io/max-replicas: "3"
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/name: model-server
        pod-template-hash: 664885b79f
      name: model-server-664885b79f
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: model-server
        uid: 9486f26c-8d5b-4ff8-8119-7aa4fefc14ef
      resourceVersion: "9876"
      uid: a0ac1e23-5dc3-4896-9e95-cc90b8f61806
    spec:
      replicas: 2
      selector:
        matchLabels:
          app.kubernetes.io/name: model-server
          pod-template-hash: 664885b79f
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: model-server
            pod-template-hash: 664885b79f
        spec:
          containers:
          - env:
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  key: hostname
                  name: redis-config-v1
            - name: REDIS_PORT
              valueFrom:
                configMapKeyRef:
                  key: port
                  name: redis-config-v1
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            - name: MODEL_NAME
              valueFrom:
                configMapKeyRef:
                  key: model_name
                  name: model-server-config-v1
            image: model-server:latest
            imagePullPolicy: Never
            name: model-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources:
              limits:
                cpu: "1"
                memory: 2Gi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 2
      fullyLabeledReplicas: 2
      observedGeneration: 1
      readyReplicas: 2
      replicas: 2
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/name: redis
        pod-template-hash: 69cf9b5676
        role: master
      name: redis-69cf9b5676
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: redis
        uid: 17fddc35-7ff5-4b50-a3ac-70d3029d154b
      resourceVersion: "9913"
      uid: e19f7630-f1e9-4091-bb32-bc61a0573fe6
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: redis
          pod-template-hash: 69cf9b5676
          role: master
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: redis
            pod-template-hash: 69cf9b5676
            role: master
        spec:
          containers:
          - args:
            - --requirepass
            - $(REDIS_PASSWORD)
            command:
            - redis-server
            env:
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            image: redis:7.4.1
            imagePullPolicy: IfNotPresent
            name: redis
            ports:
            - containerPort: 6379
              name: redis
              protocol: TCP
            resources:
              limits:
                cpu: 200m
                memory: 200Mi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
            - mountPath: /data
              name: redis-storage
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
          volumes:
          - name: redis-storage
            persistentVolumeClaim:
              claimName: redis-pvc
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "1"
        deployment.kubernetes.io/max-replicas: "2"
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/name: ui-server
        pod-template-hash: 76dc4d59d8
      name: ui-server-76dc4d59d8
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: ui-server
        uid: e71679b4-71f0-4bc5-9864-0b83b44c1202
      resourceVersion: "9881"
      uid: d293a584-ef0b-4420-9e1f-5896d07b3d7c
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: ui-server
          pod-template-hash: 76dc4d59d8
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: ui-server
            pod-template-hash: 76dc4d59d8
        spec:
          containers:
          - env:
            - name: WEB_SERVER_URL
              valueFrom:
                configMapKeyRef:
                  key: web_server_url
                  name: web-server-config-v1
            image: ui-server:latest
            imagePullPolicy: Never
            name: ui-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources:
              limits:
                cpu: 500m
                memory: 1536Mi
              requests:
                cpu: 250m
                memory: 768Mi
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 1
      fullyLabeledReplicas: 1
      observedGeneration: 1
      readyReplicas: 1
      replicas: 1
  - apiVersion: apps/v1
    kind: ReplicaSet
    metadata:
      annotations:
        deployment.kubernetes.io/desired-replicas: "2"
        deployment.kubernetes.io/max-replicas: "3"
        deployment.kubernetes.io/revision: "1"
        meta.helm.sh/release-name: fastapi-release-prod
        meta.helm.sh/release-namespace: production
      creationTimestamp: "2024-12-31T10:45:51Z"
      generation: 1
      labels:
        app.kubernetes.io/name: web-server
        pod-template-hash: 5c89b7fd87
      name: web-server-5c89b7fd87
      namespace: prod
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: web-server
        uid: 5b161fe0-3bd6-4748-b937-5956d10ee439
      resourceVersion: "9885"
      uid: 3eb4d83c-384e-49f5-ae40-5b99238cb221
    spec:
      replicas: 2
      selector:
        matchLabels:
          app.kubernetes.io/name: web-server
          pod-template-hash: 5c89b7fd87
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/name: web-server
            pod-template-hash: 5c89b7fd87
        spec:
          containers:
          - env:
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  key: hostname
                  name: redis-config-v1
            - name: REDIS_PORT
              valueFrom:
                configMapKeyRef:
                  key: port
                  name: redis-config-v1
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: db_password
                  name: redis-secret-v1
            - name: MODEL_SERVER_URL
              valueFrom:
                configMapKeyRef:
                  key: model_server_url
                  name: model-server-config-v1
            image: web-server:latest
            imagePullPolicy: Never
            name: web-server
            ports:
            - containerPort: 80
              name: http
              protocol: TCP
            resources: {}
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          restartPolicy: Always
          schedulerName: default-scheduler
          securityContext: {}
          terminationGracePeriodSeconds: 30
    status:
      availableReplicas: 2
      fullyLabeledReplicas: 2
      observedGeneration: 1
      readyReplicas: 2
      replicas: 2
  - apiVersion: batch/v1
    kind: Job
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"batch/v1","kind":"Job","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"admission-webhook","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-admission-create","namespace":"ingress-nginx"},"spec":{"template":{"metadata":{"labels":{"app.kubernetes.io/component":"admission-webhook","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-admission-create"},"spec":{"containers":[{"args":["create","--host=ingress-nginx-controller-admission,ingress-nginx-controller-admission.$(POD_NAMESPACE).svc","--namespace=$(POD_NAMESPACE)","--secret-name=ingress-nginx-admission"],"env":[{"name":"POD_NAMESPACE","valueFrom":{"fieldRef":{"fieldPath":"metadata.namespace"}}}],"image":"registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3","imagePullPolicy":"IfNotPresent","name":"create","securityContext":{"allowPrivilegeEscalation":false}}],"nodeSelector":{"kubernetes.io/os":"linux","minikube.k8s.io/primary":"true"},"restartPolicy":"OnFailure","securityContext":{"runAsNonRoot":true,"runAsUser":2000},"serviceAccountName":"ingress-nginx-admission"}}}}
      creationTimestamp: "2024-12-31T08:31:31Z"
      generation: 1
      labels:
        app.kubernetes.io/component: admission-webhook
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      name: ingress-nginx-admission-create
      namespace: ingress-nginx
      resourceVersion: "1404"
      uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
    spec:
      backoffLimit: 6
      completionMode: NonIndexed
      completions: 1
      manualSelector: false
      parallelism: 1
      podReplacementPolicy: TerminatingOrFailed
      selector:
        matchLabels:
          batch.kubernetes.io/controller-uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
      suspend: false
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/component: admission-webhook
            app.kubernetes.io/instance: ingress-nginx
            app.kubernetes.io/name: ingress-nginx
            batch.kubernetes.io/controller-uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
            batch.kubernetes.io/job-name: ingress-nginx-admission-create
            controller-uid: 56f8fa93-3475-43ef-a0b0-1c6812d89f14
            job-name: ingress-nginx-admission-create
          name: ingress-nginx-admission-create
        spec:
          containers:
          - args:
            - create
            - --host=ingress-nginx-controller-admission,ingress-nginx-controller-admission.$(POD_NAMESPACE).svc
            - --namespace=$(POD_NAMESPACE)
            - --secret-name=ingress-nginx-admission
            env:
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
            image: registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
            imagePullPolicy: IfNotPresent
            name: create
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
            minikube.k8s.io/primary: "true"
          restartPolicy: OnFailure
          schedulerName: default-scheduler
          securityContext:
            runAsNonRoot: true
            runAsUser: 2000
          serviceAccount: ingress-nginx-admission
          serviceAccountName: ingress-nginx-admission
          terminationGracePeriodSeconds: 30
    status:
      completionTime: "2024-12-31T08:31:46Z"
      conditions:
      - lastProbeTime: "2024-12-31T08:31:46Z"
        lastTransitionTime: "2024-12-31T08:31:46Z"
        message: Reached expected number of succeeded pods
        reason: CompletionsReached
        status: "True"
        type: SuccessCriteriaMet
      - lastProbeTime: "2024-12-31T08:31:46Z"
        lastTransitionTime: "2024-12-31T08:31:46Z"
        message: Reached expected number of succeeded pods
        reason: CompletionsReached
        status: "True"
        type: Complete
      ready: 0
      startTime: "2024-12-31T08:31:31Z"
      succeeded: 1
      terminating: 0
      uncountedTerminatedPods: {}
  - apiVersion: batch/v1
    kind: Job
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"batch/v1","kind":"Job","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"admission-webhook","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-admission-patch","namespace":"ingress-nginx"},"spec":{"template":{"metadata":{"labels":{"app.kubernetes.io/component":"admission-webhook","app.kubernetes.io/instance":"ingress-nginx","app.kubernetes.io/name":"ingress-nginx"},"name":"ingress-nginx-admission-patch"},"spec":{"containers":[{"args":["patch","--webhook-name=ingress-nginx-admission","--namespace=$(POD_NAMESPACE)","--patch-mutating=false","--secret-name=ingress-nginx-admission","--patch-failure-policy=Fail"],"env":[{"name":"POD_NAMESPACE","valueFrom":{"fieldRef":{"fieldPath":"metadata.namespace"}}}],"image":"registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3","imagePullPolicy":"IfNotPresent","name":"patch","securityContext":{"allowPrivilegeEscalation":false}}],"nodeSelector":{"kubernetes.io/os":"linux","minikube.k8s.io/primary":"true"},"restartPolicy":"OnFailure","securityContext":{"runAsNonRoot":true,"runAsUser":2000},"serviceAccountName":"ingress-nginx-admission"}}}}
      creationTimestamp: "2024-12-31T08:31:31Z"
      generation: 1
      labels:
        app.kubernetes.io/component: admission-webhook
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
      name: ingress-nginx-admission-patch
      namespace: ingress-nginx
      resourceVersion: "1400"
      uid: 572547be-9593-4e67-83c8-b9c09600be77
    spec:
      backoffLimit: 6
      completionMode: NonIndexed
      completions: 1
      manualSelector: false
      parallelism: 1
      podReplacementPolicy: TerminatingOrFailed
      selector:
        matchLabels:
          batch.kubernetes.io/controller-uid: 572547be-9593-4e67-83c8-b9c09600be77
      suspend: false
      template:
        metadata:
          creationTimestamp: null
          labels:
            app.kubernetes.io/component: admission-webhook
            app.kubernetes.io/instance: ingress-nginx
            app.kubernetes.io/name: ingress-nginx
            batch.kubernetes.io/controller-uid: 572547be-9593-4e67-83c8-b9c09600be77
            batch.kubernetes.io/job-name: ingress-nginx-admission-patch
            controller-uid: 572547be-9593-4e67-83c8-b9c09600be77
            job-name: ingress-nginx-admission-patch
          name: ingress-nginx-admission-patch
        spec:
          containers:
          - args:
            - patch
            - --webhook-name=ingress-nginx-admission
            - --namespace=$(POD_NAMESPACE)
            - --patch-mutating=false
            - --secret-name=ingress-nginx-admission
            - --patch-failure-policy=Fail
            env:
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: metadata.namespace
            image: registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.4.3@sha256:a320a50cc91bd15fd2d6fa6de58bd98c1bd64b9a6f926ce23a600d87043455a3
            imagePullPolicy: IfNotPresent
            name: patch
            resources: {}
            securityContext:
              allowPrivilegeEscalation: false
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
          dnsPolicy: ClusterFirst
          nodeSelector:
            kubernetes.io/os: linux
            minikube.k8s.io/primary: "true"
          restartPolicy: OnFailure
          schedulerName: default-scheduler
          securityContext:
            runAsNonRoot: true
            runAsUser: 2000
          serviceAccount: ingress-nginx-admission
          serviceAccountName: ingress-nginx-admission
          terminationGracePeriodSeconds: 30
    status:
      completionTime: "2024-12-31T08:31:46Z"
      conditions:
      - lastProbeTime: "2024-12-31T08:31:46Z"
        lastTransitionTime: "2024-12-31T08:31:46Z"
        message: Reached expected number of succeeded pods
        reason: CompletionsReached
        status: "True"
        type: SuccessCriteriaMet
      - lastProbeTime: "2024-12-31T08:31:46Z"
        lastTransitionTime: "2024-12-31T08:31:46Z"
        message: Reached expected number of succeeded pods
        reason: CompletionsReached
        status: "True"
        type: Complete
      ready: 0
      startTime: "2024-12-31T08:31:31Z"
      succeeded: 1
      terminating: 0
      uncountedTerminatedPods: {}
  kind: List
  metadata:
    resourceVersion: ""
  ```