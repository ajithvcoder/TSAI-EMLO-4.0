### Report for Assignment

- Output of the following in a .md file in your repository

  ```
  kubectl describe <your_deployment>
  kubectl describe <your_pod>
  kubectl describe <your_ingress>
  kubectl top pod
  kubectl top node
  kubectl get all -o yaml
  ```

**Get all infos**

- `kubectl get all`

  ```
  NAME                                        READY   STATUS    RESTARTS   AGE
  pod/classifer-deployment-867c948fc9-55xvd   1/1     Running   0          9m14s
  pod/classifer-deployment-867c948fc9-ccc9h   1/1     Running   0          9m14s

  NAME                         TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
  service/classifier-service   ClusterIP   10.97.45.45   <none>        80/TCP    9m14s
  service/kubernetes           ClusterIP   10.96.0.1     <none>        443/TCP   42m

  NAME                                   READY   UP-TO-DATE   AVAILABLE   AGE
  deployment.apps/classifer-deployment   2/2     2            2           9m14s

  NAME                                              DESIRED   CURRENT   READY   AGE
  replicaset.apps/classifer-deployment-867c948fc9   2         2         2       9m14s
  ```

-----------------------------


**Describe Deployment**

- `kubectl describe deployment.apps/classifer-deployment`

  ```
  Name:                   classifer-deployment
  Namespace:              default
  CreationTimestamp:      Thu, 26 Dec 2024 06:04:04 +0000
  Labels:                 app=classifier
  Annotations:            deployment.kubernetes.io/revision: 1
  Selector:               app=classifier
  Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
  StrategyType:           RollingUpdate
  MinReadySeconds:        0
  RollingUpdateStrategy:  25% max unavailable, 25% max surge
  Pod Template:
    Labels:  app=classifier
    Containers:
    classifier:
      Image:         fastapi-classifier-k8s:latest
      Port:          8000/TCP
      Host Port:     0/TCP
      Environment:   <none>
      Mounts:        <none>
    Volumes:         <none>
    Node-Selectors:  <none>
    Tolerations:     <none>
  Conditions:
    Type           Status  Reason
    ----           ------  ------
    Available      True    MinimumReplicasAvailable
    Progressing    True    NewReplicaSetAvailable
  OldReplicaSets:  <none>
  NewReplicaSet:   classifer-deployment-867c948fc9 (2/2 replicas created)
  Events:
    Type    Reason             Age   From                   Message
    ----    ------             ----  ----                   -------
    Normal  ScalingReplicaSet  10m   deployment-controller  Scaled up replica set classifer-deployment-867c948fc9 to 2
  ```

-----------------------------

**Describe Pod**

- `kubectl describe pod/classifer-deployment-867c948fc9-55xvd`

  ```
  Name:             classifer-deployment-867c948fc9-55xvd
  Namespace:        default
  Priority:         0
  Service Account:  default
  Node:             minikube/192.168.49.2
  Start Time:       Thu, 26 Dec 2024 06:04:04 +0000
  Labels:           app=classifier
                    pod-template-hash=867c948fc9
  Annotations:      <none>
  Status:           Running
  IP:               10.244.0.3
  IPs:
    IP:           10.244.0.3
  Controlled By:  ReplicaSet/classifer-deployment-867c948fc9
  Containers:
    classifier:
      Container ID:   docker://69a994f5d7200d85cb7d1cc881e040bc9f6747f6199ea7a8915c37c2854b792d
      Image:          fastapi-classifier-k8s:latest
      Image ID:       docker://sha256:22db205d4bece01938b55a43af9db867d81f8054ae5baf3472120db791f47628
      Port:           8000/TCP
      Host Port:      0/TCP
      State:          Running
        Started:      Thu, 26 Dec 2024 06:04:05 +0000
      Ready:          True
      Restart Count:  0
      Environment:    <none>
      Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-d8wwp (ro)
  Conditions:
    Type                        Status
    PodReadyToStartContainers   True 
    Initialized                 True 
    Ready                       True 
    ContainersReady             True 
    PodScheduled                True 
  Volumes:
    kube-api-access-d8wwp:
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
    Normal  Scheduled  11m   default-scheduler  Successfully assigned default/classifer-deployment-867c948fc9-55xvd to minikube
    Normal  Pulled     11m   kubelet            Container image "fastapi-classifier-k8s:latest" already present on machine
    Normal  Created    11m   kubelet            Created container classifier
    Normal  Started    11m   kubelet            Started container classifier
  ```

------------------------------------

**Describe Ingress**

- `kubectl describe ingress classifier-ingress `

  ```
  Name:             classifier-ingress
  Labels:           <none>
  Namespace:        default
  Address:          
  Ingress Class:    <none>
  Default backend:  <default>
  Rules:
    Host           Path  Backends
    ----           ----  --------
    ajith.fastapi  
                  /   classifier-service:80 (10.244.0.3:8000,10.244.0.4:8000)
  Annotations:     nginx.ingress.kubernetes.io/affinity: cookie
                  nginx.ingress.kubernetes.io/affinity-mode: balanced
                  nginx.ingress.kubernetes.io/session-cookie-expires: 172800
                  nginx.ingress.kubernetes.io/session-cookie-max-age: 172800
                  nginx.ingress.kubernetes.io/session-cookie-name: INGRESSCOOKIE
  Events:          <none>
  ```

--------------------------------

**Pod Infos**

- `kubectl top pod`

  ```
  NAME                                    CPU(cores)   MEMORY(bytes)   
  classifer-deployment-867c948fc9-55xvd   2m           168Mi           
  classifer-deployment-867c948fc9-ccc9h   2m           145Mi     
  ```

-------------------------------

**Node Infos**

` kubectl top node `

```
NAME       CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%   
minikube   123m         6%     1788Mi          22%       
```

------

**Execution Infos**

- `kubectl get all -o yaml`

  ```
  apiVersion: v1
  items:
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-26T06:04:04Z"
      generateName: classifer-deployment-867c948fc9-
      labels:
        app: classifier
        pod-template-hash: 867c948fc9
      name: classifer-deployment-867c948fc9-55xvd
      namespace: default
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: classifer-deployment-867c948fc9
        uid: aaa1ec55-ca7b-4ce6-92e5-cb91a801e835
      resourceVersion: "2025"
      uid: 3b21a042-24b1-4753-8574-6370dd2cd022
    spec:
      containers:
      - image: fastapi-classifier-k8s:latest
        imagePullPolicy: Never
        name: classifier
        ports:
        - containerPort: 8000
          protocol: TCP
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-d8wwp
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
      - name: kube-api-access-d8wwp
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
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:04Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:04Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://69a994f5d7200d85cb7d1cc881e040bc9f6747f6199ea7a8915c37c2854b792d
        image: fastapi-classifier-k8s:latest
        imageID: docker://sha256:22db205d4bece01938b55a43af9db867d81f8054ae5baf3472120db791f47628
        lastState: {}
        name: classifier
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-26T06:04:05Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-d8wwp
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.3
      podIPs:
      - ip: 10.244.0.3
      qosClass: BestEffort
      startTime: "2024-12-26T06:04:04Z"
  - apiVersion: v1
    kind: Pod
    metadata:
      creationTimestamp: "2024-12-26T06:04:04Z"
      generateName: classifer-deployment-867c948fc9-
      labels:
        app: classifier
        pod-template-hash: 867c948fc9
      name: classifer-deployment-867c948fc9-ccc9h
      namespace: default
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: ReplicaSet
        name: classifer-deployment-867c948fc9
        uid: aaa1ec55-ca7b-4ce6-92e5-cb91a801e835
      resourceVersion: "2030"
      uid: b09c4eea-b897-431e-bbe7-b9c9304681e8
    spec:
      containers:
      - image: fastapi-classifier-k8s:latest
        imagePullPolicy: Never
        name: classifier
        ports:
        - containerPort: 8000
          protocol: TCP
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-dc2l8
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
      - name: kube-api-access-dc2l8
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
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: PodReadyToStartContainers
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:04Z"
        status: "True"
        type: Initialized
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: Ready
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:06Z"
        status: "True"
        type: ContainersReady
      - lastProbeTime: null
        lastTransitionTime: "2024-12-26T06:04:04Z"
        status: "True"
        type: PodScheduled
      containerStatuses:
      - containerID: docker://95aafc54c30d0156fa6350f899fdfa6b50935677053348525f73c4835066d08e
        image: fastapi-classifier-k8s:latest
        imageID: docker://sha256:22db205d4bece01938b55a43af9db867d81f8054ae5baf3472120db791f47628
        lastState: {}
        name: classifier
        ready: true
        restartCount: 0
        started: true
        state:
          running:
            startedAt: "2024-12-26T06:04:05Z"
        volumeMounts:
        - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
          name: kube-api-access-dc2l8
          readOnly: true
          recursiveReadOnly: Disabled
      hostIP: 192.168.49.2
      hostIPs:
      - ip: 192.168.49.2
      phase: Running
      podIP: 10.244.0.4
      podIPs:
      - ip: 10.244.0.4
      qosClass: BestEffort
      startTime: "2024-12-26T06:04:04Z"
  - apiVersion: v1
    kind: Service
    metadata:
      annotations:
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"name":"classifier-service","namespace":"default"},"spec":{"ports":[{"port":80,"protocol":"TCP","targetPort":8000}],"selector":{"app":"classifier"}}}
      creationTimestamp: "2024-12-26T06:04:04Z"
      name: classifier-service
      namespace: default
      resourceVersion: "2010"
      uid: a07631e5-c65f-4dbb-88fd-106a3de68040
    spec:
      clusterIP: 10.97.45.45
      clusterIPs:
      - 10.97.45.45
      internalTrafficPolicy: Cluster
      ipFamilies:
      - IPv4
      ipFamilyPolicy: SingleStack
      ports:
      - port: 80
        protocol: TCP
        targetPort: 8000
      selector:
        app: classifier
      sessionAffinity: None
      type: ClusterIP
    status:
      loadBalancer: {}
  - apiVersion: v1
    kind: Service
    metadata:
      creationTimestamp: "2024-12-26T05:30:48Z"
      labels:
        component: apiserver
        provider: kubernetes
      name: kubernetes
      namespace: default
      resourceVersion: "231"
      uid: 212c126b-1668-4c0d-a535-f9a7ee895491
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
  - apiVersion: apps/v1
    kind: Deployment
    metadata:
      annotations:
        deployment.kubernetes.io/revision: "1"
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"app":"classifier"},"name":"classifer-deployment","namespace":"default"},"spec":{"replicas":2,"selector":{"matchLabels":{"app":"classifier"}},"template":{"metadata":{"labels":{"app":"classifier"}},"spec":{"containers":[{"image":"fastapi-classifier-k8s:latest","imagePullPolicy":"Never","name":"classifier","ports":[{"containerPort":8000}]}]}}}}
      creationTimestamp: "2024-12-26T06:04:04Z"
      generation: 1
      labels:
        app: classifier
      name: classifer-deployment
      namespace: default
      resourceVersion: "2034"
      uid: 06de2591-34d3-492f-addc-520bb3ba2f5e
    spec:
      progressDeadlineSeconds: 600
      replicas: 2
      revisionHistoryLimit: 10
      selector:
        matchLabels:
          app: classifier
      strategy:
        rollingUpdate:
          maxSurge: 25%
          maxUnavailable: 25%
        type: RollingUpdate
      template:
        metadata:
          creationTimestamp: null
          labels:
            app: classifier
        spec:
          containers:
          - image: fastapi-classifier-k8s:latest
            imagePullPolicy: Never
            name: classifier
            ports:
            - containerPort: 8000
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
      - lastTransitionTime: "2024-12-26T06:04:06Z"
        lastUpdateTime: "2024-12-26T06:04:06Z"
        message: Deployment has minimum availability.
        reason: MinimumReplicasAvailable
        status: "True"
        type: Available
      - lastTransitionTime: "2024-12-26T06:04:04Z"
        lastUpdateTime: "2024-12-26T06:04:06Z"
        message: ReplicaSet "classifer-deployment-867c948fc9" has successfully progressed.
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
        deployment.kubernetes.io/desired-replicas: "2"
        deployment.kubernetes.io/max-replicas: "3"
        deployment.kubernetes.io/revision: "1"
      creationTimestamp: "2024-12-26T06:04:04Z"
      generation: 1
      labels:
        app: classifier
        pod-template-hash: 867c948fc9
      name: classifer-deployment-867c948fc9
      namespace: default
      ownerReferences:
      - apiVersion: apps/v1
        blockOwnerDeletion: true
        controller: true
        kind: Deployment
        name: classifer-deployment
        uid: 06de2591-34d3-492f-addc-520bb3ba2f5e
      resourceVersion: "2033"
      uid: aaa1ec55-ca7b-4ce6-92e5-cb91a801e835
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: classifier
          pod-template-hash: 867c948fc9
      template:
        metadata:
          creationTimestamp: null
          labels:
            app: classifier
            pod-template-hash: 867c948fc9
        spec:
          containers:
          - image: fastapi-classifier-k8s:latest
            imagePullPolicy: Never
            name: classifier
            ports:
            - containerPort: 8000
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
  kind: List
  metadata:
    resourceVersion: ""
  ```
-----------------------------