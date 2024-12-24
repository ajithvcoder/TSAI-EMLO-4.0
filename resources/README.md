## Resources

Has important links, tools, blogs or concepts that i learnt during EMLO-4.0

### Ready Docker usage platforms

- gitpod.io
- github code spaces

### Low cost cloud GPUs

- jarvislabs
- runpod
- papersapce
- AWS spot instances

### Developing GPU scheduling in MLOPS

- Use NVIDIA Time slicing for Low end GPUS
- Use NVIDIA MIG for high end GPUs

### AWS Info

- In aws resouces are region specific and high end gpus are available only in US region

**Instances**

```
- `T3a` -> 'a' means amd
- `M8g` -> arm processor  and g1 are less charged
- `M7i` -> 'i' means intel
- `t3.micro`, `t3.nano` is free tier
- `p4de.24xlarge` - 70 billion model training -> And will be allocated only on request
```

- EBS instance are costly as input and output operations ex writing a model - `p4de.large` instance
- L4 successor of T4 GPUs
- Minimum instance - `t3a.medium` -> 2vCpu -> 4GB mem -> mnist traninig
- Highest instance - `p5e.48xlarge` -> 8 H200 are the biggest you can get 1TB of ram
- Use docker images with cuda installations
- if you dont know which instance ot use go for T3 instances first also T3 instances has a traffic limit eg: 5Gbps(Network erformance) EBS instances
- **Accelearated computing** -> fully connected layers works good (`Trn1`) accerator are not gpus. Good cost optimization can be achieved with this for inference purposes
- **Spot instance** - 10% of cost and use it with peresistent storage and **make sure you cancel the spot request after usage**

**Networking**
- only through vpc internet is accessed any thing inbound or out bound and only allow certain ports that u want
- https: 443, http: 80
- In same vpc we can use private ip to connect to another private ip
- spot fleet request -> use load balancer 

**AWS and local VScode connection**

- [Instructions](https://github.com/ajithvcoder/emlo4-session-09-ajithvcoder?tab=readme-ov-file#development-command-and-debug-commands)

**Backup and storage**

- EBS snapshot -> enable backup
- EBS helps to reduce the volume storage

**S3 instance**
- S3 
- s3 standard
- s3 glacier - for long time usage and cost is less without retrival and then

**AWS configure**
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION needs to be provided
- `aws s3 ls` - test command for connection

**Create a AMI**

```
# Configure with your accesskey and secret
aws configure

# From you own instance where you are fetches the instance-id and pushes the AMI to private ami location
aws ec2 create-image \
    --instance-id $(curl -s http://169.254.169.254/latest/meta-data/instance-id) \
    --name "Session-09-ami-Nov-19-1" \
    --description "AMI created programmatically from this instance" \
    --no-reboot

# You would get a ami-id like this ami-0af5900df6f0bfaf4
```

**Get instance information**

```
echo "Checking instance information..."
# Check if we're on EC2
# TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
# curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type
```

**Using aws ec2 for ci/cd without github tools**

- [Reference](https://github.com/ajithvcoder/emlo4-session-10-ajithvcoder/blob/main/.github/workflows/ec2-pipeline-end-to-end.yml#L64)

**Efficient AWS Instance Usage**

- if request are very sparse then go with lambda 10 request per day like that .. take small gpu - smallest is t4 16 GB

**AWS Lambda**
- Even driven execution -> output shoud be sent to somewhere -> to s3 and cost 1 unit
- stateless - s3, dynamodb, rds(traditional databse) can be used to save
- scaling is done by AWS itself -> lambda is 1 million request free every month
- lambda - downside - no gpus -> serveless gpu with lambda like frame work some other providers have
- application is stateless we can use lambda
- request is not continuously being hit, if so lambda
- Decide between EC2 and lambda - ec2 instance exact reeuqest with lambda should be measured for performance and cost decision
- Monitoring - sns in aws is quick
- Limit: 16 GB ram in lambda is limit, 15 minutes limit for single request
- No request the lambda goes to cold state, or first start is cold start
second request good time. so first request cold start is a drawback.
- cpu models work good, then if cold start if its a issue dont use aws lambda
- aws lambda cost after 1 million -> x86 -> 2M per 1M request so cost is very less
- larger docker size will take large time to start -> aws lambda so download the model from s3 and use in lambda
- load balancer is handled by aws lambda so its a big adavantage than ec2
- SemLock not possible in aws lambda but multiprocessing is possible
https://ruan.dev/blog/2019/02/19/parallel-processing-on-aws-lambda-with-python-using-multiprocessing
- lambda from image  -> 1.from ecr u can select the image to be deployed -> 2. or aws cdk - mange using code all start, stop .. its like terraform managing infrastructure. -> Infrastructure as code
- lambda-adapter is needed in docker file -> aws lambda event format .. it should be same like that so
that we are using lambda-adapter.. so that it adapts your request to aws-adapter. examples : https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi
- one lambda can be connected to another, then "add trigger" in lambda
- aws lambda has hot start time limit -> cloud watch can be used to see logs for aws lambda -> cloudwatch/log groups / lambda function name or monitoring

**AWS CDK**
- its like terraform managing infrastructure. -> Infrastructure as code

**AWS API gateway**
- timeout of 29 seconds for this service but lambda gives 15 minutes timeout max

**Info**
- good practise-> select a user and attach poliocies .. else some times it may create more resources
- Use cloud formation to check what resouces are created and manually destroy all resources

**TODO**
- In cml why only cuda 11.2 came and didn't 12.1 come ? what tool supports ec2+sport instace trigger

### Linux Debug commands

- `ctop` -> container utilzation
- `tmux` session in terminal
- `htop`
- `nv-top` - gpu and cpu utilzation
- `realpath .`
- `realpath temp` gives absolute path
- `rsync` -> for parallel processing for copying
- `gpustate -cp` for checking gpu utilization
- `ps -ef | grep python`
- `vim` can be used for faster debugging in remote
- `ssh instanceurl.com python server.py` -> Ec2 login
- `nvitop`
- `pip install .` -> for installing packages -> need pyproject.toml file

### GPU resouce calculator

- [GPU_poor](https://rahulschand.github.io/gpu_poor/)
- Either u get 1,2,4,8 gpus
- inference -> modelsize*10%
- model training -> model*2.5 times

- 70 Billion parameters - full floating point
- 70,000,000,000(parameters for model)*32(floating point)/8(Bits to byte conversion)/1024/1024/1024 = 260 GB 
- 1024/1024/1024 -> Bits to byte conversion -> GB conversion
- half bit - use 16 to 32
- llama3.1-1B-model - best for budget model - 1,000,000,000(parameters for model)*16(floating point)/8(Bits to byte conversion)/1024/1024/1024 = 1.1 GB
- calculate size for model inference -> multiply by 2.5 i.e required for training -> USE gpu size calculator then batch size and prompt length also matters

### Docker GPU utilization
- To connect docker with host gpu you might need below commands
- For docker run - `--gpus all`
- For docker compose - [docker gpu docs](https://docs.docker.com/compose/how-tos/gpu-support/#example-of-a-compose-file-for-running-a-service-with-access-to-1-gpu-device)

*Test*
- Inside docker run below you need to get `True`
    ```
    import torch
    torch.cuda.is_available()
    ```

### ML Serving framework

(as of Dec, 2024)
- litServe, vLLM, fastapi, torchserve, ollama
- Use vLLM to serve to large number of users and with batch serving
- litserve, vllm are good for llm serving as they have additional caching and optimization mechanisms
- vllm and ollama are not that customizable
- torch serve in java written
- torch servce to litservce migration is possible and lit serve very new
- lit serve each gpu model setup is called once
- good habit to see GPU utilization when model is deployed but there is no monitoring function for it

**Optimizations**

- if batch size is multiple of 2 then it will be faster or power of 2
eg: 1526 is faster than 1527
- higher gpu usage increase batch size
- while doing batch processing if the cpu is 8 core and if we have 64 batch size then there will be more context switching
- what is difference between workers and thread 
- how to use threds with async
- pqdm for jobs and it handles concurrently
- if 64 request are sent from client to server and server cant handle it parallely it will convert it to sequentialy
- llama3.1-1B-model - best for budget model - 1,000,000,000(parameters for model)*16(floating point)/8(Bits to byte conversion)/1024/1024/1024 = 1.1 GB
- question in a session-09 why does in api server there is low through put than the baseline model
- `fastapi` ->  asgi -> completely async functions
- `torchserve` - healvily used in production
- fast api faster than flask, 

**concurrency**
- request comes -> read file i/O operation eg: getting file from S3 while the i/O is happening another one can happen so that is called concurrency instead of cpu being idle another ccan be done. 
- types -> parallel, concurrent, concurrent and parallel

- IN theory -> cpu bound task or i/o bound task (todo)

**torchserve**
- swapping a model, i.e two version running and stop one and start new model again
- torch serve+promethus+grapfana, torch serve can register multiple models
- Torch serve can deploy cpu based also, even without model also it can do
- model packaging, deploy config, monitoring

### ML and LLMs Optimizations
- Use vLLM to serve to large number of users and with batch serving
- `https://platform.openai.com/tokenizer` - measure token -> bytepair encoding tokenizing and `google` uses word by word
- In python only during run time the code is compiled so it make totaly solow and its a scripting language so pytorch came up with torchscript first time the cuda kernel are compiled at first so it makes first inference slow and make warm up
- `fastapi` -> completely async functions

**torchscript**

- torchscipt -> own compilation and then makes code faster -> Also no need Model class -> just .pt file is enough
can use in cpp, only in inference it can taken, in browser also , in android also

- torchscirpt traced model are stored in .pt -> No need any class or instance creation for model
other pytorch models are stored in .pth

- torchscripts -> saves all the modules in .pt file i.e is in .forward() , only custom layers cant be saved -> 10 to 20% faster
- Refer *Multi GPU Training* in *Pytorch lighting*

**Other conversions**
- onnx runtime -> represention of model with weights in a single file
- Fast api -> nn.Transformer -> has optimization for pytorch that supports a all kinds of gpu (todo: check if nn.Transformer is related to fastapi )
- tensor-rt -> heavily optimized -> tensor-rt model optimizer takes care of it


### Pytorch lighting

**Integrations**

- Hydra
- optuna
- lora
- peft
- comet, mlflow
- lighting-gradio integration
*Multi GPU Training*
- pytorch multigpu training -- `ddp` - averaged out by master node
- DDP -> averaged out by master node
- in pytorch lighting -> strategy - ddp, -> 100 million param -> create 8 copies

    ```
    for each GPU
    num_nodes- -> 8 copies run in 2 gpus
    1 master note and other nodes -> each will get copy forward pass each of the gpua nd node, gradient are computed and averaged by master node consolidates
    - cons: a very large model we can train
    DDP-> only for training
    ```

 - FSDP

    ```
    -> splites the model to 8 parts for 8 gpu and it can train and consolidates
    FSDP -> only for training
    https://pytorch.org/tutorials/_images/fsdp_workflow.png

    sharding method used -> check what it is
    ```

### CI CD Pipeline

- self-hosted-runner - commands from runner should be ran in aws ec2
- github-hosted-runner
- auto start and auto stop ec2 spot instance using a custom ami which we are giving

### UI Developement and restapi

**Gradio**
- for 3D, 2d , text or anything -(backend python , front end swelt)
- flagging - for detecting false images or anomaly
- share=True -> share from one laptop to another
- lighting-gradio integration
- gr.Model3D -> 3D rendering
- live inferencing -> `WebRTC` component in gradio
- gradio -> SimpleCSVlogger(), it locks the log file and writes to csvlogger to avoid race condition
- fast api faster than flask
- wsgi-> synchronous copies 
- unvicorn, wsgi, nginx
- In api, we cant do batching only in litserve we can do batching
- CORS
    ```
    if in origins = ["*"] and if one domain name is calling another domain name then its not possible u need to add app.add_middleware and origins. CORS error we need to add above
    ```
- fastapi+jinja template
- In fastapi - /docs gives all end points, /redoc gives another some documentation

**Others**
- `pyodide` 
- `fasthtml` 

### Hugging face and ML Modles

- `flux.11-dev` - hf becoming popular
- `https://huggingface.co/spaces/enzostvs/zero-gpu-spaces`
- All huggingface model deployed is CPU and its free tier
- End to end pipeline to deploy model -> after training just create torchscript and deploy

### Misellaneous

- Major models - `https://aiworld.eu/embed/model/model/treemap`
- `photon` by luma - use image to convert to video
- `https://x.com/fofrAI`
- FLUX- best model - 24GB ram needed
- replica.co
- `Large models/ trasnformers` good for segmentation
- `internimage` - for segmentation
- `eva` - for segmentation
- `depth_pro` - apple's model - good for dept estimation -(more than MIDAs)
- `briyal/RMBG-2.0`
- Diffusion meaning - random image and keeps upgrading so diffusion
- prompting reference - `https://stability.ai/learning-hub/stable-diffusion-3-5-prompt-guide` - 
- stable diffusion - IMG_302.CR2 creates a photo realistic image because it was trained thinking that `.CR2` is a DSLR image


### Tools

| S.NO | Purpose                   | Package Name         |
|------|---------------------------|----------------------|
| 1    | Argument/Config management        | [Hydra](https://hydra.cc/docs/intro/)                |
| 2    | Logging                    | [aim](https://github.com/aimhubio/aim), [Comet](https://github.com/Unbabel/COMET), [MLflow](https://github.com/mlflow/mlflow)   |
| 3    | Data versioning            | [DVC with Cloud (GCS)](https://github.com/iterative/dvc) |
| 4    | Markdown file generation   | [Tabulate](https://github.com/astanin/python-tabulate)             |
| 5    | Code formatter               | [Black](https://github.com/psf/black)                |
| 6    | Google Drive data download | [gdown](https://github.com/wkentaro/gdown)                |
| 7    | GitHub Actions commenting  | [cml](https://github.com/iterative/cml)                  |
| 8    | Unit testing  python            | [Pytest](https://github.com/pytest-dev/pytest)               |
| 9    | Test coverage reporting    | [Coverage](https://github.com/nedbat/coveragepy)             |
| 10   | AI code assistance         | [Cursor](https://www.cursor.com/)               |
| 11   | Hyper parameter Optimization | [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/)  |
| 12   | Multi run parallel           | [Joblib](https://hydra.cc/docs/plugins/joblib_launcher/) |
| 13   | Run Github actions locally   | [act](https://github.com/nektos/act)               |
| 14   | GPU Requirement calculator   | [GPU_poor](https://rahulschand.github.io/gpu_poor/) |
| 15   | Reduce model size of FC layers | [torch compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)  |
| 16   | Quantatization | [torch ao](https://github.com/pytorch/ao) |
| 17   | aws cli | [awscli](https://github.com/aws/aws-cli) |
| 18   | env variable | [rootutils](https://github.com/ashleve/rootutils)  |
| 19   | sets basic root | [.projectroot](https://jolars.github.io/ProjectRoot.jl/stable/) |
| 20   | tools for serving machine learning models(genric) | [litServe](https://github.com/Lightning-AI/LitServe) |
| 21   | tools for serving machine learning models(LLM specific) |[vLLM](https://github.com/vllm-project/vllm) |
| 22   | tools for serving machine learning models(genric) | [fastapi](https://github.com/fastapi/fastapi) |
| 23   | tools for serving machine learning models(genric) | [torchserve](https://github.com/pytorch/serve) |
| 24   | tools for serving machine learning models(LLM specific) | [ollama](https://github.com/ollama/ollama) |
| 25   | load testing | [locust](https://locust.io/)  |
| 26   | Handle concurrancy | [pqdm](https://github.com/niedakh/pqdm) |
| 27   | llm optimization | [peft](https://github.com/huggingface/peft)  |
| 28   | llm optimization | [trl](https://github.com/huggingface/trl)  |
| 29   | llm optimization | [lora](https://github.com/microsoft/LoRA)  |
| 30   | AWS alternative to GitHub | [codecommit](https://aws.amazon.com/codecommit/) |
| 31   | AWS alternative to GitHub actions | [codepipeline](https://aws.amazon.com/codepipeline/) |
| 32   | LORA for vit | [timm-vit-lora](https://github.com/JamesQFreeman/LoRA-ViT) |
| 33   | testing api | [postman](https://www.postman.com/) |
| 34   | Quick ML demo with UI | [gradio](https://github.com/gradio-app/gradio)  |
| 35   | Quick ML demo with UI (python based) | [pyodide](https://github.com/pyodide/pyodide) |
| 36   | Quick ML demo with UI | [fasthtml](https://github.com/AnswerDotAI/fasthtml)  |
| 37   | Quick ML demo with UI | [streamlit](https://github.com/streamlit/streamlit) |
| 38   | Optimized model | [torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) |
| 39   | represention of model with weights in a single file | [onnx runtime](https://github.com/microsoft/onnxruntime) |
| 40   |  web servers | [unvicorn](https://www.uvicorn.org/) |
| 41   |  web servers python syc | [wsgi](https://wsgi.readthedocs.io/en/latest/) |
| 42   |  web servers python asyc | [asgi](https://asgi.readthedocs.io/en/latest/) |
| 43   |  web servers - front-end reverse proxy | [nginx](https://nginx.org/) |
| 44   | background monitoring | [celery python](https://github.com/celery/celery)  |
| 45   | Stable diffusion UI | [next-js sd3](https://github.com/satyajitghana/sd3-ui) |