import json
import os
import platform
import socket
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path

import gradio as gr
import psutil
import traceback
import torch

from PIL import Image
from torchvision import transforms

notes = f"System Details:: {platform.system()}-{platform.release()}\nArchitecture: {platform.architecture()[0]}\nSocketName: {socket.getfqdn()}\nHostname:: {socket.gethostname()}\nCPU count:: {psutil.cpu_count(logical=True)}\nBoot time: { datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S") }\nUPtime:: {str(datetime.now()-datetime.fromtimestamp( psutil.boot_time() )).split('.')[0]}(HH:MM:SS)"

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autocast(enabled=True, dtype="float16", device_type="cuda")
torch.set_default_device(device=torch.device("cpu"))

###########################################################################################################
## # Read From S3
## BUCKET_NAME = 'emlo-project'
## ARTIFACTS   = ['checkpoints/pths/sports_cpu.pt','checkpoints/pths/vegfruits_cpu.pt','checkpoints/classnames/sports.json','checkpoints/classnames/vegfruits.json']
## # makedirs
## os.makedirs(exist_ok=True,name='checkpoints')
## os.makedirs(exist_ok=True,name=os.path.join('checkpoints','pths'))
## os.makedirs(exist_ok=True,name=os.path.join('checkpoints','classnames'))
## # s3 client
## s3_client   = boto3.client("s3",aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'), region_name='ap-south-1')
## for artifact in ARTIFACTS: s3_client.download_file(BUCKET_NAME,artifact,artifact)
## print(f"accesskey_id: {os.getenv('AWS_ACCESS_KEY_ID')}")
###########################################################################################################


# Define a no-op flagging callback
class NoOpFlaggingCallback(gr.FlaggingCallback):
    def setup(self, components, flagging_dir):
        pass  # Override setup to prevent directory creation

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        pass  # Do nothing when a flag is submitted


TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(name: str, path_dir: str = ".") -> torch.nn.Module:
    if torch.cuda.is_available():
        model = torch.jit.load(os.path.join(path_dir, f"{name}.pt"))
    else:
        model = torch.jit.load(os.path.join(path_dir, f"{name}_cpu.pt"))
    print(f"model loaded from {os.path.join(path_dir, f"{name}_cpu.pt")}")
    return model





@torch.no_grad()
def spredict_fn(img: Image.Image) -> tuple[dict[str, float], float]:
    start_time = timer()
    try:
        img = TEST_TRANSFORMS(img).to(device)
        logits = smodel(img.unsqueeze(0))
        probabilites = torch.softmax(logits, dim=-1)
        res = {}

        i2ls, pos = torch.topk(probabilites, k=5, dim=-1)
        for label, conf in zip(
            [scls2lbs[i.item()] for i in pos[0]],
            [i.item() for i in i2ls[0]],
            strict=False,
        ):
            res[label] = float(conf)

        pred_time = round(timer() - start_time, 5)
        return res, pred_time
    except Exception as e:
        print("Full traceback:")
        traceback.print_exc()
        return {"Error": 0.0}, 0.0


@torch.no_grad()
def vfpredict_fn(img: Image.Image) -> tuple[dict[str, float], float]:
    start_time = timer()
    try:
        img = TEST_TRANSFORMS(img).to(device)
        logits = vfmodel(img.unsqueeze(0))
        probabilites = torch.softmax(logits, dim=-1)
        res = {}

        i2ls, pos = torch.topk(probabilites, k=5, dim=-1)

        for label, conf in zip(
            [vfcls2lbs[i.item()] for i in pos[0]],
            [i.item() for i in i2ls[0]],
            strict=False,
        ):
            res[label] = float(conf)

        pred_time = round(timer() - start_time, 5)
        return res, pred_time
    except Exception as e:
        print("Full traceback:")
        traceback.print_exc()
        return {"Error": 0.0}, 0.0

if __name__ == "__main__":

    smodel = load_model(
        path_dir=os.path.join(os.getcwd(), "s3_files", "sports", "pths"), name="sports"
    )
    vfmodel = load_model(
        path_dir=os.path.join(os.getcwd(), "s3_files", "vegfruits", "pths"), name="vegfruits"
    )

    # ' r'./checkpoints/classnames/sports.json'
    with open(
        os.path.join(os.getcwd(), "s3_files", "sports", "index_to_name.json")
    ) as class2idxfile:
        # scls2lbs: dict = json.load(class2idxfile)
        scls2lbs = {int(k): v for k, v in json.load(class2idxfile).items()}
    # r'./checkpoints/classnames/vegfruits.json'
    with open(
        os.path.join(os.getcwd(), "s3_files", "vegfruits", "index_to_name.json")
    ) as class2idxfile:
        # vfcls2lbs: dict = json.load(class2idxfile)
        vfcls2lbs = {int(k): v for k, v in json.load(class2idxfile).items()}
        
    sports_interface = gr.Interface(
        fn=spredict_fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=5, label="predictions"),
            gr.Number(label="Prediction Time(s)"),
        ],
        title="Let's DO it!!",
        description="If data were a sport, I'll be the champion by now!!",
        article=f"<h3>Created by EMLO Practicals Group </h3><pre>{notes}</pre>",
        cache_examples=True,
        flagging_options=[],
        flagging_callback=NoOpFlaggingCallback(),
    )

    vegfruits_interface = gr.Interface(
        fn=vfpredict_fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=5, label="predictions"),
            gr.Number(label="Prediction Time(s)"),
        ],
        title="Let's EAT!!",
        description="I'm non-vegetarian only on weekends!!",
        article=f"<h3>Created by EMLO Practicals Group </h3><pre>{notes}</pre>",
        cache_examples=False,
        flagging_options=[],
        flagging_callback=NoOpFlaggingCallback(),
    )


    demo = gr.TabbedInterface(
        tab_names=["Sports", "VegFruits"],
        interface_list=[sports_interface, vegfruits_interface],
        analytics_enabled=True,
    )

    demo.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0",
        server_port=8080,
        enable_monitoring=False,
    )