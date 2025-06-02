import io
import json
import os

import numpy as np
import onnxruntime
import torchvision
from PIL import Image
from scipy.special import softmax
from ts.torch_handler.base_handler import BaseHandler


class VegFruitsHandler(BaseHandler):
    def __init__(self):
        super(__class__, self).__init__()
        self.model = None
        self.initialized = False
        self.input_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def initialize(self, ctx):
        """
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_file = "vegfruits.onnx"
        model_path = os.path.join(model_dir, model_file)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])

        # Load the ONNX model
        self.model = onnxruntime.InferenceSession(model_path)
        self.session_input_name = self.model.get_inputs()[0].name
        self.initialized = True

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        with open(mapping_file_path) as mapping_file:
            self.mapping = list(json.load(mapping_file).values())
        # by default it'll look for `index_to_name.json`
        print(f"mapping ::{self.mapping=}")

    def preprocess_one_image(self,req) -> np.ndarray:
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image

        print(f"received request of type:: {type(image)}")
        image = Image.open(io.BytesIO(image)).convert('RGB').resize(self.input_size) # isinstance(image, (bytes, bytearray))

        # preprocess
        image_array = np.array(image).astype(np.float32) / 255.0

        image_array = (image_array-self.mean) / self.std
        image_array = image_array.transpose(2,0,1)           # Transpose to channel-first format (NCHW)
        image_array = np.expand_dims(image_array,0 )         # Add batch dimension
        return image_array

    def preprocess(self, requests)->np.ndarray:
        """
            Basic text preprocessing, of the user's prompt.
            Args:
                requests (str): The Input data in the form of Bytes for an IMAGE
            Returns:
                list : The preprocess function returns a list of prompts.
        """
        images  = [ self.preprocess_one_image(req) for req in requests ]
        images  = np.concat(images,axis=0)
        print(f"received images and preprocessed:: shape({ images.shape})")
        return images

    def inference(self, data):
        '''
            # argmax(softmax(model(x)))
            Given the data from .preprocess, perform inference using the model.
            args:
                np.ndarray
            return:
                position of maximum probs as per image
        '''
        if self.model is None:
            raise RuntimeError("model not init!1")
        # Run inference using the ONNX model
        inputs = {self.session_input_name: data.astype(np.float32)}
        outputs = self.model.run(None, inputs)
        return outputs[0]

    def postprocess(self, inference_outputs):
        '''
            Implement your postprocessing logic here and Convert model output to the desired format
        '''
        probabilities = softmax(inference_outputs,axis=1)
        response:list = []
        for cls0,prob in zip(
                                np.argmax(probabilities,axis=1),
                                np.max(probabilities,axis=1), strict=False
                        ):
            response.append({self.mapping[cls0]:float(prob)} )
        return response
