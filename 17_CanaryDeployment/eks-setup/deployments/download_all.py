from transformers import AutoImageProcessor, AutoModelForImageClassification

def get_processor_and_model(hf_string):
	processor = AutoImageProcessor.from_pretrained(hf_string, use_fast=True)
	model = AutoModelForImageClassification.from_pretrained(hf_string)

	return [processor, model]

def save_model_processor(model, processor, save_prefix_str):
	model.save_pretrained(f"./models/{save_prefix_str}/model")
	processor.save_pretrained(f"./models/{save_prefix_str}/processor")

[imagenet_processor, imagenet_model] = get_processor_and_model("facebook/deit-tiny-patch16-224")
save_model_processor(imagenet_model, imagenet_processor, "imagenet-m1")

[imagenet_processor_2, imagenet_model_2] = get_processor_and_model("facebook/deit-small-patch16-224")
save_model_processor(imagenet_model_2, imagenet_processor_2, "imagenet-m2")

[imagenet_processor_3, imagenet_model_3] = get_processor_and_model("WinKawaks/vit-tiny-patch16-224")
save_model_processor(imagenet_model_3, imagenet_processor_3, "imagenet-m3")


"""
https://huggingface.co/facebook/deit-tiny-patch16-224
facebook/deit-small-patch16-224
https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""