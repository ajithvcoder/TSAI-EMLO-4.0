from transformers import AutoImageProcessor, AutoModelForImageClassification

def get_processor_and_model(hf_string):
	processor = AutoImageProcessor.from_pretrained(hf_string, use_fast=True)
	model = AutoModelForImageClassification.from_pretrained(hf_string)

	return [processor, model]

def save_model_processor(model, processor, save_prefix_str):
	model.save_pretrained(f"./models/{save_prefix_str}/model")
	processor.save_pretrained(f"./models/{save_prefix_str}/processor")

[imagenet_processor, imagenet_model] = get_processor_and_model("google/vit-base-patch16-224")
save_model_processor(imagenet_model, imagenet_processor, "imagenet-vit")

[food_processor, food_model] = get_processor_and_model("nateraw/food")
save_model_processor(food_model, food_processor, "food-classifier")

[cat_processor, cat_model] = get_processor_and_model("dima806/67_cat_breeds_image_detection")
save_model_processor(cat_model, cat_processor, "cat-classifier")

[dog_processor, dog_model] = get_processor_and_model("skyau/dog-breed-classifier-vit")
save_model_processor(dog_model, dog_processor, "dog-classifier")

[indian_food_processor, indian_food_model] = get_processor_and_model("Amrrs/indian-foods")
save_model_processor(indian_food_model, indian_food_processor, "indian-food-classifier")