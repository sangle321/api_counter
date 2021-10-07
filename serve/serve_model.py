from io import BytesIO
import numpy as np
from PIL import Image
from src.merged_model import CompletedModel
from pprint import pprint

model = None


def load_model():
    print("Model loading.....")
    model = CompletedModel()
    print("!!! Completed")

    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = np.asarray(image)
    result = model.response_data(img)

    return result


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))

    return image


# if __name__ == '__main__':
#     img = read_image_file("/home/sang/prj/RealTimeObjectDetection/TensorFlow/abc.jpg")
#     result = predict(img)
#     pprint(result)
