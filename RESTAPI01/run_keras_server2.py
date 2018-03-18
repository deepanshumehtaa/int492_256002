# import libraries
import keras.models
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# constants for image dimensions
IMG_WIDTH = 150
IMG_HEIGHT = 150

# initialize Flask app and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():    
    """Load our cats and dogs VGG16 model."""    
    global model
    # Load our model 
    model = keras.models.load_model('cats_and_dogs-vgg16.h5')

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize and make the image ready as an input to ResNet50
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    iamge = imagenet_utils.preprocess_input(image)
    
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success" : False}
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess the image
            image = prepare_image(image, target=(IMG_WIDTH, IMG_HEIGHT))
            
            # classify the image
            preds = model.predict(image)
            
            # build the results
            data["predictions"] = []            
            for pred in preds:
                if pred:
                    label = 'a dog'
                else:
                    label = 'a cat'
                r = {"label" : label, "probability": None}
                
                data["predictions"].append(r)
                
            data["success"] = True
            
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* loading keras model and starting Flask server...")
    load_model()
    app.run()