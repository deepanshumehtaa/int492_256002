# import libraries
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# constants for image dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224

# initialize Flask app and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    """Load the ResNet50 pre-trained model."""
    global model
    model = ResNet50(weights='imagenet')

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
            
            # format the prediction output
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            
            for (imagenetID, label, prob) in results[0]:
                r = {"label" : label, "probability": float(prob)}
                data["predictions"].append(r)
                
            data["success"] = True
            
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* loading keras model and starting Flask server...")
    load_model()
    app.run()