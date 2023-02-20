from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow
from pickle import load
from numpy import argmax
from keras_preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

app = Flask(__name__)

max_length = 34
tokenizer = load(open('tokenizer.pkl', 'rb'))

model = load_model('model_18.h5')

model.make_predict_function()

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word


def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    in_text = ' '.join(in_text.split()[1:-1])
    return in_text

def extract_features(filename):
    # load the model
    print("in function")
    model = VGG16()
    print("i will cry")

    # re-structure the model
    model.layers.pop()
    print("whyyyyyyyyy")

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    print("loaed image")

    image = load_img(filename, target_size=(224, 224))
    print("will convert to array")

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    print("will reshape")

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    print("reshape done")
    image = preprocess_input(image)
    print("after preprocess")
    # get features
    feature = model.predict(image, verbose=0)
    print("every thing here done")
    return feature

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Auto Image Caption..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = extract_features(img_path)
		description = generate_desc(model, tokenizer, p, max_length)
		

	return render_template("index.html", prediction = description, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)