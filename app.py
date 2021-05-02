import cv2
from flask import Flask,request,jsonify,render_template,redirect,url_for,send_from_directory
from tensorflow import keras
import cv2
import numpy as np
from tensorflow import keras
# from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

IMG_SIZE = 28
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

app = Flask(__name__)


@app.route('/')
def hello():
	return render_template('index.html')

def preprocess_img(filename):
	img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
	img = img/255.0
	img = img.reshape(1,IMG_SIZE,IMG_SIZE,1)
	return img

@app.route('/upload',methods=['POST'])
def upload():
	img = request.files['image']
	img.save('static/'+img.filename)
	filename = 'static/'+img.filename
	img = preprocess_img(filename)
	model = load_model('models/sign_classifier.h5')
	category = model.predict_classes(img)
	print(category)
	#print(category[0])
	pred = classes[category[0]]

	return render_template('prediction.html',data = pred)


if __name__=="__main__":
	app.run(debug=True)