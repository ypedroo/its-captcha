from base64 import b64encode
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from wtforms import SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'

bootstrap = Bootstrap(app)

saved_model = load_model("models/train_data.h5")
saved_model._make_predict_function()

#graph = tf.get_default_graph()
#sess = tf.Session()

print(tf.__version__)


class UploadForm(FlaskForm):
    photo = FileField('Upload an image', validators=[FileAllowed(
        ['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')


def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    desired_width, desired_height = 150, 150

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height),
                    start_x+desired_width, height))
    img = img.resize((150, 150))

    img = image.img_to_array(img)
    return img / 255.


@app.route('/', methods=['GET', 'POST'])
def predict():
    #global sess
    #global graph
    #with graph.as_default():
    #   sess.run(tf.global_variables_initializer())
    #  try:
    form = UploadForm()
    if form.validate_on_submit():
        print(form.photo.data)
        image_stream = form.photo.data.stream
        original_img = Image.open(image_stream)
        img = image.img_to_array(original_img)
        img = preprocess(img)
        img = np.expand_dims(img, axis=0)
        prediction = saved_model.predict_classes(img)

        if (prediction[0][0] == 0):
            result = "DOG"
            print(result)
        else:
            result = "NOT DOG"
            print(result)

        byteIO = BytesIO()
        original_img.save(byteIO, format=original_img.format)
        byteArr = byteIO.getvalue()
        encoded = b64encode(byteArr)

        return render_template('result.html', result=result, encoded_photo=encoded.decode('ascii'))
        #except Exception as e:
         #   print(e)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
