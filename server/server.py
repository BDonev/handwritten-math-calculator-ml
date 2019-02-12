from flask import Flask, request
import tensorflow as tf
import os.path
import cv2
import numpy as np
from scipy.misc import imread, imresize
import math
from keras.models import load_model


def get_math_problem(path):
    result_string = ""
    img = cv2.imread(path, 2)
    img_org = cv2.imread(path)
    img = cv2.bilateralFilter(img, 1, 75, 75)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    ret3, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours, bounding_boxes = sort_contours(contours)
    for j, cnt in enumerate(sorted_contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if (hierarchy[0][j][3] != -1 and w > 10 and h > 10):
            # putting boundary on each digit
            cv2.rectangle(img_org, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 1)

            # cropping each image and process
            roi = img[y - 10:y + h + 10, x - 10:x + w + 10]
            # roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            ret4, thresh2 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)

            # getting prediction of cropped image
            prediction_math_symbol, accuracy_math_symbol = predict_math_symbol(thresh2)
            prediction_digit, accuracy_digit = predict_mnist_digit(thresh2)

            print("Accuracy for digit: " + str(accuracy_digit))
            print("Accuracy for mathematical symbol: " + str(accuracy_math_symbol))
            if accuracy_digit >= accuracy_math_symbol:
                result_string += str(prediction_digit)
                print("Best prediction is: digit " + str(prediction_digit))
            else:
                result_string += str(prediction_math_symbol)
                print ("Best prediction is: mathematical symbol " + str(prediction_math_symbol))

    return result_string


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def predict_math_symbol(img):
    # img = np.invert(img)
    img = imresize(img, (45, 45))
    img = img.reshape(-1, 45, 45, 1)
    img = img.astype('float32')
    img /= 255

    with graph.as_default():
        prediction_array = model_math_symbols.predict_on_batch(img)
        prediction_math_symbol_index = np.argmax(prediction_array)
        print(prediction_array)
        prediction_accuracy = prediction_array[(0, prediction_math_symbol_index)]

        for key in label_map.keys():
            if (label_map[key] == prediction_math_symbol_index):
                if (key == "forward_slash"):
                    return "/", prediction_accuracy
                if (key == "times"):
                    return "*", prediction_accuracy
                if (key == ","):
                    return ".", prediction_accuracy
                return key, prediction_accuracy


def predict_mnist_digit(x):
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # make it the right size
    x = imresize(x, (28, 28))
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    x = x.astype('float32')
    x /= 255
    with graph.as_default():
        prediction_array = model_mnist.predict(x)
        prediction_digit_index = np.argmax(prediction_array)
        prediction_digit_accuracy = prediction_array[(0, prediction_digit_index)]
        return prediction_digit_index, prediction_digit_accuracy


# refining each digit
def image_refiner(gray):
    org_size = 45
    img_size = 45
    rows, cols = gray.shape

    if rows > cols:
        factor = org_size / rows
        rows = org_size
        cols = int(round(cols * factor))
    else:
        factor = org_size / cols
        cols = org_size
        rows = int(round(rows * factor))
    gray = cv2.resize(gray, (cols, rows))

    # get padding
    colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))

    # apply apdding
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    return gray


##########################################################################################
#
# Main execution flow
#
##########################################################################################


global model_math_symbols
global model_mnist
global graph
model_math_symbols = load_model('model_math_symbols.h5')
model_mnist = load_model('model_mnist.h5')
label_map = np.load('label_map_math_symbols.npy').item()
graph = tf.get_default_graph()
app = Flask(__name__)


@app.route('/uploadImage', methods=['POST'])
def upload_photo():
    try:
        print(request.headers)
        print(str(request.values))
        file = request.files['pic']
        file.save("image_to_predict.jpg")
        if os.path.isfile(file.filepath):
            return "Success"
        else:
            return "Failure"
    except:
        return "Failure"


@app.route('/getOutput', methods=['GET'])
def get_output():
    print(request.headers)
    print(str(request.values))
    path = "image_to_predict.jpg"
    if os.path.isfile(path):
        return get_math_problem(path)
    else:
        return "An image has not been uploaded"


if __name__ == '__main__':
    app.run()