import os
import base64
import secrets
import requests
from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from flask import Flask, session
from flask import render_template
from matplotlib.image import imread
from werkzeug.utils import secure_filename
from flask import request, redirect, url_for


# Generate a random secret key
secret_key = secrets.token_hex(16)

app = Flask(__name__)

app.secret_key = secret_key
app.config["STATIC_FOLDER"] = "static"
app.config["UPLOAD_FOLDER"] = "./static/uploads/"
app.config["PREDICTION_FOLDER"] = "./static/results/"

UPLOAD_FOLDER = "/static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


# Add a route that points to '/about' and return about.html page
@app.route("/about.html")
def about():
    return render_template("about.html")


# Add a route that points to '/team' and return team.html page
@app.route("/team.html")
def team():
    return render_template("team.html")


# Add a route that points to '/results' and return result.html page
@app.route("/results.html")
def results():
    return render_template("results.html")


# Add a route that points to '/demo' and return demo.html page
@app.route("/demo.html")
def demo():
    return render_template("demo.html")


# upload route
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    print("Inside upload file")
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # generate file to be uploaded
            destination = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(destination)
            print("Image uploaded successfully")
            predictions, destination_file = make_prediction(filename)
            if predictions:
                result = get_workers_status(predictions)
                # Store filename and result in session
                session["filename"] = filename
                session["result"] = result
                return redirect(url_for("prediction", filename=filename, result=result))
            return redirect(url_for("index"))
    return redirect(url_for("index"))


@app.route("/prediction")
def prediction():
    # Get filename and result from session
    filename = session.get("filename", None)
    result = session.get("result", None)
    new_filename = filename.rsplit(".", 1)[0] + ".png"
    return render_template("demo.html", filename=new_filename, result=result)


# https://detect.roboflow.com/ppe-detection-using-cv/3?api_key=api_key
def make_prediction(filename):
    api_url = "https://detect.roboflow.com/ppe-detection-using-cv/3"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    params = {"api_key": os.environ.get("ROBOFLOW_API_KEY")}
    with open(os.path.join(app.config["UPLOAD_FOLDER"], filename), "rb") as f:
        files = f.read()
        encoded_files = base64.b64encode(files).decode("utf-8")
        response = requests.post(
            api_url, headers=headers, params=params, data=encoded_files
        )
        predictions = None
        destination_filename = ""
        try:
            responseData = response.json()
            predictions = responseData.get("predictions", [])
            destination_filename = plot_boxes(predictions, filename)

        except ValueError:
            print("Response Status Code:", response.status_code)
            print("Response Content:", response.content.decode("utf-8"))
            print("Failed to decode response JSON")
        return predictions, destination_filename


def get_workers_status(predictions):
    if len(predictions) < 1:
        return {
            "data": None,
            "message": "Sorry! Model couldnot detect any object. Please try some other image.",
        }
    class_counts = {}
    for prediction in predictions:
        class_counts[prediction.get("class")] = (
            class_counts.get(prediction.get("class"), 0) + 1
        )
    message = ""
    if "Person" in class_counts:
        if "no-vest" in class_counts:
            message += "Missing safety vest.\n"
        if "no-helmet" in class_counts:
            message += "Missing safety helmet.\n"
        if "no-goggles" in class_counts:
            message += "Missing safety goggle.\n"
        if (
            class_counts.get("helmet", 0) < class_counts.get("Person")
            or class_counts.get("vest", 0) < class_counts.get("Person")
            or class_counts.get("goggles", 0) < class_counts.get("Person")
        ):
            message += "One or more workers are missing PPEs\n"
    return {"data": class_counts, "message": message}


def draw_boundingBox(filename, prediction):
    img = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    draw = ImageDraw.Draw(img)

    class_colors = {
        "Person": "red",
        "goggles": "blue",
        "helmet": "green",
        "vest": "yellow",
        "no-vest": "orange",
        "no-goggles": "purple",
        "no-helmet": "cyan",
    }
    predictions = prediction.get("predictions")
    for boundingbox in predictions:
        x = boundingbox.get("x")
        y = boundingbox.get("y")
        width = boundingbox.get("width")
        height = boundingbox.get("height")
        labels = boundingbox.get("class")
        confidence_score = boundingbox.get("confidence")
        color = class_colors.get(labels, "white")
        draw.rectangle(
            ((x - width / 2, y - height / 2), (x + width / 2, y + height / 2)),
            outline=color,
            width=2,
        )

        text = f"{labels}: {confidence_score:.2f}"
        left, top, right, bottom = draw.textbbox(
            (x - width / 2, y - height / 2 - 10), text
        )
        draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=color)
        draw.text((x - width / 2, y - height / 2 - 10), text, fill="white")

        # draw.text((x - width / 2, y - height / 2 - 10), text, fill=color)
    new_filename = filename.rsplit(".", 1)[0] + ".png"
    img.save(os.path.join(app.config["PREDICTION_FOLDER"], new_filename), "PNG")
    return new_filename


def plot_boxes(box_data, filename):
    # read the image
    img = imread(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    class_colors = {
        "Person": "red",
        "goggles": "blue",
        "helmet": "green",
        "vest": "yellow",
        "no-vest": "orange",
        "no-goggles": "purple",
        "no-helmet": "cyan",
    }

    # create a new plot
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # hide the axes
    ax.axis("off")

    # loop over all the boxes in the box_data
    for box in box_data:
        x = box.get("x")
        y = box.get("y")
        width = box.get("width")
        height = box.get("height")
        confidence_score = box.get("confidence")
        label = box.get("class")
        color = class_colors.get(label, "white")

        # create a Rectangle patch
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            box["width"],
            box["height"],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        # add the rectangle to the Axes
        ax.add_patch(rect)

        # annotate the class and confidence
        ax.text(
            x - width / 2,
            y - height / 2 - 10,
            f'{box["class"]}: {box["confidence"]:.2f}',
            bbox=dict(facecolor=color, alpha=0.5),
            color="black",
        )
    new_filename = filename.rsplit(".", 1)[0] + ".png"
    outfile = os.path.join(app.config["PREDICTION_FOLDER"], new_filename)
    plt.savefig(outfile)

    return outfile


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3111))
    app.run(host="0.0.0.0", port=port)
