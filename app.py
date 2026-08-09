import os
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "breast_cancer_efficientnet.keras"

model = load_model(MODEL_PATH)

# ==============================
# HELPERS
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    """
    Load an image from disk and generate a prediction.

    The EfficientNetB4 model in your notebook was trained with IMG_SIZE = 380,
    so we resize to (380, 380) here to match the training configuration.
    """
    img = image.load_img(img_path, target_size=(380, 380))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return "Malignant", round(pred * 100, 2)
    else:
        return "Benign", round((1 - pred) * 100, 2)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests from the upload form.

    NOTE: The HTML form in index.html uses name="image" for the file input,
    so we must read request.files["image"] here (not "file").
    """
    # Ensure the expected field is present
    if "image" not in request.files:
        return render_template("result.html", error="No file uploaded")

    file = request.files["image"]

    # Empty filename = no file actually chosen
    if file.filename == "":
        return render_template("result.html", error="No file selected")

    # Validate extension and run prediction
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        image_url = url_for("static", filename=f"uploads/{filename}")

        label, confidence = predict_image(filepath)

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            image_url=image_url,
        )

    # Fallback: extension not allowed
    return render_template("result.html", error="Invalid file type")


if __name__ == "__main__":
    app.run(debug=True)