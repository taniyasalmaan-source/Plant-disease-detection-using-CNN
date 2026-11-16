import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError

from model_service import (
    DEFAULT_ABSTENTION_MARGIN,
    DEFAULT_DECISION_THRESHOLD,
    ModelOutputError,
    ModelService,
    ModelUnavailableError,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
DATABASE = BASE_DIR / "instance" / "predictions.sqlite3"
MAX_IMAGE_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 64 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATABASE.parent.mkdir(parents=True, exist_ok=True)

model_service = ModelService(MODEL_DIR)

CLASS_INFO = {
    "healthy": {
        "display_name": "Healthy Rose Leaf",
        "status": "healthy",
        "description": "The image was classified as healthy by the binary rose leaf model.",
        "symptoms": ["Green leaves", "No visible disease spots", "Normal leaf structure"],
        "causes": ["Not applicable. Continue monitoring for changes."],
        "precautions": [
            "Maintain regular watering.",
            "Provide sufficient sunlight.",
            "Use balanced nutrition.",
            "Monitor leaves regularly.",
        ],
        "treatment": ["No treatment is indicated by this screening. Continue regular rose care."],
    },
    "unhealthy": {
        "display_name": "Unhealthy Rose Leaf",
        "status": "unhealthy",
        "description": (
            "The binary model detected signs that may be consistent with an unhealthy leaf. "
            "This screening does not identify a specific disease or confirm a diagnosis."
        ),
        "symptoms": [
            "Possible discoloration, spots, coating, or damaged leaf tissue.",
            "Symptoms can vary and may have non-disease causes.",
        ],
        "causes": [
            "Possible causes include disease, pests, water or nutrient stress, or environmental damage."
        ],
        "precautions": [
            "Inspect nearby leaves and monitor whether symptoms spread.",
            "Remove fallen or badly damaged leaves and keep tools clean.",
            "Avoid overhead watering and improve air circulation.",
            "Ask a local plant-health expert before applying treatments.",
        ],
        "treatment": [
            "No specific disease treatment can be selected from a binary result. Seek expert advice "
            "before using a pesticide or fungicide.",
        ],
    },
}

UNIDENTIFIED_INFO = {
    "display_name": "Prediction Uncertain",
    "status": "unidentified",
    "description": (
        "The model's scores are too close to its decision threshold for a confident binary result. "
        "This app does not have a separate rose-leaf verification model."
    ),
}


def normalized_class_name(class_name):
    if class_name in {"healthy", "unhealthy", "unidentified"}:
        return class_name
    return "unhealthy"


@contextmanager
def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db():
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                image_filename TEXT NOT NULL,
                original_filename TEXT NOT NULL DEFAULT '',
                image_location TEXT NOT NULL DEFAULT '',
                probability_healthy REAL,
                probability_unhealthy REAL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "original_filename" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN original_filename TEXT NOT NULL DEFAULT ''"
            )
        if "image_location" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN image_location TEXT NOT NULL DEFAULT ''"
            )
        if "probability_healthy" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN probability_healthy REAL")
        if "probability_unhealthy" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN probability_unhealthy REAL")
        connection.execute(
            "UPDATE predictions SET original_filename = image_filename "
            "WHERE original_filename = ''"
        )
        connection.execute(
            "UPDATE predictions SET image_location = 'static/uploads/' || image_filename "
            "WHERE image_location = ''"
        )
        connection.execute(
            "UPDATE predictions SET probability_healthy = "
            "CASE WHEN class_name = 'healthy' THEN confidence ELSE 1 - confidence END "
            "WHERE probability_healthy IS NULL"
        )
        connection.execute(
            "UPDATE predictions SET probability_unhealthy = 1 - probability_healthy "
            "WHERE probability_unhealthy IS NULL"
        )


def image_error(message, status=400):
    return jsonify({"error": message}), status


def prediction_record(row):
    record = dict(row)
    record["class_name"] = normalized_class_name(record["class_name"])
    record["disease"] = (
        UNIDENTIFIED_INFO
        if record["class_name"] == "unidentified"
        else CLASS_INFO[record["class_name"]]
    )
    if record.get("probability_healthy") is None:
        record["probability_healthy"] = (
            record["confidence"]
            if record["class_name"] == "healthy"
            else 1 - record["confidence"]
        )
    if record.get("probability_unhealthy") is None:
        record["probability_unhealthy"] = 1 - record["probability_healthy"]
    record["model_score"] = record["confidence"]
    record["original_filename"] = (
        record.get("original_filename") or record["image_filename"]
    )
    record["image_location"] = (
        record.get("image_location") or "static/uploads/" + record["image_filename"]
    )
    return record


def prepare_image(file_storage):
    original_filename = (file_storage.filename or "Uploaded image").replace("\\", "/")
    original_filename = original_filename.rsplit("/", 1)[-1]
    original_filename = "".join(
        character for character in original_filename if character.isprintable()
    ).strip()[:255]
    if not original_filename:
        original_filename = "Uploaded image"

    raw = file_storage.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be 8 MB or smaller.")

    try:
        from io import BytesIO

        with Image.open(BytesIO(raw)) as image:
            image_format = image.format
            if image_format not in {"JPEG", "PNG"}:
                raise ValueError("Choose a JPG, JPEG, or PNG image.")
            if image.width < 64 or image.height < 64:
                raise ValueError("Image must be at least 64 × 64 pixels.")
            if image.width * image.height > 25_000_000:
                raise ValueError("Image dimensions are too large to process safely.")
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image = image.convert("RGB")
            image.load()
            if all(low == high for low, high in image.getextrema()):
                raise ValueError("This image appears blank or uniform. Upload a clear rose leaf photo.")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("This file is not a valid, readable image.") from error

    extension = ".jpg" if image_format == "JPEG" else ".png"
    return raw, image, extension, original_filename


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/dashboard")
def dashboard():
    with get_db() as connection:
        counts = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(CASE WHEN class_name = 'healthy' THEN 1 ELSE 0 END), 0)
                       AS healthy,
                   COALESCE(SUM(CASE WHEN class_name = 'unhealthy' OR
                       (class_name != 'healthy' AND class_name != 'unidentified') THEN 1 ELSE 0 END), 0)
                       AS unhealthy,
                   COALESCE(SUM(CASE WHEN class_name = 'unidentified' THEN 1 ELSE 0 END), 0)
                       AS unidentified
            FROM predictions
            """
        ).fetchone()
        rows = connection.execute(
            "SELECT id, image_filename, original_filename, image_location, "
            "probability_healthy, probability_unhealthy, class_name, confidence, created_at "
            "FROM predictions "
            "ORDER BY created_at DESC LIMIT 6"
        ).fetchall()
    total = counts["total"]
    healthy = counts["healthy"]
    unhealthy = counts["unhealthy"]
    unidentified = counts["unidentified"]
    stats = {
        "total": total,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "unidentified": unidentified,
        "healthy_percent": round(healthy * 100 / total) if total else 0,
        "unhealthy_percent": round(unhealthy * 100 / total) if total else 0,
        "unidentified_percent": round(unidentified * 100 / total) if total else 0,
    }
    status = model_service.status()
    return render_template(
        "dashboard.html",
        stats=stats,
        records=[prediction_record(row) for row in rows],
        model_status=status,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
def history():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT id, image_filename, original_filename, image_location, "
            "probability_healthy, probability_unhealthy, class_name, confidence, created_at "
            "FROM predictions ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    records = [prediction_record(row) for row in rows]
    return render_template("history.html", records=records)


@app.route("/result/<prediction_id>")
def result(prediction_id):
    with get_db() as connection:
        row = connection.execute(
            "SELECT id, image_filename, original_filename, image_location, "
            "probability_healthy, probability_unhealthy, class_name, confidence, created_at "
            "FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
    if row is None:
        abort(404)
    status = model_service.status()
    model_warning = status.get("warning")
    if not status.get("available"):
        model_warning = status.get("message")
    return render_template(
        "result.html",
        record=prediction_record(row),
        model_warning=model_warning,
    )


@app.route("/api/model-status")
def model_status():
    return jsonify(model_service.status())


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return image_error("Choose a rose leaf image to continue.")
    try:
        raw, image, extension, original_filename = prepare_image(request.files["image"])
    except ValueError as error:
        return image_error(str(error))

    try:
        prediction = model_service.predict(image)
    except ModelUnavailableError as error:
        return image_error(str(error), 503)
    except ModelOutputError as error:
        app.logger.warning("Unable to interpret model output: %s", error)
        message = "Unable to identify this image: {}".format(error)
        return jsonify(
            {
                "status": "unidentified",
                "prediction": "UNIDENTIFIED",
                "confidence": None,
                "disease_name": None,
                "symptoms": [],
                "treatment": [],
                "message": message,
                "error": message,
            }
        ), 422
    except (ValueError, RuntimeError) as error:
        app.logger.exception("Prediction failed")
        return image_error("The image could not be analyzed. Please try another clear leaf photo.", 500)

    if not isinstance(prediction, dict):
        app.logger.error(
            "Binary model service returned %s instead of a prediction object",
            type(prediction).__name__,
        )
        return image_error("The model returned an unsupported result. Check the trained binary model.", 500)
    class_name = prediction.get("class_name")
    if class_name is not None and class_name not in CLASS_INFO:
        app.logger.error(
            "Binary model service returned an unsupported class label: %r",
            class_name,
        )
        return image_error("The model returned an unsupported result. Check the trained binary model.", 500)
    expected_status = class_name or "unidentified"
    expected_prediction = expected_status.upper()
    if prediction.get("status", expected_status) != expected_status or prediction.get(
        "prediction", expected_prediction
    ) != expected_prediction:
        app.logger.error("Model result labels disagree: %r", prediction)
        return image_error("The model returned inconsistent result labels.", 500)
    confidence = prediction.get("confidence")
    probability_unhealthy = prediction.get("probability_unhealthy")
    probability_healthy = prediction.get(
        "probability_healthy",
        1 - probability_unhealthy if isinstance(probability_unhealthy, (int, float)) else None,
    )
    if (class_name is None and confidence is not None) or (
        class_name is not None and not isinstance(confidence, (int, float))
    ):
        app.logger.error("Binary model returned invalid confidence values: %r", prediction)
        return image_error("The model returned invalid confidence values.", 500)
    if (
        not isinstance(probability_healthy, (int, float))
        or not isinstance(probability_unhealthy, (int, float))
        or not math.isfinite(probability_healthy)
        or not math.isfinite(probability_unhealthy)
        or not 0 <= probability_healthy <= 1
        or not 0 <= probability_unhealthy <= 1
        or not math.isclose(
            probability_healthy + probability_unhealthy,
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
    ):
        app.logger.error("Binary model returned invalid probabilities: %r", prediction)
        return image_error("The model returned invalid confidence values.", 500)
    decision_threshold = prediction.get(
        "decision_threshold", DEFAULT_DECISION_THRESHOLD
    )
    abstention_margin = prediction.get(
        "abstention_margin", DEFAULT_ABSTENTION_MARGIN
    )
    threshold_calibrated = prediction.get("threshold_calibrated", False)
    if (
        not isinstance(decision_threshold, (int, float))
        or isinstance(decision_threshold, bool)
        or not math.isfinite(decision_threshold)
        or not 0 <= decision_threshold <= 1
        or not isinstance(abstention_margin, (int, float))
        or isinstance(abstention_margin, bool)
        or not math.isfinite(abstention_margin)
        or not 0 <= abstention_margin <= 0.5
        or not isinstance(threshold_calibrated, bool)
    ):
        app.logger.error("Binary model returned invalid decision settings: %r", prediction)
        return image_error("The model returned invalid decision settings.", 500)
    expected_class = (
        "unhealthy"
        if probability_unhealthy >= decision_threshold
        else "healthy"
    )
    is_uncertain = (
        abs(probability_unhealthy - decision_threshold) < abstention_margin
    )
    if class_name is None:
        if not is_uncertain:
            app.logger.error("Binary model abstained outside its uncertainty band: %r", prediction)
            return image_error("The model returned an inconsistent abstention.", 500)
        stored_class_name = "unidentified"
        stored_confidence = max(probability_healthy, probability_unhealthy)
    else:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            app.logger.error("Binary model returned invalid confidence values: %r", prediction)
            return image_error("The model returned invalid confidence values.", 500)
        expected_confidence = (
            probability_unhealthy
            if class_name == "unhealthy"
            else probability_healthy
        )
        if not math.isclose(confidence, expected_confidence, rel_tol=1e-5, abs_tol=1e-5):
            app.logger.error("Binary model returned inconsistent confidence values: %r", prediction)
            return image_error("The model returned inconsistent confidence values.", 500)
        if class_name != expected_class or is_uncertain:
            app.logger.error("Binary model returned an inconsistent class mapping: %r", prediction)
            return image_error("The model returned an inconsistent class mapping.", 500)
        stored_class_name = class_name
        stored_confidence = confidence

    prediction_id = uuid.uuid4().hex
    filename = "{}{}".format(prediction_id, extension)
    image_path = UPLOAD_DIR / filename
    try:
        image_location = image_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        image_location = image_path.as_posix()
    try:
        image_path.write_bytes(raw)
    except OSError:
        app.logger.exception("Could not save uploaded image")
        return image_error("The image could not be saved. Check the available disk space and try again.", 500)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with get_db() as connection:
            connection.execute(
                "INSERT INTO predictions "
                "(id, image_filename, original_filename, image_location, probability_healthy, "
                "probability_unhealthy, class_name, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prediction_id,
                    filename,
                    original_filename,
                    image_location,
                    probability_healthy,
                    probability_unhealthy,
                    stored_class_name,
                    stored_confidence,
                    created_at,
                ),
            )
    except sqlite3.Error:
        image_path.unlink(missing_ok=True)
        app.logger.exception("Could not save prediction history")
        return image_error("Prediction completed, but its history could not be saved.", 500)

    disease = UNIDENTIFIED_INFO if class_name is None else CLASS_INFO[class_name]
    if class_name is None:
        message = (
            "The healthy and unhealthy scores fall within the model's uncertainty band. "
            "This binary model does not include a separate rose-leaf verifier."
        )
    else:
        message = prediction.get("message")
        if not isinstance(message, str) or not message.strip():
            message = disease["description"]
    response = {
        "identifiable": class_name is not None,
        "id": prediction_id,
        "result_url": url_for("result", prediction_id=prediction_id),
        "image_url": url_for("static", filename="uploads/" + filename),
        "original_filename": original_filename,
        "class_name": class_name,
        "disease": disease["display_name"],
        "status": disease["status"],
        "prediction": expected_prediction,
        "confidence": confidence,
        "model_score": stored_confidence,
        "disease_name": disease.get("disease_name"),
        "symptoms": disease.get("symptoms", []),
        "treatment": disease.get("treatment", []),
        "message": message,
        "probability_healthy": probability_healthy,
        "probability_unhealthy": probability_unhealthy,
        "decision_threshold": decision_threshold,
        "threshold_calibrated": threshold_calibrated,
    }
    return jsonify(response)


@app.errorhandler(413)
def request_too_large(_error):
    if request.path.startswith("/api/"):
        return image_error("Image must be 8 MB or smaller.", 413)
    return render_template("error.html", message="That upload is larger than 8 MB."), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return image_error("The requested resource was not found.", 404)
    return render_template("error.html", message="We couldn't find that page."), 404


@app.errorhandler(500)
def internal_error(_error):
    if request.path.startswith("/api/"):
        return image_error("An unexpected server error occurred. Please try again.", 500)
    return render_template("error.html", message="An unexpected error occurred. Please try again."), 500


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
