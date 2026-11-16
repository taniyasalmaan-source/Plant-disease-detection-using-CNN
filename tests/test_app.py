import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as rose_app
from model_service import (
    CLASS_NAMES,
    ModelOutputError,
    ModelService,
    ModelUnavailableError,
)
from train import evaluate_binary_predictions, select_decision_threshold


class FakeModelService:
    def __init__(self):
        self.class_name = "healthy"

    def status(self):
        return {"available": True, "message": "Test model is ready."}

    def predict(self, _image):
        probability_unhealthy = 0.875 if self.class_name == "unhealthy" else 0.125
        return {
            "class_name": self.class_name,
            "confidence": 0.875,
            "probability_unhealthy": probability_unhealthy,
        }


class RoseAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = root / "predictions.sqlite3"
        self.upload_dir = root / "uploads"
        self.upload_dir.mkdir()
        self.database_patch = patch.object(rose_app, "DATABASE", self.db_path)
        self.upload_patch = patch.object(rose_app, "UPLOAD_DIR", self.upload_dir)
        self.service = FakeModelService()
        self.service_patch = patch.object(rose_app, "model_service", self.service)
        self.database_patch.start()
        self.upload_patch.start()
        self.service_patch.start()
        rose_app.init_db()
        rose_app.app.config.update(TESTING=True)
        self.client = rose_app.app.test_client()

    def tearDown(self):
        self.service_patch.stop()
        self.upload_patch.stop()
        self.database_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def image_upload(name="leaf.jpg", image_format="JPEG"):
        image_bytes = io.BytesIO()
        image = Image.new("RGB", (96, 96))
        image.putdata(
            [
                (x * 2, y * 2, (x + y) % 256)
                for y in range(96)
                for x in range(96)
            ]
        )
        image.save(image_bytes, format=image_format)
        image_bytes.seek(0)
        return {"image": (image_bytes, name)}

    def test_pages_render_and_history_starts_empty(self):
        for path in ("/", "/detect", "/dashboard", "/about", "/history"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_missing_and_invalid_uploads_are_rejected(self):
        self.assertEqual(self.client.post("/api/predict").status_code, 400)
        response = self.client.post(
            "/api/predict",
            data=self.image_upload("leaf.gif", "GIF"),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("JPG, JPEG, or PNG", response.get_json()["error"])
        corrupt = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"not an image"), "corrupt.jpg")},
        )
        self.assertEqual(corrupt.status_code, 400)
        self.assertIn("not a valid", corrupt.get_json()["error"])
        mislabeled = self.client.post(
            "/api/predict",
            data=self.image_upload("leaf.jpg", "PNG"),
        )
        self.assertEqual(mislabeled.status_code, 200)
        self.assertTrue(mislabeled.get_json()["image_url"].endswith(".png"))
        empty = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b""), "empty.jpg")},
        )
        self.assertEqual(empty.status_code, 400)
        oversized = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"x" * (rose_app.MAX_IMAGE_BYTES + 1)), "large.jpg")},
        )
        self.assertEqual(oversized.status_code, 400)

    def test_supported_image_content_is_accepted_without_a_training_filename(self):
        for filename, image_format, expected_extension in (
            ("WhatsApp Image 2026-11-13 at 19.30.00", "JPEG", ".jpg"),
            ("rose-leaf-upload.bin", "PNG", ".png"),
        ):
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/api/predict",
                    data=self.image_upload(filename, image_format),
                )
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertTrue(result["image_url"].endswith(expected_extension))
                self.assertEqual(result["original_filename"], filename)
                with rose_app.get_db() as connection:
                    row = connection.execute(
                        "SELECT original_filename, image_location FROM predictions WHERE id = ?",
                        (result["id"],),
                    ).fetchone()
                self.assertEqual(row["original_filename"], filename)
                self.assertTrue(Path(row["image_location"]).is_file())
                page = self.client.get(result["result_url"])
                self.assertIn(filename.encode(), page.data)

    def test_dashboard_uses_live_database_counts_and_recent_predictions(self):
        empty_page = self.client.get("/dashboard")
        self.assertIn(b"Total images analyzed", empty_page.data)
        self.assertIn(b"No image checks have been recorded yet", empty_page.data)
        self.assertIn(b"<strong>0</strong>", empty_page.data)

        self.client.post(
            "/api/predict",
            data=self.image_upload("garden-photo.jpeg"),
        )
        self.service.class_name = "unhealthy"
        self.client.post(
            "/api/predict",
            data=self.image_upload("WhatsApp rose leaf.png", "PNG"),
        )

        page = self.client.get("/dashboard")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Total images analyzed", page.data)
        self.assertIn(b"Healthy predictions", page.data)
        self.assertIn(b"Unhealthy predictions", page.data)
        self.assertIn(b"garden-photo.jpeg", page.data)
        self.assertIn(b"WhatsApp rose leaf.png", page.data)
        self.assertIn(b"width: 50%", page.data)
        self.assertIn(b"<strong>2</strong>", page.data)

    def test_existing_database_is_migrated_with_filename_and_image_location(self):
        legacy_db = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_db)
        try:
            connection.execute(
                "CREATE TABLE predictions ("
                "id TEXT PRIMARY KEY, image_filename TEXT NOT NULL, "
                "class_name TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                ("previous", "previous.jpg", "healthy", 0.8, "2026-01-01T00:00:00+00:00"),
            )
            connection.commit()
        finally:
            connection.close()
        with patch.object(rose_app, "DATABASE", legacy_db):
            rose_app.init_db()
            with rose_app.get_db() as connection:
                row = connection.execute(
                    "SELECT original_filename, image_location FROM predictions WHERE id = ?",
                    ("previous",),
                ).fetchone()
        self.assertEqual(row["original_filename"], "previous.jpg")
        self.assertEqual(row["image_location"], "static/uploads/previous.jpg")

    def test_blank_and_low_resolution_images_are_rejected(self):
        blank = io.BytesIO()
        Image.new("RGB", (96, 96), color=(255, 255, 255)).save(blank, format="JPEG")
        blank.seek(0)
        response = self.client.post(
            "/api/predict",
            data={"image": (blank, "blank.jpg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("blank or uniform", response.get_json()["error"])

        tiny = io.BytesIO()
        image = Image.new("RGB", (32, 32))
        image.putdata([(x * 4, y * 4, (x + y) % 255) for y in range(32) for x in range(32)])
        image.save(tiny, format="JPEG")
        tiny.seek(0)
        response = self.client.post(
            "/api/predict",
            data={"image": (tiny, "tiny.jpg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("64", response.get_json()["error"])

    def test_prediction_saves_and_renders_healthy_and_unhealthy_results(self):
        healthy = self.client.post("/api/predict", data=self.image_upload())
        self.assertEqual(healthy.status_code, 200)
        healthy_data = healthy.get_json()
        self.assertEqual(healthy_data["status"], "healthy")
        self.assertEqual(healthy_data["prediction"], "HEALTHY")
        self.assertIsNone(healthy_data["disease_name"])
        self.assertIsInstance(healthy_data["symptoms"], list)
        self.assertIsInstance(healthy_data["treatment"], list)
        self.assertTrue(healthy_data["message"])
        self.assertAlmostEqual(healthy_data["probability_healthy"], 0.875)
        self.assertAlmostEqual(healthy_data["probability_unhealthy"], 0.125)
        page = self.client.get(healthy_data["result_url"])
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Your rose leaf is healthy.", page.data)
        self.assertIn(b"Healthy score", page.data)
        self.assertIn(b"Unhealthy score", page.data)

        self.service.class_name = "unhealthy"
        diseased = self.client.post("/api/predict", data=self.image_upload("spot.png", "PNG"))
        self.assertEqual(diseased.status_code, 200)
        diseased_data = diseased.get_json()
        self.assertEqual(diseased_data["class_name"], "unhealthy")
        self.assertEqual(diseased_data["status"], "unhealthy")
        self.assertEqual(diseased_data["prediction"], "UNHEALTHY")
        self.assertIsNone(diseased_data["disease_name"])
        self.assertTrue(diseased_data["symptoms"])
        self.assertTrue(diseased_data["treatment"])
        page = self.client.get(diseased_data["result_url"])
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"does not identify a specific disease", page.data)
        self.assertIn(b"87.5%", page.data)
        self.assertIn(b"Symptoms", page.data)
        self.assertIn(b"Possible discoloration, spots", page.data)
        self.assertIn(b"Precautions", page.data)
        self.assertIn(b"improve air circulation", page.data)
        self.assertNotIn(b"Black Spot", page.data)
        self.assertNotIn(b"Powdery Mildew", page.data)
        self.assertEqual(self.client.get("/history").status_code, 200)

    def test_model_unavailable_is_reported_without_saving_history(self):
        class Unavailable:
            def predict(self, _image):
                raise rose_app.ModelUnavailableError("No trained model")

        with patch.object(rose_app, "model_service", Unavailable()):
            response = self.client.post("/api/predict", data=self.image_upload())
        self.assertEqual(response.status_code, 503)
        self.assertIn("No trained model", response.get_json()["error"])
        with rose_app.get_db() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)

    def test_unusable_model_output_is_reported_as_unidentified(self):
        class NoConfidence:
            def predict(self, _image):
                raise ModelOutputError(
                    "The model returned only a class index and no confidence score."
                )

        with patch.object(rose_app, "model_service", NoConfidence()):
            response = self.client.post("/api/predict", data=self.image_upload())
        self.assertEqual(response.status_code, 422)
        result = response.get_json()
        self.assertEqual(result["status"], "unidentified")
        self.assertEqual(result["prediction"], "UNIDENTIFIED")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["disease_name"])
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["treatment"], [])
        self.assertTrue(result["message"])
        self.assertIn("only a class index", result["error"])
        with rose_app.get_db() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)

    def test_low_confidence_prediction_is_recorded_as_unidentified(self):
        class Uncertain:
            def predict(self, _image):
                return {
                    "class_name": None,
                    "confidence": None,
                    "probability_healthy": 0.483,
                    "probability_unhealthy": 0.517,
                }

        with patch.object(rose_app, "model_service", Uncertain()):
            response = self.client.post("/api/predict", data=self.image_upload())
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertFalse(result["identifiable"])
        self.assertEqual(result["status"], "unidentified")
        self.assertEqual(result["prediction"], "UNIDENTIFIED")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["disease_name"])
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["treatment"], [])
        self.assertTrue(result["message"])
        self.assertAlmostEqual(result["probability_healthy"], 0.483)
        self.assertAlmostEqual(result["probability_unhealthy"], 0.517)
        self.assertIn("uncertainty band", result["message"])
        with rose_app.get_db() as connection:
            row = connection.execute(
                "SELECT class_name, confidence, probability_healthy, probability_unhealthy "
                "FROM predictions WHERE id = ?",
                (result["id"],),
            ).fetchone()
        self.assertEqual(row["class_name"], "unidentified")
        self.assertAlmostEqual(row["confidence"], 0.517)
        self.assertAlmostEqual(row["probability_healthy"], 0.483)
        self.assertAlmostEqual(row["probability_unhealthy"], 0.517)
        self.assertEqual(len(list(self.upload_dir.iterdir())), 1)
        page = self.client.get(result["result_url"])
        self.assertIn(b"Prediction Uncertain", page.data)
        self.assertIn(b"Top model score, not a prediction", page.data)
        self.assertIn(b"Healthy score", page.data)
        self.assertIn(b"Unhealthy score", page.data)
        self.assertNotIn(b"Symptoms", page.data)
        history = self.client.get("/history")
        self.assertIn(b"top score", history.data)
        dashboard = self.client.get("/dashboard")
        self.assertIn(b"Uncertain results", dashboard.data)

    def test_higher_unhealthy_score_above_decision_threshold_is_predicted_unhealthy(self):
        class MarginalUnhealthy:
            def predict(self, _image):
                return {
                    "class_name": "unhealthy",
                    "confidence": 0.577,
                    "probability_healthy": 0.423,
                    "probability_unhealthy": 0.577,
                }

        with patch.object(rose_app, "model_service", MarginalUnhealthy()):
            response = self.client.post("/api/predict", data=self.image_upload())
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["status"], "unhealthy")
        self.assertEqual(result["prediction"], "UNHEALTHY")
        self.assertAlmostEqual(result["confidence"], 0.577)
        self.assertAlmostEqual(result["decision_threshold"], 0.5)
        self.assertTrue(result["treatment"])
        page = self.client.get(result["result_url"])
        self.assertIn(b"Unhealthy Rose Leaf", page.data)
        self.assertIn(b"57.7%", page.data)

    def test_threshold_calibration_requires_a_sufficient_balanced_dataset(self):
        threshold, calibrated = select_decision_threshold(
            [0, 1],
            [0.423, 0.577],
            {"healthy": 2, "unhealthy": 2},
            {"healthy": 1, "unhealthy": 1},
        )
        self.assertEqual(threshold, 0.5)
        self.assertFalse(calibrated)

        threshold, calibrated = select_decision_threshold(
            [0] * 20 + [1] * 20,
            [0.1] * 20 + [0.9] * 20,
            {"healthy": 100, "unhealthy": 100},
            {"healthy": 20, "unhealthy": 20},
        )
        self.assertEqual(threshold, 0.5)
        self.assertTrue(calibrated)

    def test_binary_evaluation_reports_confusion_matrix_and_class_metrics(self):
        evaluation = evaluate_binary_predictions(
            [0, 0, 1, 1],
            [0.1, 0.8, 0.7, 0.4],
            threshold=0.5,
        )
        self.assertEqual(evaluation["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(evaluation["confusion_matrix_labels"], ["healthy", "unhealthy"])
        self.assertEqual(evaluation["sample_count"], 4)
        self.assertEqual(evaluation["accuracy"], 0.5)
        self.assertEqual(evaluation["coverage"], 1.0)
        self.assertAlmostEqual(
            evaluation["classification_report"]["healthy"]["precision"],
            0.5,
        )
        self.assertAlmostEqual(
            evaluation["classification_report"]["unhealthy"]["recall"],
            0.5,
        )
        self.assertAlmostEqual(evaluation["macro_avg"]["f1_score"], 0.5)

        no_unhealthy_predictions = evaluate_binary_predictions(
            [0, 1],
            [0.2, 0.3],
            threshold=0.5,
        )
        self.assertEqual(
            no_unhealthy_predictions["classification_report"]["unhealthy"]["precision"],
            0.0,
        )
        self.assertEqual(
            no_unhealthy_predictions["classification_report"]["unhealthy"]["f1_score"],
            0.0,
        )

        borderline = evaluate_binary_predictions(
            [0, 1],
            [0.49, 0.51],
            threshold=0.5,
            abstention_margin=0.05,
        )
        self.assertEqual(borderline["uncertain_count"], 2)
        self.assertEqual(borderline["coverage"], 0.0)
        self.assertEqual(borderline["decided_confusion_matrix"], [[0, 0], [0, 0]])

    def test_model_service_reports_missing_and_invalid_artifacts(self):
        missing = ModelService(Path(self.temp_dir.name) / "missing")
        self.assertFalse(missing.status()["available"])
        with Image.new("RGB", (96, 96), (50, 80, 110)) as image:
            with self.assertRaises(ModelUnavailableError):
                missing.predict(image)

        model_dir = Path(self.temp_dir.name) / "invalid"
        model_dir.mkdir()
        (model_dir / "rose_disease_model.keras").touch()
        (model_dir / "class_names.json").write_text(
            '["healthy", "unhealthy"]',
            encoding="utf-8",
        )
        (model_dir / "training_metadata.json").write_text(
            '{"class_names":["healthy","unhealthy"],'
            '"class_mapping":{"0":"healthy","1":"unhealthy"},'
            '"image_size":[224,224]}',
            encoding="utf-8",
        )

        def fail_load(_path):
            raise ValueError("corrupt test model")

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(
                models=SimpleNamespace(load_model=fail_load),
            ),
        )
        invalid = ModelService(model_dir)
        with patch.dict(sys.modules, {"tensorflow": fake_tensorflow}):
            status = invalid.status()
        self.assertFalse(status["available"])
        self.assertIn("corrupt test model", status["message"])

    def test_invalid_model_results_are_rejected_before_persistence(self):
        class InvalidPrediction:
            def __init__(self, result):
                self.result = result

            def predict(self, _image):
                return self.result

        invalid_results = [
            (
                {"class_name": "other", "confidence": 0.9, "probability_unhealthy": 0.9},
                "unsupported result",
            ),
            (
                {"class_name": "healthy", "confidence": 0.9, "probability_unhealthy": 0.9},
                "inconsistent confidence",
            ),
        ]
        for result, message in invalid_results:
            with self.subTest(message=message), patch.object(
                rose_app, "model_service", InvalidPrediction(result)
            ):
                response = self.client.post("/api/predict", data=self.image_upload())
                self.assertEqual(response.status_code, 500)
                self.assertIn(message, response.get_json()["error"])
                with rose_app.get_db() as connection:
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
                        0,
                    )
                self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_model_service_matches_training_preprocessing_and_maps_binary_scores(self):
        class FakeCNN:
            input_shape = (None, 224, 224, 3)

            def predict(self, batch, verbose):
                self.batch = batch
                self.verbose = verbose
                return np.array([[self.score]], dtype=np.float32)

        model = FakeCNN()
        service = ModelService(Path(self.temp_dir.name))
        service.class_names = ["healthy", "unhealthy"]
        service.class_mapping = {0: "healthy", 1: "unhealthy"}
        service.output_format = "binary_probability"
        source_pixels = np.zeros((36, 48, 3), dtype=np.uint8)
        source_pixels[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
        source_pixels[:, :, 1] = np.arange(36, dtype=np.uint8)[:, None] * 7
        source_pixels[:, :, 2] = 90
        with patch.object(service, "_load", return_value=model), Image.fromarray(
            source_pixels
        ) as image:
            model.score = 0.2
            healthy = service.predict(image)
            model.score = 0.8
            unhealthy = service.predict(image)
            model.score = 0.517
            uncertain = service.predict(image)
            model.score = 0.577
            unhealthy_with_margin = service.predict(image)

        self.assertEqual(CLASS_NAMES, ["healthy", "unhealthy"])
        self.assertEqual(healthy["class_name"], "healthy")
        self.assertAlmostEqual(healthy["confidence"], 0.8)
        self.assertEqual(unhealthy["class_name"], "unhealthy")
        self.assertAlmostEqual(unhealthy["confidence"], 0.8)
        self.assertIsNone(uncertain["class_name"])
        self.assertIsNone(uncertain["confidence"])
        self.assertAlmostEqual(uncertain["probability_healthy"], 0.483)
        self.assertAlmostEqual(uncertain["probability_unhealthy"], 0.517)
        self.assertEqual(unhealthy_with_margin["class_name"], "unhealthy")
        self.assertAlmostEqual(unhealthy_with_margin["confidence"], 0.577)
        self.assertEqual(model.batch.shape, (1, 224, 224, 3))
        import tensorflow as tf

        expected = tf.image.resize(source_pixels, [224, 224], antialias=True).numpy()
        np.testing.assert_allclose(model.batch[0], expected, atol=1e-5)

    def test_model_service_normalizes_two_class_and_class_index_outputs(self):
        service = ModelService(Path(self.temp_dir.name))
        service.class_names = ["unhealthy", "healthy"]
        service.class_mapping = {0: "unhealthy", 1: "healthy"}

        service.output_format = "two_class_probabilities"
        result = service._normalize_output(
            [np.array([[0.15, 0.85]], dtype=np.float32)]
        )
        self.assertEqual(result["class_name"], "healthy")
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["prediction"], "HEALTHY")
        self.assertIn("disease_name", result)
        self.assertIn("symptoms", result)
        self.assertIn("treatment", result)
        self.assertIn("message", result)
        self.assertAlmostEqual(result["probability_healthy"], 0.85, places=6)
        self.assertAlmostEqual(result["probability_unhealthy"], 0.15, places=6)

        service.output_format = "class_index"
        result = service._normalize_output(
            {
                "class_index": np.array([0]),
                "confidence": np.array([0.85], dtype=np.float32),
            }
        )
        self.assertEqual(result["class_name"], "unhealthy")
        self.assertAlmostEqual(result["probability_unhealthy"], 0.85, places=6)
        with self.assertRaisesRegex(ModelOutputError, "no confidence score"):
            service._normalize_output(np.array([0], dtype=np.int32))

        service.output_format = "class_index_confidence"
        result = service._normalize_output(np.array([[1, 0.85]], dtype=np.float32))
        self.assertEqual(result["class_name"], "healthy")
        self.assertAlmostEqual(result["confidence"], 0.85, places=6)

    def test_saved_model_loads_declared_binary_output_and_mapping(self):
        service = ModelService(rose_app.MODEL_DIR)
        status = service.status()
        self.assertTrue(status["available"], status)
        self.assertEqual(status["output_format"], "binary_probability")
        self.assertEqual(status["class_mapping"], {0: "healthy", 1: "unhealthy"})
        self.assertEqual(status["decision_threshold"], 0.5)
        self.assertFalse(status["threshold_calibrated"])
        self.assertIn("not calibrated", status["warning"])
        self.assertIn("metrics", status)
        metadata = json.loads(
            (rose_app.MODEL_DIR / "training_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["backbone"], "mobilenetv2")
        self.assertEqual(
            metadata["preprocessing"]["backbone_rescaling"],
            {"scale": 2.0, "offset": -1.0},
        )
        for key in ("training_accuracy", "validation_accuracy", "training_loss", "validation_loss"):
            self.assertEqual(status["metrics"][key], metadata[key])
        self.assertEqual(status["metrics"]["reliable"], metadata["metrics_reliable"] is True)

    def test_legacy_multiclass_history_is_shown_only_as_unhealthy(self):
        image_path = self.upload_dir / "legacy.jpg"
        Image.new("RGB", (48, 36), (45, 130, 65)).save(image_path, format="JPEG")
        with rose_app.get_db() as connection:
            connection.execute(
                "INSERT INTO predictions (id, image_filename, class_name, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("legacy", image_path.name, "black_spot", 0.9, "2026-01-01T00:00:00+00:00"),
            )
        response = self.client.get("/result/legacy")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"UNHEALTHY", response.data)
        self.assertNotIn(b"Black Spot", response.data)


if __name__ == "__main__":
    unittest.main()
