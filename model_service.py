import json
import logging
import math
from pathlib import Path


CLASS_NAMES = ["healthy", "unhealthy"]
DEFAULT_DECISION_THRESHOLD = 0.50
DEFAULT_ABSTENTION_MARGIN = 0.05
LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "rose_disease_model.keras"
        self.labels_path = self.model_dir / "class_names.json"
        self.metadata_path = self.model_dir / "training_metadata.json"
        self._model = None
        self._load_error = None
        self._artifact_signature = None
        self.class_names = None
        self.class_mapping = None
        self.output_format = None
        self.decision_threshold = DEFAULT_DECISION_THRESHOLD
        self.abstention_margin = DEFAULT_ABSTENTION_MARGIN
        self.threshold_calibrated = False

    def _current_artifact_signature(self):
        signature = []
        for path in (self.model_path, self.labels_path, self.metadata_path):
            try:
                stat = path.stat()
            except OSError:
                return None
            signature.append((stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    @staticmethod
    def _single_shape(shape, name):
        if isinstance(shape, dict):
            if len(shape) != 1:
                raise ValueError("{} must have exactly one output.".format(name))
            shape = next(iter(shape.values()))
        elif (
            isinstance(shape, list)
            and shape
            and isinstance(shape[0], (list, tuple))
        ):
            if len(shape) != 1:
                raise ValueError("{} must have exactly one output.".format(name))
            shape = shape[0]
        try:
            return tuple(shape)
        except TypeError as error:
            raise ValueError("{} has an unsupported shape.".format(name)) from error

    def _load(self):
        signature = self._current_artifact_signature()
        if signature == self._artifact_signature:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise ModelUnavailableError(self._load_error)

        self._model = None
        self._load_error = None
        self.class_names = None
        self.class_mapping = None
        self.output_format = None
        self.decision_threshold = DEFAULT_DECISION_THRESHOLD
        self.abstention_margin = DEFAULT_ABSTENTION_MARGIN
        self.threshold_calibrated = False
        self._artifact_signature = signature

        if signature is None:
            self._load_error = (
                "The trained binary rose model, class labels, or training metadata are missing. "
                "Train the model before requesting predictions."
            )
            raise ModelUnavailableError(self._load_error)

        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(self.model_path)
            class_names = json.loads(self.labels_path.read_text(encoding="utf-8"))
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if (
                not isinstance(class_names, list)
                or len(class_names) != 2
                or set(class_names) != set(CLASS_NAMES)
                or metadata.get("class_names") != class_names
            ):
                raise ValueError(
                    "Model labels and training metadata must declare exactly healthy and "
                    "unhealthy in the same order. Retrain the binary model."
                )
            expected_mapping = {
                str(index): class_name for index, class_name in enumerate(class_names)
            }
            if metadata.get("class_mapping") != expected_mapping:
                raise ValueError(
                    "Model class mapping does not match the saved training configuration. "
                    "Retrain the model to regenerate its metadata."
                )
            decision_threshold = metadata.get(
                "decision_threshold", DEFAULT_DECISION_THRESHOLD
            )
            abstention_margin = metadata.get(
                "abstention_margin", DEFAULT_ABSTENTION_MARGIN
            )
            threshold_calibrated = metadata.get("threshold_calibrated", False)
            if (
                not isinstance(decision_threshold, (int, float))
                or isinstance(decision_threshold, bool)
                or not math.isfinite(decision_threshold)
                or not 0 <= decision_threshold <= 1
            ):
                raise ValueError("Training metadata contains an invalid decision threshold.")
            if (
                not isinstance(abstention_margin, (int, float))
                or isinstance(abstention_margin, bool)
                or not math.isfinite(abstention_margin)
                or not 0 <= abstention_margin <= 0.5
            ):
                raise ValueError("Training metadata contains an invalid abstention margin.")
            if not isinstance(threshold_calibrated, bool):
                raise ValueError("Training metadata must declare whether its threshold is calibrated.")

            input_shape = self._single_shape(model.input_shape, "Model input")
            if len(input_shape) != 4 or input_shape[-1] != 3:
                raise ValueError("Binary model must accept a batch of RGB images.")
            image_size = metadata.get("image_size")
            if (
                image_size is not None
                and len(input_shape) >= 3
                and input_shape[1] is not None
                and input_shape[2] is not None
                and list(input_shape[1:3]) != image_size
            ):
                raise ValueError(
                    "Model input dimensions do not match training metadata. Retrain the model."
                )
            preprocessing = metadata.get("preprocessing")
            if preprocessing is not None and (
                preprocessing.get("color_mode") != "RGB"
                or preprocessing.get("resize") != image_size
                or preprocessing.get("resize_antialias") is not True
                or preprocessing.get("rescale_in_model") is not True
                or not math.isclose(
                    float(preprocessing.get("rescale_factor", 0)),
                    1.0 / 255.0,
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )
            ):
                raise ValueError(
                    "Training metadata uses preprocessing that does not match the inference pipeline."
                )

            rescaling_layer = next(
                (
                    layer
                    for layer in model.layers
                    if isinstance(layer, tf.keras.layers.Rescaling)
                ),
                None,
            )
            if rescaling_layer is None or not math.isclose(
                float(rescaling_layer.scale),
                1.0 / 255.0,
                rel_tol=1e-5,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "Model must include the training pipeline's RGB 1/255 rescaling layer."
                )

            backbone_rescaling = (
                preprocessing.get("backbone_rescaling")
                if preprocessing is not None
                else None
            )
            if metadata.get("backbone") == "mobilenetv2" and not isinstance(
                backbone_rescaling, dict
            ):
                raise ValueError(
                    "MobileNetV2 metadata must declare its [-1, 1] input rescaling."
                )
            if backbone_rescaling is not None:
                scale = backbone_rescaling.get("scale")
                offset = backbone_rescaling.get("offset")
                if (
                    not isinstance(scale, (int, float))
                    or isinstance(scale, bool)
                    or not isinstance(offset, (int, float))
                    or isinstance(offset, bool)
                    or not math.isfinite(scale)
                    or not math.isfinite(offset)
                ):
                    raise ValueError("Training metadata contains invalid backbone rescaling.")
                has_matching_rescaling = any(
                    isinstance(layer, tf.keras.layers.Rescaling)
                    and math.isclose(
                        float(layer.scale),
                        float(scale),
                        rel_tol=1e-5,
                        abs_tol=1e-7,
                    )
                    and math.isclose(
                        float(layer.offset),
                        float(offset),
                        rel_tol=1e-5,
                        abs_tol=1e-7,
                    )
                    for layer in model.layers
                )
                if not has_matching_rescaling:
                    raise ValueError(
                        "The model backbone rescaling layer does not match training metadata."
                    )

            output_shape = self._single_shape(model.output_shape, "Model output")
            if len(output_shape) != 2:
                raise ValueError("Model output must contain one result per input image.")
            units = output_shape[-1]
            output_format = metadata.get("output_format")
            activation = getattr(
                getattr(model.layers[-1], "activation", None),
                "__name__",
                None,
            )
            if output_format is None:
                if units == 1 and activation == "sigmoid":
                    output_format = "binary_probability"
                elif units == 2 and activation == "softmax":
                    output_format = "two_class_probabilities"
                else:
                    raise ValueError(
                        "Could not determine the model output format from its final layer "
                        "and training metadata."
                    )

            valid_format = (
                output_format == "binary_probability"
                and units == 1
                and activation == "sigmoid"
            ) or (
                output_format == "two_class_probabilities"
                and units == 2
                and activation == "softmax"
            ) or (
                output_format == "class_index_confidence"
                and units == 2
                and activation == "linear"
            ) or (
                output_format == "class_index"
                and units == 1
                and activation in {"linear", None}
            )
            if not valid_format:
                raise ValueError(
                    "Model output layer does not match its declared training output format."
                )

            self._model = model
            self.class_names = class_names
            self.class_mapping = {
                int(index): class_name
                for index, class_name in expected_mapping.items()
            }
            self.output_format = output_format
            self.decision_threshold = float(decision_threshold)
            self.abstention_margin = float(abstention_margin)
            self.threshold_calibrated = threshold_calibrated
            LOGGER.info(
                "Loaded model %s (input=%s, output=%s, format=%s, mapping=%s, "
                "decision_threshold=%.4f, threshold_calibrated=%s)",
                self.model_path,
                input_shape,
                output_shape,
                output_format,
                self.class_mapping,
                self.decision_threshold,
                self.threshold_calibrated,
            )
            return model
        except ImportError as error:
            self._load_error = (
                "TensorFlow is not installed. Install requirements.txt before using predictions."
            )
            raise ModelUnavailableError(self._load_error) from error
        except Exception as error:
            self._load_error = "The rose model could not be loaded: {}".format(error)
            LOGGER.exception("Rose model loading failed")
            raise ModelUnavailableError(self._load_error) from error

    def status(self):
        try:
            self._load()
            status = {
                "available": True,
                "message": "Trained healthy/unhealthy model is ready.",
                "output_format": self.output_format,
                "class_mapping": self.class_mapping,
                "decision_threshold": self.decision_threshold,
                "abstention_margin": self.abstention_margin,
                "threshold_calibrated": self.threshold_calibrated,
            }
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            metric_keys = (
                "training_accuracy",
                "validation_accuracy",
                "training_loss",
                "validation_loss",
            )
            raw_metrics = {key: metadata.get(key) for key in metric_keys}
            has_metrics = any(value is not None for value in raw_metrics.values())
            valid_metrics = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in raw_metrics.values()
            )
            valid_metrics = valid_metrics and all(
                0 <= raw_metrics[key] <= 1
                for key in ("training_accuracy", "validation_accuracy")
            ) and all(raw_metrics[key] >= 0 for key in ("training_loss", "validation_loss"))
            if valid_metrics:
                status["metrics"] = {
                    **raw_metrics,
                    "reliable": metadata.get("metrics_reliable") is True,
                }
            elif has_metrics:
                LOGGER.warning("Training metadata contains invalid or incomplete model metrics.")
            if not metadata.get("metrics_reliable", False):
                warnings = metadata.get("warnings") or [
                    "Training data is too limited to establish reliable model performance."
                ]
                status["warning"] = " ".join(warnings)
            if not self.threshold_calibrated:
                status["warning"] = " ".join(
                    part
                    for part in (
                        status.get("warning"),
                        "The decision threshold is not calibrated; using the neutral 0.50 default.",
                    )
                    if part
                )
            return status
        except ModelUnavailableError as error:
            return {"available": False, "message": str(error)}
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            LOGGER.exception("Could not read model training metadata")
            return {
                "available": False,
                "message": "Training metadata could not be verified: {}".format(error),
            }

    @staticmethod
    def _numeric_array(value, message):
        import numpy as np

        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ModelOutputError(message) from error
        if not np.isfinite(values).all():
            raise ModelOutputError(message)
        return values

    def _normalize_output(self, output):
        import numpy as np

        class_index = None
        supplied_confidence = None
        if isinstance(output, dict):
            if "class_index" in output:
                class_index = self._numeric_array(
                    output["class_index"],
                    "The model returned an invalid class index.",
                )
                supplied_confidence = output.get("confidence")
            elif len(output) == 1:
                output = next(iter(output.values()))
            else:
                raise ModelOutputError(
                    "The model returned multiple named outputs that cannot be mapped "
                    "to one class result."
                )
        elif (
            isinstance(output, (list, tuple))
            and len(output) == 1
            and not np.isscalar(output[0])
        ):
            output = output[0]

        if class_index is None:
            values = self._numeric_array(
                output,
                "The model returned non-numeric prediction values.",
            )
            LOGGER.debug(
                "Raw model output type=%s shape=%s dtype=%s format=%s",
                type(output).__name__,
                values.shape,
                values.dtype,
                self.output_format,
            )
            if self.output_format == "binary_probability":
                if values.shape not in {(1,), (1, 1)}:
                    raise ModelOutputError(
                        "The binary model must return one probability for the uploaded image."
                    )
                positive_probability = float(values.reshape(-1)[0])
                if not math.isfinite(positive_probability) or not 0 <= positive_probability <= 1:
                    raise ModelOutputError("The binary model returned an invalid probability.")
                probabilities = {
                    self.class_mapping[0]: 1.0 - positive_probability,
                    self.class_mapping[1]: positive_probability,
                }
            elif self.output_format == "two_class_probabilities":
                if values.shape not in {(2,), (1, 2)}:
                    raise ModelOutputError(
                        "The two-class model must return two probabilities for the uploaded image."
                    )
                values = values.reshape(-1)
                if (
                    not np.isfinite(values).all()
                    or np.any(values < 0)
                    or np.any(values > 1)
                    or not math.isclose(float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
                ):
                    raise ModelOutputError(
                        "The two-class model returned invalid probabilities."
                    )
                probabilities = {
                    self.class_mapping[index]: float(values[index])
                    for index in range(2)
                }
            elif self.output_format == "class_index_confidence":
                if values.shape not in {(2,), (1, 2)}:
                    raise ModelOutputError(
                        "The class-index model must return one class index and one confidence."
                    )
                values = values.reshape(-1)
                class_index = values[:1]
                supplied_confidence = values[1:2]
            elif self.output_format == "class_index":
                if values.shape not in {(1,), (1, 1)}:
                    raise ModelOutputError(
                        "The class-index model returned an unsupported shape."
                    )
                class_index = values
            else:
                raise ModelOutputError(
                    "The model output format is not supported by this binary classifier."
                )

            if class_index is None:
                return self._prediction_from_probabilities(probabilities)

        class_index = np.asarray(class_index).reshape(-1)
        if class_index.shape != (1,):
            raise ModelOutputError("The model must return one class index per uploaded image.")
        raw_index = float(class_index[0])
        if not math.isfinite(raw_index) or raw_index not in self.class_mapping:
            raise ModelOutputError("The model returned a class index absent from its training map.")
        if supplied_confidence is None:
            raise ModelOutputError(
                "The model returned only a class index and no confidence score; unable to "
                "identify the leaf reliably."
            )
        confidence_values = self._numeric_array(
            supplied_confidence,
            "The class-index model returned an invalid confidence score.",
        ).reshape(-1)
        if confidence_values.shape != (1,):
            raise ModelOutputError(
                "The class-index model must return one confidence score per image."
            )
        confidence = float(confidence_values[0])
        selected_class = self.class_mapping[int(raw_index)]
        if not 0.5 <= confidence <= 1:
            raise ModelOutputError("The class-index model returned an invalid confidence score.")
        probabilities = {
            selected_class: confidence,
            next(name for name in self.class_names if name != selected_class): 1.0 - confidence,
        }
        return self._prediction_from_probabilities(probabilities)

    def _prediction_from_probabilities(self, probabilities):
        probability_healthy = probabilities["healthy"]
        probability_unhealthy = probabilities["unhealthy"]
        predicted_class = "unhealthy" if probability_unhealthy >= self.decision_threshold else "healthy"
        predicted_confidence = probabilities[predicted_class]
        class_name = (
            None
            if abs(probability_unhealthy - self.decision_threshold) < self.abstention_margin
            else predicted_class
        )
        status = class_name or "unidentified"
        message = (
            "The healthy and unhealthy scores are within the model's uncertainty band."
            if class_name is None
            else "The binary model predicts this rose leaf as {}.".format(class_name)
        )
        return {
            "status": status,
            "prediction": status.upper(),
            "class_name": class_name,
            "probability_healthy": probability_healthy,
            "probability_unhealthy": probability_unhealthy,
            "confidence": predicted_confidence if class_name is not None else None,
            "decision_threshold": self.decision_threshold,
            "abstention_margin": self.abstention_margin,
            "threshold_calibrated": self.threshold_calibrated,
            "disease_name": None,
            "symptoms": [],
            "treatment": [],
            "message": message,
        }

    def predict(self, image):
        model = self._load()
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as error:
            raise ModelUnavailableError(
                "TensorFlow and NumPy are required for image preprocessing. Install requirements.txt."
            ) from error

        input_shape = self._single_shape(model.input_shape, "Model input")
        height, width = input_shape[1:3]
        if height is None or width is None:
            height, width = 224, 224
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        pixels = tf.image.resize(pixels, [height, width], antialias=True).numpy()
        pixels = pixels.astype(np.float32, copy=False)
        try:
            output = model.predict(pixels[np.newaxis, ...], verbose=0)
        except Exception as error:
            LOGGER.exception("TensorFlow/Keras prediction execution failed")
            raise ModelOutputError("The model could not evaluate this image.") from error
        return self._normalize_output(output)
