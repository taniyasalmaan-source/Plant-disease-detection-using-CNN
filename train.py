import argparse
import hashlib
import json
import math
from pathlib import Path

import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


CLASS_NAMES = ["healthy", "unhealthy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OTHER_IMAGE_EXTENSIONS = {".bmp", ".gif", ".tif", ".tiff", ".webp"}
IMAGE_FORMATS = {"JPEG", "PNG"}
IMAGE_SIZE = (224, 224)
PROJECT_DIR = Path(__file__).resolve().parent
SEED = 42
MIN_TRAIN_IMAGES_PER_CLASS = 100
MIN_VALIDATION_IMAGES_PER_CLASS = 20
MIN_TEST_IMAGES_PER_CLASS = 20
DEFAULT_DECISION_THRESHOLD = 0.50
ABSTENTION_MARGIN = 0.05


def make_model(backbone="mobilenetv2"):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255)),
        ],
        name="augmentation",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rose_leaf_rgb")
    values = augmentation(inputs)
    values = tf.keras.layers.Rescaling(1.0 / 255, name="pixel_rescaling")(values)

    if backbone == "mobilenetv2":
        values = tf.keras.layers.Rescaling(
            2.0,
            offset=-1.0,
            name="mobilenet_v2_input_range",
        )(values)
        feature_extractor = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
            alpha=0.35,
        )
        feature_extractor.trainable = False
        values = feature_extractor(values, training=False)
        values = tf.keras.layers.GlobalAveragePooling2D()(values)
        values = tf.keras.layers.Dropout(0.35)(values)
        values = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(values)
        values = tf.keras.layers.Dropout(0.25)(values)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(values)
        return tf.keras.Model(inputs, outputs, name="rose_binary_mobilenetv2")

    if backbone != "cnn":
        raise ValueError("backbone must be 'mobilenetv2' or 'cnn'.")

    values = tf.keras.layers.Conv2D(
        24,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(values)
    values = tf.keras.layers.MaxPooling2D()(values)
    values = tf.keras.layers.Conv2D(
        48,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(values)
    values = tf.keras.layers.MaxPooling2D()(values)
    values = tf.keras.layers.Conv2D(
        96,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(values)
    values = tf.keras.layers.MaxPooling2D()(values)
    values = tf.keras.layers.SpatialDropout2D(0.2)(values)
    values = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(values)
    values = tf.keras.layers.GlobalAveragePooling2D()(values)
    values = tf.keras.layers.Dropout(0.4)(values)
    values = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(values)
    values = tf.keras.layers.Dropout(0.3)(values)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(values)
    return tf.keras.Model(inputs, outputs, name="rose_binary_cnn")


def resolve_class_directories(split_dir):
    if not split_dir.is_dir():
        raise ValueError("Dataset split directory does not exist: {}".format(split_dir))

    folders = {}
    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        folder_name = child.name.casefold()
        if folder_name in folders:
            raise ValueError("Duplicate class directory name (case-insensitive): {}".format(child.name))
        folders[folder_name] = child

    missing = [name for name in CLASS_NAMES if name not in folders]
    if missing:
        raise ValueError(
            "Missing class directories under {}: {}".format(split_dir, ", ".join(missing))
        )
    unexpected = sorted(set(folders) - set(CLASS_NAMES) - {"validation"})
    if unexpected:
        raise ValueError(
            "Only healthy and unhealthy class directories are supported; found: {}".format(
                ", ".join(unexpected)
            )
        )
    return [folders[name] for name in CLASS_NAMES]


def image_files(directory):
    images = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
            continue
        try:
            with Image.open(path) as image:
                if image.format in IMAGE_FORMATS:
                    images.append(path)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            continue
    return sorted(images)


def unsupported_image_files(class_directories):
    files = []
    for directory in class_directories:
        candidates = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in OTHER_IMAGE_EXTENSIONS
        ]
        for path in candidates:
            try:
                with Image.open(path) as image:
                    if image.format not in IMAGE_FORMATS:
                        files.append(path)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                files.append(path)
    return sorted(files)


def validate_image_files(class_directories):
    errors = []
    for directory in class_directories:
        for path in image_files(directory):
            try:
                with Image.open(path) as image:
                    image_format = image.format
                    width, height = image.size
                    image.verify()
                if image_format not in IMAGE_FORMATS:
                    errors.append("{}: unsupported image format {}".format(path, image_format))
                elif width < 64 or height < 64:
                    errors.append("{}: image is smaller than 64 x 64".format(path))
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                errors.append("{}: unreadable image ({})".format(path, error))
    if errors:
        raise ValueError("Dataset image validation failed:\n  " + "\n  ".join(errors))


def perceptual_hash(path):
    with Image.open(path) as image:
        small = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(small.getpixel((x, y)) > small.getpixel((x + 1, y)))
    return bits


def find_split_overlaps(train_directories, validation_directories):
    train_images = [
        path for directory in train_directories for path in image_files(directory)
    ]
    validation_images = [
        path for directory in validation_directories for path in image_files(directory)
    ]
    train_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest(): path
        for path in train_images
    }
    train_perceptual = [(path, perceptual_hash(path)) for path in train_images]
    overlaps = []
    for validation_path in validation_images:
        file_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
        if file_hash in train_hashes:
            overlaps.append((train_hashes[file_hash], validation_path, "identical file"))
            continue
        validation_hash = perceptual_hash(validation_path)
        for train_path, train_hash in train_perceptual:
            if (train_hash ^ validation_hash).bit_count() <= 4:
                overlaps.append((train_path, validation_path, "near-identical image"))
                break
    return overlaps


def select_decision_threshold(labels, unhealthy_probabilities, train_counts, validation_counts):
    if any(count < MIN_TRAIN_IMAGES_PER_CLASS for count in train_counts.values()):
        return DEFAULT_DECISION_THRESHOLD, False
    if any(count < MIN_VALIDATION_IMAGES_PER_CLASS for count in validation_counts.values()):
        return DEFAULT_DECISION_THRESHOLD, False
    if len(labels) != len(unhealthy_probabilities) or not labels:
        raise ValueError("Validation labels and model probabilities must have matching samples.")
    if set(labels) != {0, 1}:
        raise ValueError("Validation data must contain both binary classes.")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Validation labels must be 0 (healthy) or 1 (unhealthy).")
    if any(
        not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1
        for probability in unhealthy_probabilities
    ):
        raise ValueError("Validation probabilities must be finite values from 0 to 1.")

    unique_probabilities = sorted(set(float(value) for value in unhealthy_probabilities))
    candidates = {DEFAULT_DECISION_THRESHOLD, 0.0, 1.0}
    candidates.update(
        (left + right) / 2
        for left, right in zip(unique_probabilities, unique_probabilities[1:])
    )
    healthy_indices = [index for index, label in enumerate(labels) if label == 0]
    unhealthy_indices = [index for index, label in enumerate(labels) if label == 1]
    balanced_accuracies = {}
    for threshold in candidates:
        specificity = sum(
            unhealthy_probabilities[index] < threshold for index in healthy_indices
        ) / len(healthy_indices)
        sensitivity = sum(
            unhealthy_probabilities[index] >= threshold for index in unhealthy_indices
        ) / len(unhealthy_indices)
        balanced_accuracies[threshold] = (specificity + sensitivity) / 2

    best_accuracy = max(balanced_accuracies.values())
    best_threshold = min(
        (
            threshold
            for threshold, accuracy in balanced_accuracies.items()
            if math.isclose(accuracy, best_accuracy, rel_tol=0, abs_tol=1e-12)
        ),
        key=lambda threshold: (abs(threshold - DEFAULT_DECISION_THRESHOLD), threshold),
    )
    return float(best_threshold), True


def evaluate_binary_predictions(
    labels,
    unhealthy_probabilities,
    threshold,
    abstention_margin=ABSTENTION_MARGIN,
):
    if len(labels) != len(unhealthy_probabilities) or not labels:
        raise ValueError("Evaluation labels and probabilities must have matching samples.")
    if set(labels) - {0, 1} or set(labels) != {0, 1}:
        raise ValueError("Evaluation labels must contain both classes: 0 and 1.")
    if (
        not math.isfinite(float(threshold))
        or isinstance(threshold, bool)
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError("Evaluation threshold must be a finite value from 0 to 1.")
    if (
        not math.isfinite(float(abstention_margin))
        or isinstance(abstention_margin, bool)
        or not 0 <= float(abstention_margin) <= 0.5
    ):
        raise ValueError("Evaluation abstention margin must be between 0 and 0.5.")
    threshold = float(threshold)
    abstention_margin = float(abstention_margin)
    probabilities = [float(value) for value in unhealthy_probabilities]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Evaluation probabilities must be finite values from 0 to 1.")

    confusion = [[0, 0], [0, 0]]
    decided_confusion = [[0, 0], [0, 0]]
    uncertain_count = 0
    for label, probability in zip(labels, probabilities):
        predicted = int(probability >= threshold)
        confusion[int(label)][predicted] += 1
        if abs(probability - threshold) < abstention_margin:
            uncertain_count += 1
        else:
            decided_confusion[int(label)][predicted] += 1

    class_metrics = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[other][class_index]
            for other in range(2)
            if other != class_index
        )
        false_negative = sum(
            confusion[class_index][other]
            for other in range(2)
            if other != class_index
        )
        support = sum(confusion[class_index])
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        }

    total = sum(sum(row) for row in confusion)
    class_accuracy = (confusion[0][0] + confusion[1][1]) / total
    return {
        "sample_count": total,
        "accuracy": class_accuracy,
        "threshold": float(threshold),
        "confusion_matrix": confusion,
        "confusion_matrix_labels": CLASS_NAMES,
        "decided_confusion_matrix": decided_confusion,
        "uncertain_count": uncertain_count,
        "coverage": (total - uncertain_count) / total,
        "classification_report": class_metrics,
        "macro_avg": {
            key: sum(class_metrics[name][key] for name in CLASS_NAMES) / 2
            for key in ("precision", "recall", "f1_score")
        },
    }


def labels_for_directories(class_directories):
    return [
        class_index
        for class_index, directory in enumerate(class_directories)
        for _ in image_files(directory)
    ]


def print_evaluation(name, evaluation):
    confusion = evaluation["confusion_matrix"]
    print("\n{} classification evaluation (threshold {:.4f}):".format(
        name, evaluation["threshold"]
    ))
    print("Confusion matrix (rows=actual, columns=predicted; order=healthy, unhealthy):")
    print("  {}".format(confusion[0]))
    print("  {}".format(confusion[1]))
    print(
        "Prediction coverage after uncertainty abstention: {}/{} ({:.1%}); uncertain: {}".format(
            evaluation["sample_count"] - evaluation["uncertain_count"],
            evaluation["sample_count"],
            evaluation["coverage"],
            evaluation["uncertain_count"],
        )
    )
    print("Confusion matrix among decided predictions:")
    print("  {}".format(evaluation["decided_confusion_matrix"][0]))
    print("  {}".format(evaluation["decided_confusion_matrix"][1]))
    print("Class                 Precision  Recall  F1-score  Support")
    for class_name in CLASS_NAMES:
        metrics = evaluation["classification_report"][class_name]
        print(
            "{:<20} {:>9.3f} {:>7.3f} {:>9.3f} {:>8}".format(
                class_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["support"],
            )
        )
    print(
        "Macro average         {:>9.3f} {:>7.3f} {:>9.3f}".format(
            evaluation["macro_avg"]["precision"],
            evaluation["macro_avg"]["recall"],
            evaluation["macro_avg"]["f1_score"],
        )
    )


def assess_overfitting(training_accuracy, validation_accuracy, training_loss, validation_loss):
    accuracy_gap = float(training_accuracy - validation_accuracy)
    possible_overfitting = accuracy_gap >= 0.15 or (
        validation_loss > training_loss * 1.25
        and validation_loss - training_loss >= 0.1
    )
    return {
        "status": "possible_overfitting" if possible_overfitting else "no_large_gap_detected",
        "training_validation_accuracy_gap": accuracy_gap,
        "training_loss": float(training_loss),
        "validation_loss": float(validation_loss),
        "assessment_note": (
            "Training-history metrics include augmentation; this comparison is indicative only, "
            "and a very small validation set cannot reliably rule overfitting in or out."
        ),
    }


def build_dataset(class_directories, batch_size, shuffle):
    paths = []
    labels = []
    for label, directory in enumerate(class_directories):
        class_images = image_files(directory)
        paths.extend(str(path) for path in class_images)
        labels.extend([label] * len(class_images))

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        return image, tf.reshape(tf.cast(label, tf.float32), (1,))

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a binary healthy/unhealthy rose leaf classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--backbone",
        choices=("mobilenetv2", "cnn"),
        default="mobilenetv2",
        help="Use pretrained MobileNetV2 (downloads ImageNet weights once) or the local CNN.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")

    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    test_dir = data_dir / "test"
    if not validation_dir.is_dir():
        validation_dir = train_dir / "validation"
        if validation_dir.is_dir():
            print(
                "WARNING: dataset/validation is missing; using the existing nested "
                "validation directory: {}".format(validation_dir)
            )
    try:
        train_classes = resolve_class_directories(train_dir)
        validation_classes = resolve_class_directories(validation_dir)
        test_classes = resolve_class_directories(test_dir) if test_dir.is_dir() else None
    except ValueError as error:
        parser.error(str(error))

    train_counts = {
        class_name: len(image_files(directory))
        for class_name, directory in zip(CLASS_NAMES, train_classes)
    }
    validation_counts = {
        class_name: len(image_files(directory))
        for class_name, directory in zip(CLASS_NAMES, validation_classes)
    }
    test_counts = (
        {
            class_name: len(image_files(directory))
            for class_name, directory in zip(CLASS_NAMES, test_classes)
        }
        if test_classes
        else None
    )
    if any(count == 0 for count in train_counts.values()):
        parser.error("Both training classes must contain at least one supported image.")
    if any(count == 0 for count in validation_counts.values()):
        parser.error("Both validation classes must contain at least one supported image.")
    if test_counts and any(count == 0 for count in test_counts.values()):
        parser.error("Both test classes must contain at least one supported image.")

    try:
        all_class_directories = train_classes + validation_classes
        if test_classes:
            all_class_directories += test_classes
        validate_image_files(all_class_directories)
    except ValueError as error:
        parser.error(str(error))

    for path in unsupported_image_files(all_class_directories):
        print(
            "WARNING: unsupported image format is not used: {} "
            "(convert it to JPG, JPEG, or PNG)".format(path)
        )

    train_total = sum(train_counts.values())
    validation_total = sum(validation_counts.values())
    test_total = sum(test_counts.values()) if test_counts else 0
    split_pairs = [(train_classes, validation_classes)]
    if test_classes:
        split_pairs.extend(
            [
                (train_classes, test_classes),
                (validation_classes, test_classes),
            ]
        )
    overlaps = [
        overlap
        for first_split, second_split in split_pairs
        for overlap in find_split_overlaps(first_split, second_split)
    ]
    if overlaps:
        overlap_details = [
            "{} <-> {} ({})".format(train_path, validation_path, reason)
            for train_path, validation_path, reason in overlaps
        ]
        parser.error(
            "Training stopped to prevent train/validation leakage:\n  "
            + "\n  ".join(overlap_details)
        )

    warnings = []
    for class_name in CLASS_NAMES:
        train_count = train_counts[class_name]
        validation_count = validation_counts[class_name]
        if train_count < MIN_TRAIN_IMAGES_PER_CLASS:
            warnings.append(
                "{} has {} training image(s); add at least {} to reach the "
                "pipeline floor of {}.".format(
                    class_name,
                    train_count,
                    MIN_TRAIN_IMAGES_PER_CLASS - train_count,
                    MIN_TRAIN_IMAGES_PER_CLASS,
                )
            )
        if validation_count < MIN_VALIDATION_IMAGES_PER_CLASS:
            warnings.append(
                "{} has {} validation image(s); add at least {} to reach the "
                "pipeline floor of {}.".format(
                    class_name,
                    validation_count,
                    MIN_VALIDATION_IMAGES_PER_CLASS - validation_count,
                    MIN_VALIDATION_IMAGES_PER_CLASS,
                )
            )
        if test_counts and test_counts[class_name] < MIN_TEST_IMAGES_PER_CLASS:
            warnings.append(
            "{} has {} independent test image(s); add at least {} to reach the "
            "test-set floor of {}.".format(
                class_name,
                test_counts[class_name],
                MIN_TEST_IMAGES_PER_CLASS - test_counts[class_name],
                MIN_TEST_IMAGES_PER_CLASS,
            )
            )
    if not test_classes:
        warnings.append(
            "No dataset/test split exists; independent test accuracy cannot be measured."
        )
    threshold_calibration_ready = all(
        train_counts[class_name] >= MIN_TRAIN_IMAGES_PER_CLASS
        and validation_counts[class_name] >= MIN_VALIDATION_IMAGES_PER_CLASS
        for class_name in CLASS_NAMES
    )
    if not threshold_calibration_ready:
        warnings.append(
            "Decision-threshold calibration is unavailable with the current data; "
            "using the neutral 0.50 threshold."
        )
    print("Training class counts: {}".format(train_counts))
    print("Training image total: {}".format(train_total))
    print("Validation class counts: {}".format(validation_counts))
    print("Validation image total: {}".format(validation_total))
    print("Test class counts: {}".format(test_counts or "no independent test split"))
    print("Test image total: {}".format(test_total))
    print("Training directory: {}".format(train_dir))
    print("Validation directory: {}".format(validation_dir))
    print("Model backbone: {}".format(args.backbone))
    for warning in warnings:
        print("WARNING: {}".format(warning))

    tf.keras.utils.set_random_seed(SEED)
    train_ds = build_dataset(train_classes, args.batch_size, shuffle=True)
    train_eval_ds = build_dataset(train_classes, args.batch_size, shuffle=False)
    val_ds = build_dataset(validation_classes, args.batch_size, shuffle=False)
    test_ds = (
        build_dataset(test_classes, args.batch_size, shuffle=False)
        if test_classes
        else None
    )

    model = make_model(args.backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "rose_disease_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
    )
    best_epoch_index = min(
        range(len(history.history["val_loss"])),
        key=lambda index: history.history["val_loss"][index],
    )
    best_epoch = best_epoch_index + 1
    train_loss, train_accuracy = model.evaluate(train_eval_ds, verbose=0)
    validation_loss, validation_accuracy = model.evaluate(val_ds, verbose=0)
    train_labels = labels_for_directories(train_classes)
    validation_labels = labels_for_directories(validation_classes)
    train_probabilities = model.predict(train_eval_ds, verbose=0).reshape(-1).tolist()
    validation_probabilities = model.predict(val_ds, verbose=0).reshape(-1).tolist()
    try:
        decision_threshold, threshold_calibrated = select_decision_threshold(
            validation_labels,
            validation_probabilities,
            train_counts,
            validation_counts,
        )
    except ValueError as error:
        parser.error(str(error))
    if test_ds is not None:
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        test_labels = labels_for_directories(test_classes)
        test_probabilities = model.predict(test_ds, verbose=0).reshape(-1).tolist()
        test_loss = float(test_loss)
        test_accuracy = float(test_accuracy)
    else:
        test_loss = None
        test_accuracy = None
        test_labels = None
        test_probabilities = None
    training_evaluation = evaluate_binary_predictions(
        train_labels, train_probabilities, decision_threshold
    )
    validation_evaluation = evaluate_binary_predictions(
        validation_labels, validation_probabilities, decision_threshold
    )
    test_evaluation = (
        evaluate_binary_predictions(test_labels, test_probabilities, decision_threshold)
        if test_ds is not None
        else None
    )
    overfitting_assessment = assess_overfitting(
        history.history["accuracy"][best_epoch_index],
        history.history["val_accuracy"][best_epoch_index],
        history.history["loss"][best_epoch_index],
        history.history["val_loss"][best_epoch_index],
    )
    if overfitting_assessment["status"] == "possible_overfitting":
        warnings.append(
            "Training metrics suggest possible overfitting; the small validation set cannot "
            "confirm this reliably."
        )
    model.save(model_path)
    (output_dir / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "class_names": CLASS_NAMES,
        "class_mapping": {"0": "healthy", "1": "unhealthy"},
        "output_format": "binary_probability",
        "backbone": args.backbone,
        "pretrained_weights": "imagenet" if args.backbone == "mobilenetv2" else None,
        "image_size": list(IMAGE_SIZE),
        "preprocessing": {
            "color_mode": "RGB",
            "resize": list(IMAGE_SIZE),
            "resize_antialias": True,
            "rescale_in_model": True,
            "rescale_factor": 1.0 / 255.0,
            "backbone_rescaling": (
                {"scale": 2.0, "offset": -1.0}
                if args.backbone == "mobilenetv2"
                else None
            ),
        },
        "training_images": train_total,
        "validation_images": validation_total,
        "minimum_training_images_per_class": MIN_TRAIN_IMAGES_PER_CLASS,
        "minimum_validation_images_per_class": MIN_VALIDATION_IMAGES_PER_CLASS,
        "minimum_test_images_per_class": MIN_TEST_IMAGES_PER_CLASS,
        "training_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "test_class_counts": test_counts,
        "overlapping_images_between_splits": False,
        "metrics_reliable": not warnings,
        "decision_threshold": decision_threshold,
        "threshold_calibrated": threshold_calibrated,
        "abstention_margin": ABSTENTION_MARGIN,
        "training_accuracy": float(train_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "training_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "best_epoch": best_epoch,
        "training_evaluation": training_evaluation,
        "validation_evaluation": validation_evaluation,
        "test_images": test_total,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_evaluation": test_evaluation,
        "overfitting_assessment": overfitting_assessment,
        "warnings": warnings,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nAccuracy/loss for the saved model weights:")
    print("Training accuracy:   {:.4f}".format(train_accuracy))
    print("Validation accuracy: {:.4f}".format(validation_accuracy))
    print("Training loss:       {:.4f}".format(train_loss))
    print("Validation loss:     {:.4f}".format(validation_loss))
    print("Best epoch by validation loss: {}".format(best_epoch))
    print_evaluation("Training", training_evaluation)
    print_evaluation("Validation", validation_evaluation)
    if test_ds is not None:
        print("Independent test accuracy: {:.4f}".format(test_accuracy))
        print("Independent test loss:     {:.4f}".format(test_loss))
        print_evaluation("Independent test", test_evaluation)
    else:
        print("Independent test accuracy: not measured (dataset/test is missing).")
    print("Overfitting assessment: {}".format(overfitting_assessment["status"]))
    print(overfitting_assessment["assessment_note"])
    print(
        "Unhealthy decision threshold: {:.4f} ({})".format(
            decision_threshold,
            "validation-selected" if threshold_calibrated else "uncalibrated default",
        )
    )
    print("Saved binary model to {}.".format(model_path))
    print("Class mapping: 0 = healthy, 1 = unhealthy.")
    if warnings:
        print(
            "WARNING: These metrics are not reliable. Collect many varied images per class, "
            "keep plants separate between splits, and reserve an independent test set."
        )


if __name__ == "__main__":
    main()
