# RoseGuard — AI-Based Rose Plant Health Screening

RoseGuard is an existing Flask application for screening one rose-leaf photo as **HEALTHY** or **UNHEALTHY** with a TensorFlow/Keras convolutional neural network (CNN). It provides photo upload and preview, prediction confidence, plant-care guidance, a SQLite-backed dashboard and history, and a repeatable training pipeline.

The model does not name individual diseases. An unhealthy result is a broad image screening result, not a diagnosis.

## Problem statement

Rose growers may notice leaf changes but not know whether to monitor a plant or seek help. RoseGuard offers an accessible first-look workflow that combines a leaf image with general care guidance, while making clear that an automated prediction is not a plant-health diagnosis.

## Objectives

- Accept a single rose-leaf image from a computer, phone gallery, or WhatsApp-saved image.
- Classify it using image pixels into exactly two classes: healthy or unhealthy.
- Show the image preview, original filename, result, confidence, and relevant care information.
- Keep prediction counts and recent history based on saved SQLite records.
- Provide an honest training workflow and show limitations when the dataset is too small.

## System architecture and modules

1. **Browser UI:** Flask/Jinja templates, responsive CSS, Bootstrap navigation, and JavaScript for file selection, signature checks, preview, drag-and-drop, and camera capture.
2. **Flask application (`app.py`):** Validates and decodes JPEG/PNG image bytes, invokes the model service, stores uploads and prediction records, and renders the home, detection, dashboard, history, and result pages.
3. **Model service (`model_service.py`):** Loads and validates the latest model, verifies class-order metadata, preprocesses RGB pixels, normalizes supported Keras probability outputs, and reports malformed or confidence-free output without inventing a prediction.
4. **Training (`train.py`):** Reads labeled train/validation folders, builds and trains the CNN, prints metrics and data-quality warnings, and saves the model and metadata.
5. **SQLite:** Stores the original uploaded name, generated image filename/location, binary result, confidence, and timestamp. Existing databases are migrated at application startup.

## Technologies

- Python 3.10–3.12, Flask, SQLite
- TensorFlow/Keras CNN, NumPy, Pillow
- HTML5, CSS3, JavaScript, Bootstrap 5

## Dataset structure

Preferred layout:

```text
dataset/
├── train/
│   ├── healthy/
│   └── unhealthy/
├── validation/
    ├── healthy/
    └── unhealthy/
└── test/                 # optional, independent final evaluation
    ├── healthy/
    └── unhealthy/
```

Class-directory names are case-insensitive. For compatibility with this workspace, training also accepts validation folders nested under `dataset/train/validation/` when `dataset/validation/` is absent. The trainer reads JPG/JPEG/PNG files, checks actual image contents and dimensions, reports unsupported image files, and fails on unreadable supported files rather than silently skipping them. Put only correctly labeled rose-leaf examples in the class folders. Keep images of the same leaf or plant in a single split and reserve separate images for final testing.

**Current workspace inventory:** `dataset/train/Healthy/` and `dataset/train/Unhealthy/` each contain 4 readable JPEGs; the nested `dataset/train/validation/Healthy/` and `dataset/train/validation/Unhealthy/` folders also each contain 4. That is 8 training and 8 validation images (16 total). The expected `dataset/validation/Healthy/` and `dataset/validation/Unhealthy/` directories are absent. The trainer finds all 16 images in the existing nested validation layout, and no corrupt supported images were found. Visual inspection found two training images in opposite-to-content class folders; those files were moved to the matching class directories before retraining. Validation examples were visually checked. There is no `dataset/test/`, so independent test accuracy cannot be reported.

The trainer uses a minimum pipeline floor of 100 training and 20 validation images per class. From the current counts, add at least 96 healthy and 96 unhealthy training images, plus 16 healthy and 16 unhealthy validation images. For an independent test, add at least 20 images per class under `dataset/test/`. These are minimum targets, not a guarantee of accuracy; use varied leaves, plants, lighting, and backgrounds, and keep the same plant out of all splits. The latest MobileNetV2 model was retrained on the 16 available images as a pipeline smoke test. Its measured training and validation accuracy are both 100% (losses 0.1655 and 0.3088), but these results are unreliable with only 4 images per class and no independent test set; overfitting is possible.

## CNN methodology and preprocessing

- By default, training uses a frozen ImageNet-pretrained MobileNetV2 feature extractor with a small binary classification head. Its RGB input is 224 × 224; embedded model layers rescale pixels from `[0, 255]` to `[0, 1]`, then to MobileNetV2's expected `[-1, 1]` range. The pretrained weights download once on the first training run. Use `--backbone cnn` for the regularized custom CNN if the weights are unavailable. Neither architecture can provide reliable results from the current tiny dataset.
- Training uses class labels `0 = healthy` and `1 = unhealthy`, binary cross-entropy, and accuracy.
- Training and inference both decode to RGB and use TensorFlow image resizing with antialiasing. Pixel values are rescaled to `[0, 1]` by the model's `Rescaling(1/255)` layer. Horizontal flips, rotations, zoom, and brightness changes are active only during training; validation images are not augmented. Convolution/dense L2 regularization and dropout are used to limit overfitting.
- Exact and near-identical images across train and validation are checked, and training stops if a split overlap is found.
- Model output interpretation is checked against `class_names.json` and the training metadata's explicit `class_mapping`. The service supports a single sigmoid probability, a two-class softmax probability vector, and class-index outputs that include confidence. Index-only outputs are rejected as unidentifiable instead of being assigned a made-up confidence.
- Successful `/api/predict` responses include `status`, `prediction`, `confidence`, `disease_name`, `symptoms`, `treatment`, and `message`. Because this project trains only healthy/unhealthy classes, `disease_name` is `null`; the app does not infer a specific disease.
- The sigmoid output is `P(unhealthy)` because training maps folder order `healthy=0`, `unhealthy=1`. The class decision uses a validation-selected threshold when at least 100 training and 20 validation images per class exist; otherwise it uses the neutral 0.50 threshold and explicitly reports that it is uncalibrated.
- A separate 0.05 probability band around the decision boundary is an abstention guard, not a class or rose-verification threshold. With the default threshold of 0.50, 42.3% healthy / 57.7% unhealthy is outside that band and returns **UNHEALTHY**. Scores very close to the boundary remain **UNCERTAIN**; this is a presentation state, not a third trained class. The threshold is currently uncalibrated because the validation set is too small.
- There is no separate rose-leaf-versus-other-image detector and no labeled non-rose dataset. The upload checks only file validity, dimensions, and blank images; valid non-rose photos may still receive a binary model result. Do not treat the model score as a calibrated probability or diagnosis.
- No filename-based or random predictions are used. Training supports an optional `dataset/test/` split and reports independent test metrics only when actual held-out test images are provided.
- Training prints training/validation accuracy and loss, a validation confusion matrix, per-class precision/recall/F1, macro averages, and an overfitting warning/assessment. These evaluations are also saved in `models/training_metadata.json`; when present, the optional held-out test split receives its own evaluation report.

## Installation

In Windows PowerShell, from the project directory:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is unavailable, install with the environment's Python directly:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the model

```powershell
py -3.12 train.py --data-dir dataset --epochs 20
```

The trainer reports per-class sample counts, train/validation accuracy and loss, validation confusion matrix, precision/recall/F1, and warnings for insufficient data, overlapping splits, possible overfitting, uncalibrated thresholds, or a missing independent test set. Relative paths resolve from the project directory. Training creates:

- `models/rose_disease_model.keras`
- `models/class_names.json`
- `models/training_metadata.json`

The current 16-image dataset is a pipeline test only. Do not treat its metrics or model predictions as reliable. The model was retrained to verify the pipeline; collect more correctly labeled, varied images before relying on predictions. MobileNetV2 transfer learning is not a substitute for the missing data or independent validation.

## Run RoseGuard

After installing dependencies and placing a trained model in `models/`:

```powershell
py -3.12 app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). To run the interface without a model, omit the model artifacts; the UI reports that predictions are unavailable and the API returns an error instead of fabricating a result.

## Expected application output

- **Home:** RoseGuard introduction and links to image detection and the dashboard.
- **Detection:** select, drag, or capture one JPG/JPEG/PNG. The selected image and its original filename are previewed before analysis.
- **Healthy result:** green status, confidence, and rose maintenance tips.
- **Unhealthy result:** red status, confidence, possible symptoms/causes, precautions, and general care/treatment suggestions. The app does not show specific disease names.
- **Unable to identify:** when the probabilities are too close to the decision boundary to support a reliable class result, the app shows both scores and photo guidance without claiming a result.
- **Dashboard:** total analyzed images, healthy, unhealthy, and abstention counts, graphical count bars, and recent checks computed from SQLite—not sample or hardcoded statistics.
- **History:** locally stored prior checks and original uploaded filenames.

Image types are identified from file contents, not a required basename or MIME label, so ordinary WhatsApp/gallery filenames are supported. Uploads are limited to 8 MB and stored under generated safe names in `static/uploads/`; the original basename is recorded for display. SQLite is stored in `instance/predictions.sqlite3`.

## Limitations and future enhancements

- A binary model cannot diagnose a specific disease or reliably distinguish disease from pests, nutrient stress, or environmental damage.
- Very small, biased, or duplicated datasets lead to unreliable predictions and confidence. The current model must be treated as a pipeline artifact only.
- Current uploads get basic readability, size, and blank-image checks. There is no separate rose-leaf-versus-unrelated-image detector because the dataset has no labeled non-rose examples; the binary model cannot reliably reject unrelated photos.
- In the latest API check, 14 of 16 available train/validation images received a class result and 2 were left uncertain; all 8 validation images were either classified correctly or abstained (6 classified, 2 uncertain). This is only an eight-image validation check, not evidence of reliable accuracy. The model must be evaluated on substantially more correctly labeled images from unseen plants before it is relied on.
- The binary classifier has no rose-leaf-versus-unrelated-photo detector. It cannot reliably identify unrelated images; a separately trained and evaluated rose-leaf detector plus representative negative examples would be required.
- Future improvements include a larger expert-labeled dataset, independent test sets, calibrated confidence, field evaluation across cultivars and lighting, and optional exportable reports.

## Tests

Run the existing automated test suite:

```powershell
py -3.12 -m unittest discover -s tests -v
```

Tests cover Flask pages/API, binary outcomes, upload validation, SQLite persistence/dashboard values, model loading failures, and preprocessing consistency. Passing software tests does not establish model accuracy.
