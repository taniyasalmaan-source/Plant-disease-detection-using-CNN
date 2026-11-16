(() => {
  const input = document.getElementById("imageInput");
  const camera = document.getElementById("cameraInput");
  const dropZone = document.getElementById("dropZone");
  const previewWrap = document.getElementById("previewWrap");
  const preview = document.getElementById("imagePreview");
  const fileNameDisplay = document.getElementById("selectedFileName");
  const prompt = document.getElementById("uploadPrompt");
  const browse = document.getElementById("browseButton");
  const remove = document.getElementById("removeButton");
  const detect = document.getElementById("detectButton");
  const error = document.getElementById("uploadError");
  const loading = document.getElementById("loadingState");
  const notice = document.getElementById("modelNotice");
  const maxBytes = 8 * 1024 * 1024;
  let selectedFile = null;
  let previewUrl = null;
  let selectionId = 0;

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  function clearError() {
    error.textContent = "";
    error.hidden = true;
  }

  function clearPreview() {
    selectionId += 1;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    selectedFile = null;
    input.value = "";
    camera.value = "";
    preview.removeAttribute("src");
    fileNameDisplay.textContent = "";
    fileNameDisplay.hidden = true;
    previewWrap.hidden = true;
    prompt.hidden = false;
    remove.hidden = true;
    detect.disabled = true;
  }

  async function hasSupportedImageSignature(file) {
    const bytes = new Uint8Array(await file.slice(0, 8).arrayBuffer());
    const isJpeg = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
    const pngSignature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    const isPng = pngSignature.every((byte, index) => bytes[index] === byte);
    return isJpeg || isPng;
  }

  async function useFile(file) {
    clearError();
    if (!file) return;
    clearPreview();
    const currentSelection = selectionId;
    if (file.size === 0) {
      showError("This file is empty. Choose another image.");
      return;
    }
    if (file.size > maxBytes) {
      showError("Image must be 8 MB or smaller.");
      return;
    }
    let supportedFormat;
    try {
      supportedFormat = await hasSupportedImageSignature(file);
    } catch (_error) {
      if (currentSelection === selectionId) {
        showError("This file could not be read. Choose a JPG or PNG image.");
      }
      return;
    }
    if (currentSelection !== selectionId) return;
    if (!supportedFormat) {
      showError("Please choose a JPG or PNG image.");
      return;
    }
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    fileNameDisplay.textContent = file.name || "Uploaded image";
    fileNameDisplay.hidden = false;
    previewWrap.hidden = false;
    prompt.hidden = true;
    remove.hidden = false;
    detect.disabled = false;
  }

  browse.addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    input.click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => { void useFile(input.files[0]); });
  camera.addEventListener("change", () => { void useFile(camera.files[0]); });
  remove.addEventListener("click", () => {
    clearPreview();
    clearError();
    input.value = "";
    camera.value = "";
  });
  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => { void useFile(event.dataTransfer.files[0]); });
  window.addEventListener("beforeunload", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
  detect.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();
    notice.hidden = true;
    detect.disabled = true;
    loading.hidden = false;
    const data = new FormData();
    data.append("image", selectedFile);
    try {
      const response = await fetch("/api/predict", { method: "POST", body: data });
      const result = await response.json();
      if (!response.ok) {
        if (result && result.status === "unidentified" && result.error) {
          throw new Error(result.error);
        }
        throw new Error((result && result.error) || "We couldn't analyze that image.");
      }
      if (!result || typeof result !== "object") {
        console.error("Prediction API returned a non-object response:", result);
        throw new Error(
          "The prediction service returned incomplete result data. Restart Flask and refresh this page."
        );
      }
      const validStatus = ["healthy", "unhealthy", "unidentified"].includes(result.status);
      const classMatchesStatus =
        (result.status === "healthy" && result.class_name === "healthy") ||
        (result.status === "unhealthy" && result.class_name === "unhealthy") ||
        (result.status === "unidentified" && result.class_name === null);
      const confidenceIsValid =
        result.status === "unidentified"
          ? result.confidence === null
          : Number.isFinite(result.confidence) && result.confidence >= 0 && result.confidence <= 1;
      if (
        !result.result_url ||
        !validStatus ||
        result.prediction !== String(result.status).toUpperCase() ||
        typeof result.identifiable !== "boolean" ||
        result.identifiable !== (result.status !== "unidentified") ||
        !classMatchesStatus ||
        !confidenceIsValid ||
        !Object.prototype.hasOwnProperty.call(result, "disease_name") ||
        (result.disease_name !== null && typeof result.disease_name !== "string") ||
        !Array.isArray(result.symptoms) ||
        !Array.isArray(result.treatment) ||
        typeof result.message !== "string" ||
        !result.message.trim()
      ) {
        console.error("Prediction API returned an incompatible response:", result);
        throw new Error(
          "The prediction service returned incomplete result data. Restart Flask and refresh this page."
        );
      }
      window.location.assign(result.result_url);
    } catch (requestError) {
      showError(requestError.message);
      detect.disabled = false;
      loading.hidden = true;
    }
  });

  fetch("/api/model-status")
    .then((response) => response.json())
    .then((status) => {
      if (!status.available) {
        notice.textContent = `${status.message} You can still preview an image, but predictions will be unavailable until a model is trained.`;
        notice.hidden = false;
      } else if (status.warning || status.metrics) {
        const metrics = status.metrics
          ? `Measured accuracy (validation ${status.metrics.reliable ? "set" : "set; unreliable"}): training ${(status.metrics.training_accuracy * 100).toFixed(1)}%, validation ${(status.metrics.validation_accuracy * 100).toFixed(1)}%.`
          : "";
        notice.textContent = [status.warning, metrics].filter(Boolean).join(" ");
        notice.hidden = false;
      }
    })
    .catch(() => {
      notice.textContent = "Model status could not be checked. Try again shortly.";
      notice.hidden = false;
    });
})();
