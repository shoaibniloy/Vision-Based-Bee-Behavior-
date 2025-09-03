
# Bee Vision – Real-Time Honeybee Sensing GUI  

A real-time PyQt6 application for **honeybee activity monitoring** powered by **YOLOv11**, **Ultralytics**, and **Supervision**.  
This tool detects bees, mites, pollen, queens, queen cells, and varroa mites, while tracking activity, behaviors, and anomalies.  

---

## 🚀 Getting Started  

### 1. Clone this repository  
```bash
git clone https://github.com/shoaibniloy/Vision-Based-Bee-Behavior.git
cd Vision-Based-Bee-Behavior
````

### 2. Install dependencies

It’s recommended to use a **Python 3.10+ virtual environment**.

```bash
pip install -r requirements.txt
```

**Required libraries include:**

* `PyQt6` (GUI)
* `opencv-python` (image processing)
* `ultralytics` (YOLOv11)
* `supervision` (tracking & visualization)
* `scikit-learn` (anomaly detection)
* `onnxruntime` *(optional, for behavior classification)*

---

## 🐝 Running the Application

1. Ensure you have a trained YOLOv11 model file named `bee.pt` in the project root.

   * Classes should include: `bee`, `mite`, `pollen`, `queen`, `queen_cell`, `varroa`.

2. Start the app:

```bash
python bee-vision.py
```

3. Default behavior:

   * Opens webcam (`0`) as video source.
   * You can switch to a video file via the **"Open File…"** option in the GUI.

---

## ✨ Features

* **Real-Time Detection & Tracking**

  * YOLOv11 object detection
  * ByteTrack multi-object tracking
  * Class aliasing for robust labeling

* **Bee Behavior Insights**

  * Entrance/exit counting
  * Behavior classification (via ONNX model or heuristics)
  * Queen bee tracking

* **Health & Anomaly Monitoring**

  * Dead bee detection
  * Pest (mite, varroa) statistics
  * Anomaly detection with **Isolation Forest**
  * Swarm risk estimation

* **Interactive GUI (PyQt6)**

  * Live annotated video feed
  * Metrics panel (bees, pests, queen sightings, behaviors, etc.)
  * Reset counters anytime

---

## 📂 Project Structure

```
Vision-Based-Bee-Behavior/
│── bee-vision.py       # Main application script
│── bee.pt              # YOLOv11 weights (user-supplied)
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## 🖼️ Example GUI

* **Left panel:** Live video stream with bounding boxes, IDs, and stats overlay.
* **Right panel:** Control panel, metrics log, and counters.

---

## ⚠️ Notes

* This project requires a **local YOLOv11 model (`bee.pt`)** trained on honeybee datasets.
* For **behavior recognition**, you may provide a trained ONNX model. If not available, heuristic fallbacks are used.
* Performance depends on hardware; GPU acceleration is recommended for smooth inference.

