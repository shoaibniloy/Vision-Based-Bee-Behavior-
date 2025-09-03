#!/usr/bin/env python3
"""
Bee Vision – Real-Time PyQt6 App (fixed & hardened)
Patched version: moves inference/capture to a worker thread, improves robustness,
adds safe shutdown, prunes stale tracks, and hardens inference wrapper.

This version loads a local YOLOv11 'bee.pt' model via Ultralytics and
uses class aliases for: bee, mite, pollen, queen, queen_cell, varroa.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QTextEdit,
    QCheckBox,
    QFileDialog,
)

# CV / ML
from ultralytics import YOLO
import supervision as sv
from sklearn.ensemble import IsolationForest

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False


# ---------------------------- Configuration ---------------------------------

IMG_SIZE = 640
CONF_DEFAULT = 0.25
IOU_THRESH = 0.45

VIDEO_SOURCE = 0  # 0 for default webcam; or set to a video file path

# Aliases built specifically for classes:
#   bee, mite, pollen, queen, queen_cell, varroa
# (We keep "pest_*" keys so existing "group.startswith('pest')" logic still works.)
CLASS_ALIASES: Dict[str, set[str]] = {
    "bee": {"bee", "worker", "worker_bee"},
    "queen": {"queen", "queen_bee"},
    "pest_mite": {"mite"},
    "pest_varroa": {"varroa", "varroa_mite"},
    "queen_cell": {"queen_cell", "queen_cup", "queen-cell"},
    "pollen": {"pollen", "pollen_basket", "pollen-basket"},
    # Note: no entries for capped/uncapped/honey_store here; those will remain 0 unless your model has them.
}

BEHAVIOR_LABELS = [
    "foraging",
    "guarding",
    "fanning",
    "grooming",
    "brood_care",
    "trembling",
]

GATE_Y_REL = 0.8
ROLL_SEC = 60
FPS_EST = 25


# --------------------------- YOLOv11 Wrapper --------------------------------

class YOLOv11Wrapper:
    """
    Thin adapter so the rest of the app can call model.infer(frame, conf=..., iou=...).
    Returns a supervision.Detections object and attaches per-detection class_name.
    """

    def __init__(self, weights_path: str = "bee.pt", img_size: int = 640):
        self.yolo = YOLO(weights_path)
        self.imgsz = img_size
        names = self.yolo.names
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names.keys())]
        else:
            self.class_names = list(names)

    # keep signature flexible to match the worker's try/except around kw names
    def infer(
        self,
        frame,
        confidence: float = 0.25,
        iou: float = 0.45,
        conf: float | None = None,
        **kwargs,
    ):
        if conf is not None:
            confidence = conf
        results = self.yolo.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=float(confidence),
            iou=float(iou),
            verbose=False,
        )
        det = sv.Detections.from_ultralytics(results[0])
        # Attach per-detection class names for UI logic
        try:
            det.data["class_name"] = np.array(
                [
                    self.class_names[int(i)] if 0 <= int(i) < len(self.class_names) else str(int(i))
                    for i in det.class_id
                ],
                dtype=object,
            )
        except Exception:
            pass
        # Also stash the full list
        det.data["class_names_list"] = self.class_names
        return det


# Load detector early so we can fail fast with a clear message
try:
    model = YOLOv11Wrapper(weights_path="bee.pt", img_size=IMG_SIZE)
except Exception as e:
    print(f"Error loading YOLOv11 weights from bee.pt: {e}")
    sys.exit(1)


# ------------------------------ Helper utils ---------------------------------

def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.5, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    h, w, ch = frame_bgr.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg)


# ------------------------------ Models / ML ----------------------------------

class BehaviorModel:
    def __init__(self, onnx_path: Optional[str] = None):
        self.enabled = False
        self.session = None
        if onnx_path and HAS_ORT:
            try:
                self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.enabled = True
            except Exception:
                self.enabled = False

    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        proc = []
        for c in crops:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_LINEAR)
            c = (c.astype(np.float32) / 255.0).transpose(2, 0, 1)
            proc.append(c)
        if not proc:
            return np.zeros((0, 3, 128, 128), dtype=np.float32)
        return np.stack(proc, axis=0)

    def infer(self, crops: List[np.ndarray]) -> List[str]:
        if not self.enabled or not crops:
            return []
        x = self.preprocess(crops)
        if x.shape[0] == 0:
            return []
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        idx = out.argmax(axis=1)
        return [BEHAVIOR_LABELS[i] if i < len(BEHAVIOR_LABELS) else f"class_{i}" for i in idx]


class AnomalyDetector:
    def __init__(self, window_seconds: int, fps: int):
        self.window_frames = max(30, window_seconds * max(1, fps))
        self.buffer = deque(maxlen=self.window_frames)
        self.model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.ready = False
        self.last_refit = 0.0

    def add_features(self, feat: np.ndarray) -> None:
        self.buffer.append(feat)
        if len(self.buffer) == self.buffer.maxlen and (time.time() - self.last_refit > 10):
            X = np.stack(self.buffer)
            try:
                self.model.fit(X)
                self.ready = True
                self.last_refit = time.time()
            except Exception:
                self.ready = False

    def score(self, feat: np.ndarray) -> float:
        if not self.ready:
            return 0.0
        s = -float(self.model.decision_function([feat])[0])
        return float(np.tanh(max(0.0, s)))


# ------------------------------ Worker Thread -------------------------------

class InferenceWorker(QThread):
    """Capture + inference run in this worker to keep UI responsive."""
    frame_ready = pyqtSignal(np.ndarray, object)  # annotated frame / raw result object (res)
    error = pyqtSignal(str)

    def __init__(self, src, model_obj, conf=0.25, iou=0.45, parent=None):
        super().__init__(parent)
        self.src = src
        self.model = model_obj
        self.conf = conf
        self.iou = iou
        self._stop = False
        self.cap = None

    def run(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                self.error.emit("Failed to open capture source.")
                return
            while not self._stop:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    # if file source, break; if webcam, continue
                    if isinstance(self.src, str):
                        break
                    time.sleep(0.01)
                    continue
                # Try multiple inference signatures to be robust
                res = None
                try:
                    try:
                        res = self.model.infer(frame, confidence=float(self.conf), iou=float(self.iou))
                    except TypeError:
                        # alternate kw names
                        try:
                            res = self.model.infer(frame, conf=float(self.conf), iou=float(self.iou))
                        except TypeError:
                            res = self.model.infer(frame)
                    except Exception as e:
                        self.error.emit(f"Inference error: {e}")
                        res = None
                except Exception as e:
                    self.error.emit(f"Inference exception: {e}")
                    res = None
                # emit original frame and raw result for main thread processing
                self.frame_ready.emit(frame, res)
                # small sleep to avoid tight loop if capture is very fast
                time.sleep(0.001)
        finally:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop = True
        self.wait(2000)


# ------------------------------ Main App ------------------------------------

@dataclass
class TrackMeta:
    last_center: Tuple[float, float] = (0.0, 0.0)
    last_side: Optional[str] = None
    behavior: str = ""
    last_seen_ts: float = 0.0


class BeeVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bee Vision – Honeybee Sensing GUI")
        self.resize(1280, 800)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.video_label = QLabel("Camera initializing…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        layout.addWidget(self.video_label, stretch=3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        layout.addWidget(right, stretch=2)

        ctrl_box = QGroupBox("Controls")
        ctrl_form = QFormLayout(ctrl_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam (0)", "Open File…"])
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 99)
        self.conf_spin.setValue(int(CONF_DEFAULT * 100))
        self.pause_chk = QCheckBox("Pause")
        self.reset_btn = QPushButton("Reset Counters")
        ctrl_form.addRow("Source", self.source_combo)
        ctrl_form.addRow("Conf %", self.conf_spin)
        ctrl_form.addRow(self.pause_chk)
        ctrl_form.addRow(self.reset_btn)
        right_layout.addWidget(ctrl_box)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        right_layout.addWidget(self.metrics_text, stretch=1)

        # State
        self.frame_w = 1280
        self.frame_h = 720
        self.gate_y = int(GATE_Y_REL * self.frame_h)
        self.model = model
        self.tracker = sv.ByteTrack()
        self.track_meta: Dict[int, TrackMeta] = {}
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.behavior_model = BehaviorModel(onnx_path=None)
        self.frame_idx = 0
        self.t0 = time.time()
        self.fps_est = FPS_EST
        self.rolling_counts: deque[int] = deque(maxlen=ROLL_SEC * self.fps_est)
        self.behavior_hist: Counter[str] = Counter()
        self.in_count = 0
        self.out_count = 0
        self.total_bees_seen = 0
        self.dead_bee_total = 0
        self.pest_hist: Counter[str] = Counter()
        self.queen_last_seen_time: Optional[float] = None
        self.anom = AnomalyDetector(window_seconds=ROLL_SEC, fps=self.fps_est)
        self.swarm_risk = 0.0

        self.reset_btn.clicked.connect(self.reset_counters)
        self.source_combo.currentIndexChanged.connect(self.on_source_change)

        # Worker: start default source
        self.worker: Optional[InferenceWorker] = None
        self.open_source(VIDEO_SOURCE)

        # UI refresh timer (lightweight)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_refresh)
        self.timer.start(100)  # update UI at 10 Hz

    def on_source_change(self, idx: int) -> None:
        if idx == 0:
            self.open_source(0)
        else:
            file, _ = QFileDialog.getOpenFileName(
                self, "Select video file", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file:
                self.open_source(file)
            else:
                self.source_combo.blockSignals(True)
                self.source_combo.setCurrentIndex(0)
                self.source_combo.blockSignals(False)
                self.open_source(0)

    def open_source(self, src) -> None:
        # stop old worker if any
        if self.worker:
            self.worker.stop()
            self.worker = None
        # spawn new worker
        conf = max(0.01, min(0.99, self.conf_spin.value() / 100.0))
        self.worker = InferenceWorker(src, self.model, conf=conf, iou=IOU_THRESH)
        self.worker.frame_ready.connect(self.on_frame_result)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

        # reset metrics appropriate for new source
        self.frame_idx = 0
        self.t0 = time.time()
        self.rolling_counts = deque(maxlen=ROLL_SEC * self.fps_est)
        self.anom = AnomalyDetector(window_seconds=ROLL_SEC, fps=self.fps_est)
        self.metrics_text.setPlainText("Source opened, waiting for frames...")

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.metrics_text.append(f"[Worker] {msg}")

    @pyqtSlot(np.ndarray, object)
    def on_frame_result(self, frame: np.ndarray, res) -> None:
        """Process one capture+inference result (runs on main thread)."""
        if self.pause_chk.isChecked():
            return
        if frame is None:
            return

        self.frame_idx += 1

        # Recompute dims
        self.frame_h, self.frame_w = frame.shape[:2]
        self.gate_y = int(GATE_Y_REL * self.frame_h)

        # Convert inference result robustly
        detections = self.results_to_detections(res)
        try:
            tracked = self.tracker.update_with_detections(detections)
        except Exception:
            # fallback: if tracker update fails, try converting detections to tracks or continue
            tracked = detections  # hope annotate still works; robust code below handles various types

        class_names_master = self.get_class_names(detections, res)

        labels: List[str] = []
        bees_now = 0
        dead_now = 0
        queen_now = 0
        pests_now: Counter[str] = Counter()
        brood_cells: Counter[str] = Counter()
        crop_map: Dict[int, np.ndarray] = {}

        # Helper: iterate tracked objects robustly
        tracked_len = 0
        try:
            tracked_len = len(tracked)
        except Exception:
            # if tracked has no len, try attribute (some supervision versions)
            try:
                tracked_len = len(getattr(tracked, "boxes", []))
            except Exception:
                tracked_len = 0

        seen_ids = set()

        for i in range(tracked_len):
            # robust extraction of bounding box
            xyxy = None
            try:
                xyxy = tracked.xyxy[i]
            except Exception:
                try:
                    xyxy = tracked.boxes[i].xyxy
                except Exception:
                    continue

            # confidence, class, id
            conf_i = 0.0
            try:
                conf_i = float(tracked.confidence[i]) if getattr(tracked, "confidence", None) is not None else 0.0
            except Exception:
                try:
                    conf_i = float(tracked.boxes[i].conf[i])
                except Exception:
                    conf_i = 0.0

            cls_i = -1
            try:
                cls_i = int(tracked.class_id[i]) if getattr(tracked, "class_id", None) is not None else -1
            except Exception:
                try:
                    cls_i = int(tracked.boxes[i].class_id)
                except Exception:
                    cls_i = -1

            tid = -1
            try:
                tid_val = tracked.tracker_id[i] if getattr(tracked, "tracker_id", None) is not None else None
                tid = int(tid_val) if tid_val is not None else -1
            except Exception:
                try:
                    tid = int(tracked.boxes[i].id)
                except Exception:
                    tid = -1
            seen_ids.add(tid)

            # determine name
            name = None
            try:
                if getattr(tracked, "data", None) and "class_name" in tracked.data:
                    arr = tracked.data["class_name"]
                    if isinstance(arr, (list, tuple, np.ndarray)) and i < len(arr):
                        name = arr[i]
            except Exception:
                name = None

            if name is None and class_names_master and 0 <= cls_i < len(class_names_master):
                name = class_names_master[cls_i]
            if not name:
                name = str(cls_i)

            # map to group
            group = None
            for k, vals in CLASS_ALIASES.items():
                if name in vals or name == k:
                    group = k
                    break
            if group == "bee":
                bees_now += 1
            elif group == "dead_bee":
                dead_now += 1
            elif group == "queen":
                queen_now += 1
                self.queen_last_seen_time = time.time()
            elif group and group.startswith("pest"):
                pests_now[group] += 1
            elif group in ("capped_cell", "uncapped_cell", "honey_store"):
                brood_cells[group] += 1

            # center & gate crossing
            try:
                x1, y1, x2, y2 = map(float, xyxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            except Exception:
                cx, cy = 0.0, 0.0

            tm = self.track_meta.get(tid, TrackMeta())
            side = "up" if cy < self.gate_y else "down"
            if tm.last_side and tm.last_side != side:
                if side == "up":
                    self.out_count += 1
                else:
                    self.in_count += 1
            tm.last_side = side
            tm.last_center = (cx, cy)
            tm.last_seen_ts = time.time()
            self.track_meta[tid] = tm

            # crops
            if group in ("bee", "queen"):
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                x1i = max(0, x1i)
                y1i = max(0, y1i)
                x2i = min(self.frame_w - 1, x2i)
                y2i = min(self.frame_h - 1, y2i)
                if x2i > x1i and y2i > y1i:
                    crop_map[tid] = frame[y1i:y2i, x1i:x2i].copy()

            labels.append(f"{name} {conf_i:.2f} id={tid}")

        # prune stale tracks (not seen for > 10s)
        now_ts = time.time()
        stale_keys = [k for k, v in self.track_meta.items() if now_ts - getattr(v, "last_seen_ts", now_ts) > 10.0]
        for k in stale_keys:
            self.track_meta.pop(k, None)

        # behavior inference
        behaviors_now: Counter[str] = Counter()
        if self.behavior_model.enabled and crop_map:
            tids = list(crop_map.keys())
            crops = [crop_map[t] for t in tids]
            preds = self.behavior_model.infer(crops)
            for tid, beh in zip(tids, preds):
                self.track_meta[tid].behavior = beh
                behaviors_now[beh] += 1
        else:
            # fallback heuristic
            for tid, tm in list(self.track_meta.items()):
                cx, cy = tm.last_center
                beh = "guarding" if abs(cy - self.gate_y) < 40 else "foraging"
                tm.behavior = beh
                behaviors_now[beh] += 1

        # features & anomaly
        pest_load = sum(pests_now.values())
        capped = brood_cells["capped_cell"]
        uncapped = brood_cells["uncapped_cell"]
        honey = brood_cells["honey_store"]
        brood_total = max(1, capped + uncapped + honey)
        brood_ratio = capped / brood_total
        activity = bees_now + 0.5 * (self.in_count + self.out_count) / max(1, self.frame_idx)
        feature_vec = np.array([bees_now, queen_now, dead_now, pest_load, brood_ratio, activity], dtype=np.float32)
        self.rolling_counts.append(bees_now)
        self.anom.add_features(feature_vec)
        anom_score = self.anom.score(feature_vec)

        bees_norm = np.tanh(bees_now / 50.0)
        flow_norm = np.tanh(abs(self.in_count - self.out_count) / max(1.0, self.frame_idx / max(1, self.fps_est)))
        pest_norm = np.tanh(pest_load / 5.0)
        self.swarm_risk = float(
            np.clip(0.3 * bees_norm + 0.2 * flow_norm + 0.2 * (1.0 - brood_ratio) + 0.3 * anom_score, 0.0, 1.0)
        )

        self.total_bees_seen += bees_now
        self.dead_bee_total += dead_now
        self.pest_hist.update(pests_now)
        self.behavior_hist.update(behaviors_now)

        # annotate
        try:
            annotated = self.box_annotator.annotate(scene=frame.copy(), detections=tracked)
            annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
        except Exception:
            annotated = frame.copy()

        cv2.line(annotated, (0, self.gate_y), (self.frame_w, self.gate_y), (0, 255, 255), 2)
        draw_text(annotated, f"Gate (entrance)", 10, max(20, self.gate_y - 10), 0.6, 2)

        hud = [
            f"FPS~{self.estimate_fps():.1f}",
            f"Bees:{bees_now} Queen:{queen_now} Dead:{dead_now}",
            f"In:{self.in_count} Out:{self.out_count}",
            f"Pests: {dict(pests_now)}",
            f"Brood: capped={capped} uncapped={uncapped} honey={honey}",
            f"SwarmRisk:{self.swarm_risk:.2f} Anomaly:{anom_score:.2f}",
        ]
        for i, line in enumerate(hud):
            draw_text(annotated, line, 10, 30 + i * 22, 0.6, 2)

        # update UI
        self.video_label.setPixmap(to_qpixmap(annotated))
        self.update_metrics_text(bees_now, queen_now, dead_now, pests_now, brood_cells, behaviors_now, anom_score)

    # name without underscore since caller uses results_to_detections(...)
    def results_to_detections(self, res) -> sv.Detections:
        try:
            # passthrough if it's already a Detections
            if isinstance(res, sv.Detections):
                return res
        except Exception:
            pass
        try:
            # if list with single dict from some APIs
            if isinstance(res, list) and len(res) == 1 and isinstance(res[0], dict):
                res = res[0]
            if isinstance(res, dict):
                return sv.Detections.from_inference(res)
        except Exception:
            pass
        return sv.Detections.empty()

    # alias kept in case other paths call the underscored version
    _results_to_detections = results_to_detections

    def get_class_names(self, detections: sv.Detections, res) -> List[str]:
        names: List[str] = []
        # 1) from detections data (preferred, we attach this in the wrapper)
        try:
            data = getattr(detections, "data", {}) or {}
            class_names_list = data.get("class_names_list")
            if isinstance(class_names_list, (list, tuple, np.ndarray)):
                return list(class_names_list)
        except Exception:
            pass
        # 2) fall back to per-detection class_name (less ideal)
        try:
            data = getattr(detections, "data", {}) or {}
            class_names = data.get("class_name")
            if class_names is not None:
                return list(class_names)
        except Exception:
            pass
        # 3) or from dict-like raw result if provided
        try:
            if isinstance(res, dict):
                for k in ("classes", "class_names", "names"):
                    cand = res.get(k)
                    if isinstance(cand, (list, tuple)):
                        return list(cand)
        except Exception:
            pass
        return names

    # alias to match any prior calls
    _get_class_names = get_class_names

    def estimate_fps(self) -> float:
        dt = time.time() - self.t0
        if dt < 1e-3:
            return float(self.fps_est)
        return self.frame_idx / dt

    def reset_counters(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self.total_bees_seen = 0
        self.dead_bee_total = 0
        self.pest_hist.clear()
        self.behavior_hist.clear()
        self.rolling_counts.clear()
        self.frame_idx = 0
        self.t0 = time.time()
        self.anom = AnomalyDetector(window_seconds=ROLL_SEC, fps=self.fps_est)
        self.metrics_text.append("\n[Reset] Counters and anomaly model reset.\n")

    def update_metrics_text(
        self,
        bees_now: int,
        queen_now: int,
        dead_now: int,
        pests_now: Counter[str],
        brood_cells: Counter[str],
        behaviors_now: Counter[str],
        anom_score: float,
    ) -> None:
        lines: List[str] = []
        lines.append("=== Current Frame ===")
        lines.append(f"Bees: {bees_now} | Queen: {queen_now} | Dead: {dead_now}")
        lines.append(f"Pests: {dict(pests_now)}")
        lines.append(f"Brood: {dict(brood_cells)}")
        lines.append(f"Behaviors: {dict(behaviors_now)}")
        lines.append(f"Entrance/Exit — In: {self.in_count} | Out: {self.out_count}")
        last_seen = time.strftime('%H:%M:%S', time.localtime(self.queen_last_seen_time)) if self.queen_last_seen_time else 'never'
        lines.append(f"Queen last seen: {last_seen}")
        lines.append("")
        lines.append("=== Totals / Rolling ===")
        lines.append(f"Total bees observed: {self.total_bees_seen}")
        lines.append(f"Dead bee cumulative: {self.dead_bee_total}")
        lines.append(f"Pest tally: {dict(self.pest_hist)}")
        lines.append(f"Behavior tally: {dict(self.behavior_hist)}")
        lines.append(f"Swarm risk: {self.swarm_risk:.2f} | Anomaly score: {anom_score:.2f}")
        self.metrics_text.setPlainText("\n".join(lines))

    def _on_refresh(self) -> None:
        # Could update lightweight, non-frame metrics here (e.g., rolling averages)
        pass

    def closeEvent(self, event) -> None:
        # cleanup worker & capture
        if self.worker:
            try:
                self.worker.stop()
            except Exception:
                pass
        super().closeEvent(event)


# ------------------------------- Entrypoint ----------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BeeVisionApp()
    w.show()
    sys.exit(app.exec())