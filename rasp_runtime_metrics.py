import os

# Must be set before importing torch or cv2.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import csv
import gc
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2
from ultralytics import YOLO
import ultralytics

from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


# ============================================================
# CONFIGURATION
# ============================================================

torch.set_num_threads(2)
cv2.setNumThreads(2)

MODEL_PATH = "/home/rasp/Downloads/AIExpo/8mMine30.pt"
FONT_PATH = "/usr/local/share/fonts/amiri/amiri-regular.ttf"

IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.4
MAX_DETECTIONS = 1

DETECTION_THRESHOLD = 1.0
DETECTION_TIMEOUT = 2.0
NO_SIGN_TIMEOUT = 4.0
INFERENCE_INTERVAL = 0.1

# Profiling settings
WARMUP_FRAMES = 20
MEMORY_SAMPLE_EVERY = 10

# Separate results folder for each model file.
PROFILE_DIR = (
    Path(MODEL_PATH).parent
    / f"profile_{Path(MODEL_PATH).stem}"
)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROFILING UTILITIES
# ============================================================

metric_samples = defaultdict(list)
frame_rows = []
output_event_rows = []

profiled_frames = 0
measurement_start = None


def record_metric(name, value, enabled=True):
    """Store numeric measurements for later statistical analysis."""
    if not enabled or value is None:
        return

    try:
        value = float(value)
    except (TypeError, ValueError):
        return

    if np.isfinite(value):
        metric_samples[name].append(value)


def read_process_memory():
    """
    Read current and peak resident memory from Linux /proc.
    Values are reported in MiB.
    """
    values = {}

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    values["rss_mib"] = (
                        float(line.split()[1]) / 1024.0
                    )
                elif line.startswith("VmHWM:"):
                    values["peak_rss_mib"] = (
                        float(line.split()[1]) / 1024.0
                    )
    except (OSError, ValueError, IndexError):
        pass

    return values


def calculate_statistics(values):
    """Return useful summary statistics for a collection of measurements."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def get_yolo_speed(results, key):
    """
    Read Ultralytics' per-image timing for preprocessing,
    forward inference and detection postprocessing.
    Returns None if unavailable.
    """
    try:
        speed = results[0].speed
        value = speed.get(key)
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def get_model_complexity(model):
    """Get parameter count and estimate GFLOPs at IMG_SIZE."""
    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    gflops = None

    # Preferred: use the model summary from Ultralytics.
    try:
        info = model.info(verbose=True, imgsz=IMG_SIZE)

        # Common return format: layers, parameters, gradients, GFLOPs.
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            candidate = float(info[3])
            if np.isfinite(candidate) and candidate > 0:
                gflops = candidate

    except Exception as error:
        print(f"Model summary warning: {error}")

    # Fallback for compatible Ultralytics versions.
    if gflops is None:
        try:
            from ultralytics.utils.torch_utils import get_flops

            candidate = float(
                get_flops(model.model, imgsz=IMG_SIZE)
            )

            if np.isfinite(candidate) and candidate > 0:
                gflops = candidate

        except Exception as error:
            print(f"GFLOPs estimation unavailable: {error}")

    return parameter_count, gflops


# ============================================================
# INITIALIZE MODEL AND RECORD COMPLEXITY
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH, task="detect")

parameter_count, gflops = get_model_complexity(model)

model_file_size_mib = os.path.getsize(MODEL_PATH) / (1024 ** 2)

try:
    inference_device = str(next(model.model.parameters()).device)
except (StopIteration, AttributeError):
    inference_device = "unknown"

print("\n========== MODEL COMPLEXITY ==========")
print(f"Model: {MODEL_PATH}")
print(f"Parameters: {parameter_count:,}")
print(f"Parameters (millions): {parameter_count / 1e6:.3f}")
print(
    f"GFLOPs at {IMG_SIZE}x{IMG_SIZE}: "
    f"{gflops:.3f}" if gflops is not None
    else "GFLOPs: unavailable"
)
print(f"Weight file size: {model_file_size_mib:.2f} MiB")
print(f"Model device at initialization: {inference_device}")
print("======================================\n")


# ============================================================
# INITIALIZE OLED AND CAMERA
# ============================================================

serial = i2c(port=1, address=0x3C)
oled_device = sh1106(serial)

arabic_font = ImageFont.truetype(FONT_PATH, 20)

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={
        "AfMode": 2,
        "AfSpeed": 1,
        "LensPosition": 0.0,
    },
)

picam2.configure(config)
picam2.start()


# ============================================================
# APPLICATION STATE
# ============================================================

sentence = ""
current_sequence = ""

last_label = None
last_accepted_output = None
last_output_type = None

label_accumulated_time = 0.0
last_detection_time = 0.0
last_inference_time = 0.0

no_sign_start_time = time.monotonic()
sentence_completed = False

# Used to measure latency until the sentence is displayed.
last_box_detected_perf = None
first_buffered_output_perf = None


# ============================================================
# TEXT AND DISPLAY FUNCTIONS
# ============================================================

def is_arabic_letter(label: str) -> bool:
    return len(label) == 1 and "\u0600" <= label <= "\u06FF"


def update_oled_display(text: str):
    """Render Arabic text to the SH1106 OLED."""
    if not text.strip():
        return

    reshaped_text = arabic_reshaper.reshape(text.strip())
    bidi_text = get_display(reshaped_text)

    with canvas(oled_device) as draw:
        draw.rectangle(
            oled_device.bounding_box,
            outline="black",
            fill="black",
        )
        draw.text(
            (0, 20),
            bidi_text[::-1],
            font=arabic_font,
            fill="white",
        )


# ============================================================
# MAIN PROFILING LOOP
# ============================================================

frame_id = 0

try:
    while True:
        # Preserve the configured inference pacing.
        pacing_time = time.monotonic()

        if pacing_time - last_inference_time < INFERENCE_INTERVAL:
            time.sleep(0.005)
            continue

        last_inference_time = pacing_time

        frame_id += 1
        is_measuring = frame_id > WARMUP_FRAMES

        frame_start = time.perf_counter()

        if is_measuring and measurement_start is None:
            measurement_start = frame_start

        # Initialize per-frame measurements.
        capture_ms = 0.0
        predict_wall_ms = 0.0
        temporal_filter_ms = 0.0
        sentence_buffer_ms = 0.0
        segmentation_ms = 0.0
        oled_update_ms = 0.0
        plot_ms = 0.0
        preview_ms = 0.0
        cleanup_ms = 0.0

        # ----------------------------------------------------
        # 1. CAMERA ACQUISITION
        # ----------------------------------------------------
        stage_start = time.perf_counter()
        frame = picam2.capture_array("main")
        capture_ms = (
            time.perf_counter() - stage_start
        ) * 1000.0

        # ----------------------------------------------------
        # 2. YOLO PREDICTION
        # ----------------------------------------------------
        # Ultralytics internally performs preprocessing,
        # model inference, and detection postprocessing.
        stage_start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

        predict_wall_ms = (
            time.perf_counter() - stage_start
        ) * 1000.0

        # These timings are internal components of predict_wall_ms.
        yolo_preprocess_ms = get_yolo_speed(
            results, "preprocess"
        )
        yolo_inference_ms = get_yolo_speed(
            results, "inference"
        )
        yolo_postprocess_ms = get_yolo_speed(
            results, "postprocess"
        )

        current_time = time.monotonic()

        boxes = results[0].boxes
        has_detection = (
            boxes is not None and len(boxes.cls) > 0
        )

        # ----------------------------------------------------
        # 3. TEMPORAL FILTERING AND TEXT BUFFERING
        # ----------------------------------------------------
        if has_detection:
            current_label = model.names[
                int(boxes.cls[0])
            ].strip()

            last_box_detected_perf = time.perf_counter()

            sentence_completed = False
            no_sign_start_time = current_time

            stage_start = time.perf_counter()

            # Accumulate the duration for consecutive detections
            # of the same label. Do not reset it on every frame.
            if current_label == last_label:
                delta = current_time - last_detection_time

                if delta <= DETECTION_TIMEOUT:
                    label_accumulated_time += max(delta, 0.0)
                else:
                    label_accumulated_time = 0.0
            else:
                last_label = current_label
                label_accumulated_time = 0.0

            # Only buffer an output after it remains stable long enough.
            if (
                label_accumulated_time >= DETECTION_THRESHOLD
                and current_label != last_accepted_output
            ):
                buffer_start = time.perf_counter()

                # Record when the first output token enters the buffer.
                if first_buffered_output_perf is None:
                    first_buffered_output_perf = buffer_start

                if is_arabic_letter(current_label):
                    if (
                        last_output_type == "word"
                        and sentence
                        and not sentence.endswith(" ")
                    ):
                        sentence += " "

                    current_sequence += current_label

                    print("Detected letter:", current_label)
                    last_output_type = "letter"

                else:
                    if current_sequence:
                        sentence += current_sequence
                        current_sequence = ""

                    if sentence and not sentence.endswith(" "):
                        sentence += " "

                    sentence += current_label

                    print("Detected word:", current_label)
                    last_output_type = "word"

                last_accepted_output = current_label

                sentence_buffer_ms = (
                    time.perf_counter() - buffer_start
                ) * 1000.0

            last_detection_time = current_time

            total_state_logic_ms = (
                time.perf_counter() - stage_start
            ) * 1000.0

            # Exclusive time: subtract the separately measured
            # sentence-buffering section to avoid double counting.
            temporal_filter_ms = max(
                0.0,
                total_state_logic_ms - sentence_buffer_ms,
            )

        # ----------------------------------------------------
        # 4. INACTIVITY-BASED SEGMENTATION
        # ----------------------------------------------------
        else:
            stage_start = time.perf_counter()

            if (
                current_time - no_sign_start_time
                >= NO_SIGN_TIMEOUT
                and not sentence_completed
            ):
                if current_sequence:
                    sentence += current_sequence
                    current_sequence = ""

                final_text = sentence.strip()

                if final_text:
                    print("Final sentence:", final_text)

                    oled_start = time.perf_counter()
                    update_oled_display(final_text)
                    oled_update_ms = (
                        time.perf_counter() - oled_start
                    ) * 1000.0

                    oled_completed_perf = time.perf_counter()

                    last_detection_to_oled_ms = None
                    if last_box_detected_perf is not None:
                        last_detection_to_oled_ms = (
                            oled_completed_perf
                            - last_box_detected_perf
                        ) * 1000.0

                    first_buffer_to_oled_ms = None
                    if first_buffered_output_perf is not None:
                        first_buffer_to_oled_ms = (
                            oled_completed_perf
                            - first_buffered_output_perf
                        ) * 1000.0

                    if is_measuring:
                        output_event_rows.append({
                            "event_id": len(output_event_rows) + 1,
                            "text": final_text,
                            "last_detection_to_oled_ms":
                                last_detection_to_oled_ms,
                            "first_buffered_output_to_oled_ms":
                                first_buffer_to_oled_ms,
                            "oled_update_ms": oled_update_ms,
                            "inactivity_timeout_s": NO_SIGN_TIMEOUT,
                        })

                        record_metric(
                            "oled_update_event_ms",
                            oled_update_ms,
                        )
                        record_metric(
                            "last_detection_to_oled_ms",
                            last_detection_to_oled_ms,
                        )
                        record_metric(
                            "first_buffered_output_to_oled_ms",
                            first_buffer_to_oled_ms,
                        )

                sentence_completed = True
                sentence = ""
                last_label = None
                label_accumulated_time = 0.0
                last_accepted_output = None
                last_output_type = None
                first_buffered_output_perf = None

            total_segmentation_ms = (
                time.perf_counter() - stage_start
            ) * 1000.0

            # Keep OLED transfer time separate from segmentation logic.
            segmentation_ms = max(
                0.0,
                total_segmentation_ms - oled_update_ms,
            )

        # ----------------------------------------------------
        # 5. PREVIEW RENDERING
        # ----------------------------------------------------
        stage_start = time.perf_counter()
        annotated_frame = results[0].plot()
        plot_ms = (
            time.perf_counter() - stage_start
        ) * 1000.0

        stage_start = time.perf_counter()
        cv2.imshow(
            "Sign Language Detection",
            annotated_frame,
        )
        key = cv2.waitKey(1) & 0xFF

        preview_ms = (
            time.perf_counter() - stage_start
        ) * 1000.0

        # ----------------------------------------------------
        # 6. CLEANUP
        # ----------------------------------------------------
        stage_start = time.perf_counter()

        del frame
        del annotated_frame
        del results

        gc.collect()

        cleanup_ms = (
            time.perf_counter() - stage_start
        ) * 1000.0

        # Capture-to-end-of-frame-processing latency.
        frame_pipeline_ms = (
            time.perf_counter() - frame_start
        ) * 1000.0

        # ----------------------------------------------------
        # 7. RECORD PROFILING DATA
        # ----------------------------------------------------
        if is_measuring:
            profiled_frames += 1

            record_metric("camera_capture_ms", capture_ms)
            record_metric("yolo_predict_wall_ms", predict_wall_ms)
            record_metric("temporal_filter_ms", temporal_filter_ms)
            record_metric("sentence_buffer_ms", sentence_buffer_ms)
            record_metric("inactivity_segmentation_ms", segmentation_ms)
            record_metric("oled_update_per_frame_ms", oled_update_ms)
            record_metric("plot_render_ms", plot_ms)
            record_metric("preview_display_ms", preview_ms)
            record_metric("cleanup_ms", cleanup_ms)
            record_metric(
                "frame_pipeline_latency_ms",
                frame_pipeline_ms,
            )

            record_metric(
                "yolo_preprocess_ms",
                yolo_preprocess_ms,
            )
            record_metric(
                "yolo_inference_ms",
                yolo_inference_ms,
            )
            record_metric(
                "yolo_postprocess_ms",
                yolo_postprocess_ms,
            )

            memory = {}
            if (
                profiled_frames == 1
                or profiled_frames % MEMORY_SAMPLE_EVERY == 0
            ):
                memory = read_process_memory()

                record_metric(
                    "current_rss_mib",
                    memory.get("rss_mib"),
                )
                record_metric(
                    "peak_rss_mib",
                    memory.get("peak_rss_mib"),
                )

            frame_rows.append({
                "frame_id": frame_id,
                "camera_capture_ms": capture_ms,
                "yolo_predict_wall_ms": predict_wall_ms,
                "yolo_preprocess_ms": yolo_preprocess_ms,
                "yolo_inference_ms": yolo_inference_ms,
                "yolo_postprocess_ms": yolo_postprocess_ms,
                "temporal_filter_ms": temporal_filter_ms,
                "sentence_buffer_ms": sentence_buffer_ms,
                "inactivity_segmentation_ms": segmentation_ms,
                "oled_update_ms": oled_update_ms,
                "plot_render_ms": plot_ms,
                "preview_display_ms": preview_ms,
                "cleanup_ms": cleanup_ms,
                "frame_pipeline_latency_ms": frame_pipeline_ms,
                "current_rss_mib": memory.get("rss_mib", ""),
                "peak_rss_mib": memory.get("peak_rss_mib", ""),
            })

            # Periodic console feedback.
            if profiled_frames % 100 == 0:
                elapsed = time.perf_counter() - measurement_start
                actual_fps = (
                    profiled_frames / elapsed
                    if elapsed > 0 else 0.0
                )

                memory_now = read_process_memory()

                print(
                    f"[PROFILE] Frames={profiled_frames}, "
                    f"actual FPS={actual_fps:.2f}, "
                    f"frame latency={frame_pipeline_ms:.2f} ms, "
                    f"RAM={memory_now.get('rss_mib', float('nan')):.1f} MiB"
                )

        if key == ord("q"):
            break


finally:
    # Capture run duration before hardware cleanup.
    profile_end = time.perf_counter()
    final_memory = read_process_memory()

    try:
        picam2.stop()
    except Exception:
        pass

    cv2.destroyAllWindows()

    try:
        oled_device.clear()
    except Exception:
        pass

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    frames_csv = PROFILE_DIR / "frame_timings.csv"
    events_csv = PROFILE_DIR / "oled_output_events.csv"
    summary_csv = PROFILE_DIR / "timing_summary.csv"
    summary_json = PROFILE_DIR / "run_summary.json"

    frame_fields = [
        "frame_id",
        "camera_capture_ms",
        "yolo_predict_wall_ms",
        "yolo_preprocess_ms",
        "yolo_inference_ms",
        "yolo_postprocess_ms",
        "temporal_filter_ms",
        "sentence_buffer_ms",
        "inactivity_segmentation_ms",
        "oled_update_ms",
        "plot_render_ms",
        "preview_display_ms",
        "cleanup_ms",
        "frame_pipeline_latency_ms",
        "current_rss_mib",
        "peak_rss_mib",
    ]

    with open(frames_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    event_fields = [
        "event_id",
        "text",
        "last_detection_to_oled_ms",
        "first_buffered_output_to_oled_ms",
        "oled_update_ms",
        "inactivity_timeout_s",
    ]

    with open(events_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(output_event_rows)

    runtime_seconds = (
        profile_end - measurement_start
        if measurement_start is not None
        else 0.0
    )

    actual_fps = (
        profiled_frames / runtime_seconds
        if runtime_seconds > 0
        else 0.0
    )

    mean_inference_ms = None
    if metric_samples["yolo_inference_ms"]:
        mean_inference_ms = float(
            np.mean(metric_samples["yolo_inference_ms"])
        )

    model_only_fps = (
        1000.0 / mean_inference_ms
        if mean_inference_ms and mean_inference_ms > 0
        else None
    )

    summary_stats = {
        name: calculate_statistics(values)
        for name, values in metric_samples.items()
    }

    # Include the final memory reading.
    summary_stats["final_memory_mib"] = final_memory

    metadata = {
        "model_path": MODEL_PATH,
        "model_file_size_mib": model_file_size_mib,
        "model_parameters": parameter_count,
        "model_parameters_millions": parameter_count / 1e6,
        "model_gflops_at_256": gflops,
        "model_device_at_initialization": inference_device,
        "input_camera_resolution": "640x480",
        "yolo_imgsz": IMG_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "inference_interval_s": INFERENCE_INTERVAL,
        "no_sign_timeout_s": NO_SIGN_TIMEOUT,
        "warmup_frames_excluded": WARMUP_FRAMES,
        "profiled_frames": profiled_frames,
        "runtime_seconds": runtime_seconds,
        "actual_system_fps": actual_fps,
        "approx_model_only_fps": model_only_fps,
        "hardware_platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "ultralytics_version": ultralytics.__version__,
        "metrics": summary_stats,
    }

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Compact statistical table for the manuscript.
    units = {
        "camera_capture_ms": "ms",
        "yolo_predict_wall_ms": "ms",
        "yolo_preprocess_ms": "ms",
        "yolo_inference_ms": "ms",
        "yolo_postprocess_ms": "ms",
        "temporal_filter_ms": "ms",
        "sentence_buffer_ms": "ms",
        "inactivity_segmentation_ms": "ms",
        "oled_update_per_frame_ms": "ms/frame",
        "oled_update_event_ms": "ms/event",
        "last_detection_to_oled_ms": "ms/event",
        "first_buffered_output_to_oled_ms": "ms/event",
        "plot_render_ms": "ms",
        "preview_display_ms": "ms",
        "cleanup_ms": "ms",
        "frame_pipeline_latency_ms": "ms",
        "current_rss_mib": "MiB",
        "peak_rss_mib": "MiB",
    }

    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        fields = [
            "metric", "unit", "count",
            "mean", "std", "median", "p95", "min", "max",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, stats in summary_stats.items():
            if "count" not in stats:
                continue

            writer.writerow({
                "metric": name,
                "unit": units.get(name, ""),
                **stats,
            })

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print("\n\n========== PROFILING SUMMARY ==========")
    print(f"Profiled frames: {profiled_frames}")
    print(f"Measurement duration: {runtime_seconds:.2f} s")
    print(f"Actual application FPS: {actual_fps:.2f}")

    if mean_inference_ms is not None:
        print(f"Mean YOLO forward inference: {mean_inference_ms:.2f} ms")

    if model_only_fps is not None:
        print(
            "Approx. model-only FPS (1000 / mean inference time): "
            f"{model_only_fps:.2f}"
        )

    print("\nMean / standard deviation / p95:")
    display_metrics = [
        "camera_capture_ms",
        "yolo_preprocess_ms",
        "yolo_inference_ms",
        "yolo_postprocess_ms",
        "yolo_predict_wall_ms",
        "temporal_filter_ms",
        "sentence_buffer_ms",
        "inactivity_segmentation_ms",
        "oled_update_event_ms",
        "last_detection_to_oled_ms",
        "frame_pipeline_latency_ms",
    ]

    for name in display_metrics:
        stats = summary_stats.get(name, {})

        if stats.get("count", 0) == 0:
            continue

        print(
            f"{name:38s} "
            f"mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}, "
            f"p95={stats['p95']:.3f}"
        )

    print("\nModel complexity:")
    print(f"Parameters: {parameter_count:,}")
    print(f"GFLOPs at {IMG_SIZE}: {gflops}")
    print(f"Model file size: {model_file_size_mib:.2f} MiB")

    print("\nMemory:")
    print(f"Final memory sample: {final_memory}")
    print(f"Profiling results saved to: {PROFILE_DIR}")
    print("=======================================\n")