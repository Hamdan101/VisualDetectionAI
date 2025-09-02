# filename: gate_line_alarm.py
# Usage:
#   python3 gate_line_alarm.py --select
#   python3 gate_line_alarm.py --camera 0
#
# First run (no line_calibration.json):
#   - Click TWO points for the line.
#   - Preview shows shaded "alert" side.
#   - Press F to flip alert side, ENTER to save, R to redo, Q to cancel.
#
# Features:
#   • Boxes turn RED when footpoint is on alert side of the line.
#   • Top-right shows People (total) and Inside (in red zone).
#   • LOUD alarm loops while Inside > 0; stops immediately when Inside == 0.
#   • Snapshot to ./imgs/ on entry (one per entry per person).
#   • Timer label per RED box: "in: X.XXs".
#   • Lightweight tracker (footpoint distance matching).

import os, json, argparse, time, math, subprocess, platform, shutil
import numpy as np
import cv2
from ultralytics import YOLO
import threading

alarm_thread = None
alarm_stop = threading.Event()
CALIB_FILE = "line_calibration.json"
MODEL_NAME = "yolov8n.pt"   # tiny, person class = 0
CONF_THRES = 0.35
IMG_SIZE   = 640
ALARM_FILE = "alarm.mp3"    # put a loud siren here

# ------------- Camera helpers -------------
def _cap_backend():
    return getattr(cv2, "CAP_AVFOUNDATION", 0)

def try_open(idx, w=1280, h=720):
    cap = cv2.VideoCapture(idx, _cap_backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def pick_camera(max_probe=8):
    tried = set(); idx = 0
    while len(tried) < max_probe:
        cap = None; start = idx
        while True:
            if idx in tried:
                idx = (idx + 1) % max_probe
                if idx == start: break
                continue
            tried.add(idx)
            cap = try_open(idx)
            if cap is not None: break
            idx = (idx + 1) % max_probe
            if idx == start: break
        if cap is None:
            print("No cameras found."); return None, None

        print(f"Previewing camera index {idx}. SPACE=select, N=next, Q=quit")
        while True:
            ok, frame = cap.read()
            if not ok: break
            cv2.putText(frame, f"Camera index {idx}  [SPACE=select, N=next, Q=quit]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("camera picker", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord(' '), 13):  # space/enter
                cv2.destroyWindow("camera picker"); return cap, idx
            if k in (ord('n'), ord('N')): break
            if k in (ord('q'), ord('Q'), 27):
                cap.release(); cv2.destroyWindow("camera picker"); return None, None
        cap.release(); cv2.destroyWindow("camera picker")
    print("Finished probing cameras."); return None, None

def open_camera(args):
    if args.select:
        cap, idx = pick_camera()
        if cap is None: raise SystemExit("No camera selected.")
        print(f"Selected camera index {idx}"); return cap
    cap = try_open(args.camera)
    if cap is None: raise SystemExit(f"Could not open camera index {args.camera}. Try --select.")
    print(f"Using camera index {args.camera}"); return cap

# ------------- Line calibration (2 clicks + flip) -------------
_clicks = []
def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))

def save_line(p1, p2, side_sign):
    with open(CALIB_FILE, "w") as f:
        json.dump({"p1": p1, "p2": p2, "side_sign": side_sign}, f)

def load_line():
    if not os.path.exists(CALIB_FILE): return None
    with open(CALIB_FILE) as f:
        d = json.load(f)
    p1 = tuple(map(int, d["p1"]))
    p2 = tuple(map(int, d["p2"]))
    side_sign = float(d["side_sign"])
    return p1, p2, side_sign

def point_side(p1, p2, pt):
    v1 = np.array([pt[0]-p1[0], pt[1]-p1[1]], dtype=np.float64)
    v2 = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=np.float64)
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return np.sign(cross)  # -1, 0, +1

def draw_line_overlay(frame, p1, p2, side_sign, shade_alpha=0.25):
    cv2.line(frame, p1, p2, (255,255,255), 2)
    cv2.putText(frame, "Trigger line", (p2[0]+6, p2[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    n = np.array([-dy, dx], dtype=np.float64)
    if np.linalg.norm(n) < 1e-6: return
    n = n / np.linalg.norm(n)
    test_pt = (int(p1[0] + n[0]*5), int(p1[1] + n[1]*5))
    if point_side(p1, p2, test_pt) != np.sign(side_sign):
        n = -n
    K = 5000.0
    p1_far = (int(p1[0] + n[0]*K), int(p1[1] + n[1]*K))
    p2_far = (int(p2[0] + n[0]*K), int(p2[1] + n[1]*K))
    overlay = frame.copy()
    poly = np.array([p1, p2, p2_far, p1_far], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], (0,0,255))
    cv2.addWeighted(overlay, shade_alpha, frame, 1-shade_alpha, 0, dst=frame)

def calibrate_line_two_clicks(cap):
    global _clicks
    _clicks = []
    print("Calibration: click TWO points for the line. [Q=abort]")
    while True:
        ok, frame = cap.read()
        if not ok: print("Camera read failed."); return None
        view = frame.copy()
        cv2.putText(view, "Click P1 & P2 (line). Q=abort",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        for i, p in enumerate(_clicks[:2]):
            cv2.circle(view, p, 6, (0,255,0), -1)
            cv2.putText(view, "P1" if i==0 else "P2", (p[0]+6, p[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("calibrate-line", view)
        cv2.setMouseCallback("calibrate-line", _on_click)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow("calibrate-line"); return None
        if len(_clicks) >= 2:
            p1 = _clicks[0]; p2 = _clicks[1]
            side_sign = 1.0  # default; user can flip
            # Preview & flip/save
            while True:
                ok, frame2 = cap.read()
                if not ok: break
                prev = frame2.copy()
                draw_line_overlay(prev, p1, p2, side_sign, shade_alpha=0.25)
                cv2.putText(prev, "ENTER=save, F=flip side, R=redo, Q=cancel",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("calibrate-line", prev)
                k2 = cv2.waitKey(1) & 0xFF
                if k2 in (ord('f'), ord('F')):
                    side_sign = -side_sign
                elif k2 in (ord('r'), ord('R')):
                    _clicks = []
                    break
                elif k2 in (13,):  # Enter
                    save_line(p1, p2, side_sign)
                    cv2.destroyWindow("calibrate-line")
                    print(f"Saved {CALIB_FILE}: p1={p1}, p2={p2}, side_sign={int(side_sign)}")
                    return (p1, p2, side_sign)
                elif k2 in (ord('q'), ord('Q'), 27):
                    cv2.destroyWindow("calibrate-line")
                    return None

# ------------- Simple tracker (footpoint matching) -------------
class Track:
    _next_id = 1
    def __init__(self, foot_px, bbox):
        self.id = Track._next_id; Track._next_id += 1
        self.foot = foot_px
        self.bbox = bbox
        self.last_seen = time.time()
        self.inside = False
        self.enter_time = None
        self.snap_taken = False

def match_tracks(tracks, detections, max_dist=80):
    now = time.time()
    used_det = set()
    for t in tracks:
        best_i, best_d = None, 1e9
        for i, (fpt, bbox) in enumerate(detections):
            if i in used_det: continue
            d = math.hypot(fpt[0] - t.foot[0], fpt[1] - t.foot[1])
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= max_dist:
            fpt, bbox = detections[best_i]
            t.foot = fpt; t.bbox = bbox; t.last_seen = now
            used_det.add(best_i)
    for i, (fpt, bbox) in enumerate(detections):
        if i in used_det: continue
        tracks.append(Track(fpt, bbox))
    tracks = [t for t in tracks if now - t.last_seen <= 1.0]
    return tracks

# ------------- Alarm control (loop while inside>0) -------------
alarm_process = None

def alarm_worker():
    while not alarm_stop.is_set():
        if os.path.exists(ALARM_FILE):
            # play the file once
            subprocess.run(["afplay", ALARM_FILE])
        else:
            # fallback: macOS voice
            subprocess.run(["say", "Intruder detected"])
        # small delay to avoid hammering CPU
        time.sleep(0.1)

def play_alarm_loop():
    global alarm_thread, alarm_stop
    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_stop.clear()
        alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
        alarm_thread.start()

def stop_alarm_loop():
    global alarm_stop
    alarm_stop.set()

# ------------- Snapshot -------------
def save_snapshot(frame, track_id):
    try:
        os.makedirs("imgs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join("imgs", f"alert_t{track_id}_{ts}.jpg")
        cv2.imwrite(path, frame)
    except Exception:
        pass

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index to open")
    ap.add_argument("--select", action="store_true", help="Interactive camera picker")
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE, help="YOLO inference size")
    args = ap.parse_args()

    cap = open_camera(args)
    model = YOLO(MODEL_NAME)

    calib = load_line()
    if calib is None:
        calib = calibrate_line_two_clicks(cap)
        if calib is None: cap.release(); return
    p1, p2, side_sign = calib

    tracks = []
    print("Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        draw_line_overlay(frame, p1, p2, side_sign, shade_alpha=0.22)

        pred = model.predict(frame, imgsz=args.imgsz, conf=CONF_THRES, classes=[0], verbose=False)[0]

        detections = []
        for b in pred.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            foot = ((x1 + x2)//2, y2)
            detections.append((foot, (x1,y1,x2,y2)))

        tracks = match_tracks(tracks, detections, max_dist=80)

        now = time.time()
        total_people = len(detections)
        inside_count = 0

        # Draw / state updates
        for t in tracks:
            x1,y1,x2,y2 = t.bbox
            foot = t.foot
            side = point_side(p1, p2, foot)
            is_inside = (side == np.sign(side_sign))

            # Entering
            if is_inside and not t.inside:
                t.inside = True
                t.enter_time = now
                t.snap_taken = False
            # Snapshot once per entry
            if is_inside and not t.snap_taken:
                save_snapshot(frame, t.id)
                t.snap_taken = True
            # Leaving
            if not is_inside and t.inside:
                t.inside = False
                t.enter_time = None
                t.snap_taken = False

            if is_inside: inside_count += 1
            color = (0,0,255) if is_inside else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.circle(frame, foot, 4, color, -1)

            label = f"ID {t.id}"
            if is_inside and t.enter_time is not None:
                elapsed = now - t.enter_time
                label += f" | in: {elapsed:.2f}s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0,0,0), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Alarm control based on aggregate inside_count
        if inside_count > 0:
            play_alarm_loop()
        else:
            stop_alarm_loop()

        # Counters panel
        panel = f"People: {total_people}   Inside: {inside_count}"
        (tw, th), _ = cv2.getTextSize(panel, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = frame.shape[1] - tw - 12; y = 28
        cv2.rectangle(frame, (x-8, y-th-8), (x+tw+8, y+8), (0,0,0), -1)
        cv2.putText(frame, panel, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, "ESC=quit", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("gate line detector (alarm)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    stop_alarm_loop()  # ensure it stops if window closed
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()