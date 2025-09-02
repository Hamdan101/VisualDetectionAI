
import os, json, argparse, time, math, subprocess, platform, threading
import numpy as np
import cv2

# ---- import RF-DETR from your local editable install (-e .) ----
from rfdetr import RFDETRNano   # fast; swap to RFDETRSmall / RFDETRMedium if you want
from rfdetr.util.coco_classes import COCO_CLASSES

CALIB_FILE = "line_calibration.json"
ALARM_FILE = "alarm.mp3"  # afplay supports mp3 or wav
CONF_THRES = 0.35
PERSON_ID = 1              # your RF-DETR showed person as cls:1

# ---- NEW: robustness knobs ----
BUFFER_PX = 12            # tolerance band around the line (pixels)
FOOT_OFFSET_PX = 8        # push "foot" below bbox bottom (pixels)
EMA_ALPHA = 0.5           # 0..1; higher = snappier, lower = smoother

# ---------------- Camera helpers ----------------
def _cap_backend():
    return getattr(cv2, "CAP_AVFOUNDATION", 0)

def try_open(idx, w=640, h=360):
    cap = cv2.VideoCapture(idx, _cap_backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
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
            if k in (ord(' '), 13):
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

# ---------------- Line helpers (existing) ----------------
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
    return np.sign(cross)

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
            side_sign = 1.0
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
                elif k2 in (13,):
                    save_line(p1, p2, side_sign)
                    cv2.destroyWindow("calibrate-line")
                    print(f"Saved {CALIB_FILE}: p1={p1}, p2={p2}, side_sign={int(side_sign)}")
                    return (p1, p2, side_sign)
                elif k2 in (ord('q'), ord('Q'), 27):
                    cv2.destroyWindow("calibrate-line")
                    return None

# ---------------- NEW: geometric helpers for robustness ----------------
def signed_distance_to_line(p1, p2, pt):
    """Signed perpendicular distance from pt to line p1->p2 (pixels)."""
    x1,y1 = p1; x2,y2 = p2; x,y = pt
    dx, dy = (x2 - x1), (y2 - y1)
    denom = math.hypot(dx, dy) + 1e-9
    return ((x - x1) * dy - (y - y1) * dx) / denom  # sign = which side

def draw_buffer_band(frame, p1, p2, band_px=12, color=(0,255,255), alpha=0.18):
    """Visualize the +/- buffer band around the line."""
    dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L, dx / L  # unit normal
    p1a = (int(p1[0] + nx*band_px), int(p1[1] + ny*band_px))
    p2a = (int(p2[0] + nx*band_px), int(p2[1] + ny*band_px))
    p1b = (int(p1[0] - nx*band_px), int(p1[1] - ny*band_px))
    p2b = (int(p2[0] - nx*band_px), int(p2[1] - ny*band_px))
    overlay = frame.copy()
    poly = np.array([p1a, p2a, p2b, p1b], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, dst=frame)

# ---------------- Alarm loop ----------------
alarm_thread = None
alarm_stop = threading.Event()

def alarm_worker():
    while not alarm_stop.is_set():
        if os.path.exists(ALARM_FILE):
            subprocess.run(["afplay", ALARM_FILE])
        else:
            subprocess.run(["say", "Intruder detected"])
        time.sleep(0.05)

def play_alarm_loop():
    global alarm_thread
    if platform.system() == "Darwin":
        if alarm_thread is None or not alarm_thread.is_alive():
            alarm_stop.clear()
            alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
            alarm_thread.start()

def stop_alarm_loop():
    alarm_stop.set()

def save_snapshot(frame, track_id):
    try:
        os.makedirs("imgs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join("imgs", f"alert_t{track_id}_{ts}.jpg")
        cv2.imwrite(path, frame)
    except Exception:
        pass

# ---------------- Simple tracker (with smoothing) ----------------
class Track:
    _next_id = 1
    def __init__(self, foot_px, bbox):
        self.id = Track._next_id; Track._next_id += 1
        self.foot = foot_px
        self.foot_ema = foot_px  # smoothed foot
        self.bbox = bbox
        self.last_seen = time.time()
        self.inside = False
        self.enter_time = None
        self.snap_taken = False

    def update(self, foot_px, bbox):
        # EMA smoothing for foot position
        self.foot = foot_px
        self.foot_ema = (
            int(EMA_ALPHA * foot_px[0] + (1-EMA_ALPHA) * self.foot_ema[0]),
            int(EMA_ALPHA * foot_px[1] + (1-EMA_ALPHA) * self.foot_ema[1]),
        )
        self.bbox = bbox
        self.last_seen = time.time()

def match_tracks(tracks, detections, max_dist=80):
    now = time.time()
    used = set()
    for t in tracks:
        best_i, best_d = None, 1e9
        for i, (foot,bbox) in enumerate(detections):
            if i in used: continue
            d = math.hypot(foot[0]-t.foot[0], foot[1]-t.foot[1])
            if d < best_d: best_d, best_i = d, i
        if best_i is not None and best_d <= max_dist:
            foot,bbox = detections[best_i]
            t.update(foot, bbox)
            used.add(best_i)
    for i,(foot,bbox) in enumerate(detections):
        if i not in used:
            tracks.append(Track(foot,bbox))
    return [t for t in tracks if now - t.last_seen <= 1.0]

# ---------------- RF-DETR inference wrapper ----------------
class RFDetrDetector:
    def __init__(self, use_nano=True):
        self.model = RFDETRNano() if use_nano else RFDETRNano()
        self.model.optimize_for_inference()  # native graph optimizations

    def infer(self, frame_bgr, conf=CONF_THRES):
        # RF-DETR expects RGB [0..1]; .predict() accepts NumPy RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(rgb, threshold=conf)  # Detections object
        out = []
        for (xyxy, cls_id, score) in zip(detections.xyxy, detections.class_id, detections.confidence):
            x1,y1,x2,y2 = map(int, xyxy)
            out.append((x1,y1,x2,y2,float(score), int(cls_id)))
        return out

# ---------------- Main ----------------
def main():
    global BUFFER_PX, FOOT_OFFSET_PX  # allow live tuning keys

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    ap.add_argument("--skip", type=int, default=1, help="process every Nth frame (2 or 3 boosts FPS)")
    args = ap.parse_args()

    cap = open_camera(args)
    detector = RFDetrDetector(use_nano=True)

    calib = load_line()
    if calib is None:
        calib = calibrate_line_two_clicks(cap)
        if calib is None: cap.release(); return
    p1, p2, side_sign = calib

    tracks = []
    frame_idx = 0
    print("Press ESC to quit.  [+/-] buffer  [ [ / ] ] foot offset")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # Optional frame skipping for speed
        if args.skip > 1 and (frame_idx % args.skip != 0):
            draw_line_overlay(frame, p1, p2, side_sign, shade_alpha=0.22)
            draw_buffer_band(frame, p1, p2, band_px=BUFFER_PX, color=(0,255,255), alpha=0.18)
            cv2.putText(frame, "Skipping for speed...", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"buffer={BUFFER_PX}px  foot_offset={FOOT_OFFSET_PX}px",
                        (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.imshow("RF-DETR (local) line alarm", frame)
            k = (cv2.waitKey(1) & 0xFF)
            if k == 27: break
            elif k in (ord('+'), ord('=')): BUFFER_PX = min(100, BUFFER_PX + 2)
            elif k in (ord('-'), ord('_')): BUFFER_PX = max(0, BUFFER_PX - 2)
            elif k == ord(']'): FOOT_OFFSET_PX = min(40, FOOT_OFFSET_PX + 2)
            elif k == ord('['): FOOT_OFFSET_PX = max(-40, FOOT_OFFSET_PX - 2)
            continue

        draw_line_overlay(frame, p1, p2, side_sign, shade_alpha=0.22)
        draw_buffer_band(frame, p1, p2, band_px=BUFFER_PX, color=(0,255,255), alpha=0.18)

        # RF-DETR inference (local)
        dets = detector.infer(frame, conf=args.conf)

        # ---- DEBUG: list first 10 raw detections (cls, score)

        # Build detections for tracker (filter to person here)
        detections = []
        for x1,y1,x2,y2,score,cls_id in dets:
            if int(cls_id) != PERSON_ID or score < args.conf:
                continue
            cx = (x1 + x2) // 2
            foot_y = y2 + FOOT_OFFSET_PX  # push foot below bbox
            detections.append(((cx, foot_y), (x1,y1,x2,y2)))

        tracks = match_tracks(tracks, detections, max_dist=80)

        now = time.time()
        total_people = len(detections)
        inside_count = 0
        target_sign = np.sign(side_sign)

        for t in tracks:
            x1,y1,x2,y2 = t.bbox
            foot = t.foot_ema  # use smoothed footpoint
            sd = signed_distance_to_line(p1, p2, foot)  # signed distance in px
            is_inside = (np.sign(sd) == target_sign) and (abs(sd) >= BUFFER_PX)

            if is_inside and not t.inside:
                t.inside = True; t.enter_time = now; t.snap_taken = False
            if is_inside and not t.snap_taken:
                save_snapshot(frame, t.id); t.snap_taken = True
            if not is_inside and t.inside:
                t.inside = False; t.enter_time = None; t.snap_taken = False

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

        # Alarm control
        if inside_count > 0: play_alarm_loop()
        else:                stop_alarm_loop()

        # Counters
        panel = f"Risk: {total_people}   Breaches: {inside_count}"
        (tw, th), _ = cv2.getTextSize(panel, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = frame.shape[1] - tw - 12; y = 28
        cv2.rectangle(frame, (x-8, y-th-8), (x+tw+8, y+8), (0,0,0), -1)
        cv2.putText(frame, panel, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"buffer={BUFFER_PX}px  foot_offset={FOOT_OFFSET_PX}px   ESC=quit  [+/-][/[ ]]",
                    (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        cv2.imshow("RF-DETR (local) line alarm", frame)
        k = (cv2.waitKey(1) & 0xFF)
        if k == 27: break
        elif k in (ord('+'), ord('=')): BUFFER_PX = min(100, BUFFER_PX + 2)
        elif k in (ord('-'), ord('_')): BUFFER_PX = max(0, BUFFER_PX - 2)
        elif k == ord(']'): FOOT_OFFSET_PX = min(40, FOOT_OFFSET_PX + 2)
        elif k == ord('['): FOOT_OFFSET_PX = max(-40, FOOT_OFFSET_PX - 2)

    stop_alarm_loop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()