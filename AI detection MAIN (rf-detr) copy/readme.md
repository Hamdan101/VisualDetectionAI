Setup:

1: cd into the directory
2: git clone https://github.com/roboflow/rf-detr.git
2:
A) python3 -m venv venv
B) source venv/bin/activate
C) python -m pip install --upgrade pip
D) pip install -e ./rf-detr
E) pip install opencv-python numpy
F) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
3: python3 rf_line_alarm_local.py --select




Feature list:

	•	Detects humans only (configurable class ID)
	•	Line crossing with a tolerance buffer (reduces jitter/false triggers)
	•	Foot-point offset and EMA smoothing for stable crossings
	•	Alarm (loops while anyone is inside) + auto-stop on exit
	•	Snapshots saved to ./imgs/ on first entry per person
	•	Counters: total people visible + inside count
	•	Camera picker for multi-webcam setups

Project layout:

AI detection (rf-detr)/
├── rf-detr/                   # RF-DETR repo (cloned)
├── rf_line_alarm_local.py     # the app script
├── alarm.mp3                  # optional siren (or alarm.wav)
├── imgs/                      # snapshots (auto-created)
├── line_calibration.json      # saved after first calibration
└── venv/                      # Python virtual env

Requirements:

	•	macOS with a webcam (internal or USB)
	•	Python 3.11+ (3.13 also works for many setups)
	•	Permissions: System Settings → Privacy & Security → Camera → enable Terminal


