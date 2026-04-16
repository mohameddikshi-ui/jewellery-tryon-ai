from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import mediapipe as mp
import os
from ultralytics import YOLO

ear_model = YOLO("models/best.pt")

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= FOLDERS =================
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ✅ Serve output + jewellery
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/jewelry", StaticFiles(directory="jewelry"), name="jewelry")

mp_face = mp.solutions.face_mesh


# ================= BLEND =================
def realistic_blend(bg, fg):
    alpha = fg[:, :, 3] / 255.0
    alpha = np.clip(alpha * 0.9, 0, 1)

    # 🔥 SHADOW (depth)
    shadow = cv2.GaussianBlur(alpha, (25, 25), 10) * 0.25

    for c in range(3):
        bg[:, :, c] = (1 - shadow) * bg[:, :, c]
        bg[:, :, c] = alpha * fg[:, :, c] + (1 - alpha) * bg[:, :, c]

    return bg.astype(np.uint8)


# ================= ENHANCE =================
def enhance_jewellery(fg):
    rgb = fg[:, :, :3]
    rgb = cv2.convertScaleAbs(rgb, alpha=1.3, beta=10)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    rgb = cv2.filter2D(rgb, -1, kernel)

    fg[:, :, :3] = rgb
    return fg


# ================= LANDMARK =================
def get_points(lm, w, h):
    left = lm[234]
    right = lm[454]
    chin = lm[152]

    return (
        (int(left.x*w), int(left.y*h)),
        (int(right.x*w), int(right.y*h)),
        (int(chin.x*w), int(chin.y*h))
    )


# ================= NECKLACE =================
def place_necklace(img, necklace, left, right, chin):
    h, w, _ = img.shape

    # 🔥 BETTER WIDTH (jaw based, not face)
    face_width = abs(right[0] - left[0])
    width = int(face_width * 1.25)   # reduced from 1.6 → more natural

    necklace = cv2.resize(
        necklace,
        (width, int(width * necklace.shape[0] / necklace.shape[1]))
    )

    necklace = enhance_jewellery(necklace)

    # 🔥 CENTER
    cx = (left[0] + right[0]) // 2

    # 🔥 KEY FIX: LOWER POSITION (neck depth)
    x = cx - necklace.shape[1] // 2
    y = chin[1] + int(h * 0.015)   # SMALL offset (was 40 → too big)

    # 🔥 ADD GRAVITY DROP (center slightly down)
    drop = int(necklace.shape[0] * 0.08)

    # create warp (middle goes down slightly)
    rows, cols = necklace.shape[:2]
    for i in range(cols):
        shift = int(drop * (1 - abs((i - cols/2) / (cols/2))))
        necklace[:, i] = np.roll(necklace[:, i], shift, axis=0)

    # 🔥 SOFT EDGE (important)
    alpha = necklace[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (9, 9), 5)
    necklace[:, :, 3] = alpha

    # 🔥 CANVAS
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    x = max(0, min(x, w - necklace.shape[1]))
    y = max(0, min(y, h - necklace.shape[0]))

    canvas[y:y+necklace.shape[0], x:x+necklace.shape[1]] = necklace

    return canvas

# ================= EARRINGS () =================
def place_earrings_ai(img, earring):
    h, w, _ = img.shape

    results = ear_model(img)[0]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    earlobes = []

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls == 0:  # earlobe
            x1, y1, x2, y2 = box.xyxy[0]

            cx = int((x1 + x2) / 2)

            # ✅ KEY FIX: use LOWER PART of box (not center)
            cy = int(y2)   # bottom of earlobe

            earlobes.append((cx, cy))

    # sort left → right
    earlobes = sorted(earlobes, key=lambda x: x[0])

    size = int(w * 0.07)

    ear = cv2.resize(
        earring,
        (size, int(size * earring.shape[0] / earring.shape[1]))
    )

    ear = enhance_jewellery(ear)

    for i, (cx, cy) in enumerate(earlobes):

        x = cx - ear.shape[1] // 2

        # ✅ DROP slightly BELOW earlobe
        y = cy - int(ear.shape[0] * 0.1)

        if i == 1:
            ear_use = cv2.flip(ear, 1)
        else:
            ear_use = ear

        if 0 <= x < w - ear_use.shape[1] and 0 <= y < h - ear_use.shape[0]:
            canvas[y:y+ear_use.shape[0], x:x+ear_use.shape[1]] = ear_use

    return canvas
# ================= API =================
@app.post("/tryon")
async def tryon(data: dict):
    try:
        img_bytes = base64.b64decode(data["image"])

        with open("uploads/input.jpg", "wb") as f:
            f.write(img_bytes)

        img = cv2.imread("uploads/input.jpg")
        original = img.copy()

        h, w, _ = img.shape

        with mp_face.FaceMesh(static_image_mode=True) as face:
            res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not res.multi_face_landmarks:
                return {"error": "No face detected"}

            lm = res.multi_face_landmarks[0].landmark

            jewellery = cv2.imread(
                f"jewelry/{data['item']}.png",
                cv2.IMREAD_UNCHANGED
            )

            if data["type"] == "necklace":
                left, right, chin = get_points(lm, w, h)
                placed = place_necklace(img, jewellery, left, right, chin)

            elif data["type"] == "earring":
                placed = place_earrings_ai(img, jewellery)

            else:
                return {"error": "Invalid type"}

            img = realistic_blend(img, placed)

        output = cv2.resize(img, (original.shape[1], original.shape[0]))
        cv2.imwrite("outputs/output.jpg", output)

        return {"output": "outputs/output.jpg"}

    except Exception as e:
        return {"error": str(e)}