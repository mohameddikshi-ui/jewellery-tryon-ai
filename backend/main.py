from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import mediapipe as mp
import os

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
    alpha = np.clip(alpha * 1.2, 0, 1)

    for c in range(3):
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

    width = int(abs(right[0]-left[0]) * 1.6)
    necklace = cv2.resize(
        necklace,
        (width, int(width * necklace.shape[0] / necklace.shape[1]))
    )

    necklace = enhance_jewellery(necklace)

    cx = (left[0] + right[0]) // 2
    x = cx - necklace.shape[1] // 2
    y = chin[1] + 40

    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    x = max(0, min(x, w - necklace.shape[1]))
    y = max(0, min(y, h - necklace.shape[0]))

    canvas[y:y+necklace.shape[0], x:x+necklace.shape[1]] = necklace

    return canvas


# ================= EARRINGS =================
def place_earrings(img, earring, lm, w, h):
    left_ear = lm[234]
    right_ear = lm[454]

    size = int(w * 0.08)

    ear = cv2.resize(
        earring,
        (size, int(size * earring.shape[0] / earring.shape[1]))
    )

    ear = enhance_jewellery(ear)

    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    y_offset = int(h * 0.035)

    lx = int(left_ear.x * w) - ear.shape[1] // 2
    ly = int(left_ear.y * h) + y_offset

    rx = int(right_ear.x * w) - ear.shape[1] // 2
    ry = int(right_ear.y * h) + y_offset

    if 0 <= lx < w - ear.shape[1] and 0 <= ly < h - ear.shape[0]:
        canvas[ly:ly+ear.shape[0], lx:lx+ear.shape[1]] = ear

    ear_flip = cv2.flip(ear, 1)

    if 0 <= rx < w - ear_flip.shape[1] and 0 <= ry < h - ear_flip.shape[0]:
        canvas[ry:ry+ear_flip.shape[0], rx:rx+ear_flip.shape[1]] = ear_flip

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
                placed = place_earrings(img, jewellery, lm, w, h)

            else:
                return {"error": "Invalid type"}

            img = realistic_blend(img, placed)

        output = cv2.resize(img, (original.shape[1], original.shape[0]))
        cv2.imwrite("outputs/output.jpg", output)

        return {"output": "outputs/output.jpg"}

    except Exception as e:
        return {"error": str(e)}