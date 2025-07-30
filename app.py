from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from rembg import remove
import uuid, os, base64, tempfile, shutil
from collections import Counter

app = FastAPI(title="AI 4 Green - API")

# CORS mở rộng cho giai đoạn phát triển
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler rõ ràng
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"[❌ Exception]: {str(exc)}")
    return PlainTextResponse(str(exc), status_code=500)

# Kiểm tra và tải model
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
model = YOLO(MODEL_PATH)

LABELS_VI = {
    "plastic_bottle": "Chai nhựa",
    "plastic_bottle_cap": "Nắp chai nhựa",
    "paper_cup": "Cốc giấy",
    "tongue_depressor": "Que đè lưỡi",
    "cardboard": "Bìa cứng",
    "straw": "Ống hút"
}

TEMP_DIR = tempfile.mkdtemp()
os.makedirs(TEMP_DIR, exist_ok=True)

def process_image(file_data: bytes) -> tuple[Image.Image, str]:
    uid = str(uuid.uuid4())
    raw_path = os.path.join(TEMP_DIR, f"{uid}.png")
    with open(raw_path, "wb") as f:
        f.write(file_data)
    image = Image.open(raw_path).convert("RGBA")
    fg = remove(image)
    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    white_image = Image.alpha_composite(white_bg, fg).convert("RGB")
    white_path = os.path.join(TEMP_DIR, f"{uid}_white.jpg")
    white_image.save(white_path, quality=95)
    return white_image, white_path

@app.get("/")
def root():
    return {
        "status": "✅ API is running",
        "model_name": getattr(model, "name", "Unknown"),
        "num_classes": len(model.names),
        "class_names": {i: LABELS_VI.get(name, name) for i, name in model.names.items()}
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh!")
        file_data = await file.read()
        if len(file_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File quá lớn (tối đa 10MB)!")

        white_image, white_path = process_image(file_data)

        results = model.predict(
            source=white_path,
            conf=0.3,
            save=True,
            save_txt=False,
            save_conf=False,
            project=TEMP_DIR,
            name="predict",
            exist_ok=True
        )

        result_file = os.path.join(TEMP_DIR, "predict", os.path.basename(white_path))
        if not os.path.exists(result_file):
            raise HTTPException(status_code=500, detail="Không tìm thấy ảnh kết quả.")

        class_counts = Counter([model.names[int(cls)] for cls in results[0].boxes.cls])
        items = [{"name": LABELS_VI.get(name, name), "quantity": count} for name, count in class_counts.items()]

        with open(result_file, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        return JSONResponse(content={
            "items": items,
            "image": f"data:image/jpeg;base64,{img_base64}"
        })

    except HTTPException as http_err:
        raise http_err
