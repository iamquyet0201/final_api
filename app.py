import os
import uuid
import base64
import tempfile
import shutil
from collections import Counter
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from rembg import remove

app = FastAPI(title="AI 4 Green - API")

# Cho phép CORS toàn bộ (dùng cho dev, nên giới hạn khi deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bắt lỗi toàn cục
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"[❌ Exception]: {str(exc)}")
    return PlainTextResponse(str(exc), status_code=500)

# Load mô hình
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
model = YOLO(MODEL_PATH)

# Nhãn tiếng Việt tương ứng
LABELS_VI = {
    "plastic_bottle": "Chai nhựa",
    "plastic_bottle_cap": "Nắp chai nhựa",
    "paper_cup": "Cốc giấy",
    "tongue_depressor": "Que đè lưỡi",
    "cardboard": "Bìa cứng",
    "straw": "Ống hút"
}

# Hàm xử lý ảnh: xóa nền và chuyển sang trắng
def process_image(file_data: bytes) -> Image.Image:
    image = Image.open(tempfile.NamedTemporaryFile(delete=False, suffix=".png")).convert("RGBA")
    image = Image.open(image.fp.name)
    fg = remove(image)
    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_bg, fg).convert("RGB")

# Route kiểm tra
@app.get("/")
def root():
    return {
        "status": "✅ API is running",
        "model_name": model.model.args.get("name", "Unknown"),
        "num_classes": len(model.names),
        "class_names": {i: LABELS_VI.get(name, name) for i, name in model.names.items()}
    }

# Route dự đoán
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh!")
        file_data = await file.read()
        if len(file_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File quá lớn (tối đa 10MB)!")

        image = process_image(file_data)
        results = model.predict(image, conf=0.3)

        # Đếm số lượng vật thể theo lớp
        class_counts = Counter([model.names[int(cls)] for cls in results[0].boxes.cls])

        items = [{
            "name": name,
            "label": LABELS_VI.get(name, name),
            "quantity": count
        } for name, count in class_counts.items()]

        # Annotate kết quả trực tiếp lên ảnh bằng OpenCV + base64
        annotated = results[0].plot()  # Numpy array (RGB)
        img = Image.fromarray(annotated)
        buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(buffer.name, format="JPEG")
        with open(buffer.name, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        return JSONResponse(content={
            "items": items,
            "image": f"data:image/jpeg;base64,{img_base64}"
        })

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"[❌ Exception]: {str(e)}")
        return JSONResponse(content={"detail": "Lỗi không xác định trong quá trình dự đoán."}, status_code=500)
