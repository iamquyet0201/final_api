from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from rembg import remove
import base64
import io
from collections import Counter
import os

app = FastAPI(title="AI 4 Green - API")

# Cho phép CORS toàn bộ (dùng cho dev, nên giới hạn khi deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bắt lỗi toàn cục để không bị crash
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"[❌ Exception]: {str(exc)}")
    return PlainTextResponse(str(exc), status_code=500)

# Load mô hình
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
model = YOLO(MODEL_PATH)

# Nhãn tiếng Việt tương ứng với lớp YOLO
LABELS_VI = {
    "plastic_bottle": "Chai nhựa",
    "plastic_bottle_cap": "Nắp chai nhựa",
    "paper_cup": "Cốc giấy",
    "tongue_depressor": "Que đè lưỡi",
    "cardboard": "Bìa cứng",
    "straw": "Ống hút"
}

# Hàm xử lý ảnh: xóa nền và chuyển sang nền trắng
def remove_background_and_whiten(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        fg = remove(image)
        white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        white_image = Image.alpha_composite(white_bg, fg).convert("RGB")
        return white_image
    except Exception as e:
        raise ValueError(f"Lỗi xử lý ảnh hoặc xóa nền: {str(e)}")

@app.get("/")
def root():
    return {
        "status": "✅ API is running",
        "model_name": model.model.args.get("name", "Unknown"),
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

        # Xử lý ảnh
        white_image = remove_background_and_whiten(file_data)

        # Dự đoán
        results = model.predict(white_image, conf=0.3)
        result = results[0]

        # Đếm số lượng vật thể
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            class_counts = {}
        else:
            class_counts = Counter([model.names[int(cls)] for cls in boxes.cls])

        items = [{
            "name": name,
            "label": LABELS_VI.get(name, name),
            "quantity": count
        } for name, count in class_counts.items()]

        # Annotate ảnh bằng .plot() và encode base64
        annotated_image = Image.fromarray(result.plot())
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "items": items,
            "image": f"data:image/jpeg;base64,{img_base64}"
        })

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"[❌ Exception]: {str(e)}")
        return JSONResponse(content={"detail": "Lỗi không xác định trong quá trình dự đoán."}, status_code=500)
