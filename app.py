import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart Chest X-Ray API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# إعدادات الـ API الخاصة بـ Hugging Face
# ══════════════════════════════════════════════════════════
HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/eyad-ai/SmartChestXRay"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.get("/")
async def health():
    return {"status": "running on API mode"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="تعذر فتح أو قراءة الصورة")

    response = requests.post(API_URL, headers=headers, data=image_bytes)

    if response.status_code != 200:
        if "is currently loading" in response.text:
            raise HTTPException(status_code=503, detail="النموذج قيد التحضير في Hugging Face، يرجى المحاولة بعد 20 ثانية.")
        raise HTTPException(status_code=response.status_code, detail=f"خطأ من API: {response.text}")

    result = response.json()
    
    all_probabilities = {}
    predicted_class = "Unknown"
    confidence = 0.0

    if isinstance(result, list) and len(result) > 0:
        sorted_results = sorted(result, key=lambda x: x.get("score", 0), reverse=True)
        
        predicted_class = sorted_results[0].get("label", "Unknown")
        confidence = sorted_results[0].get("score", 0) * 100

        for item in sorted_results:
            label = item.get("label", "Unknown")
            score = item.get("score", 0) * 100
            all_probabilities[label] = f"{score:.1f}%"

    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.1f}%",
        "all_probabilities": all_probabilities
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
