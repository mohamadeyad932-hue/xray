import io, base64, os
import requests as http_requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ══════════════════════════════════════════════════════════
# سيرفر وسيط خفيف (Proxy) — يدعم إدخال الـ Token عبر متغيرات البيئة
# ══════════════════════════════════════════════════════════

HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "https://eyad-ai-xray-gradio.hf.space")
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # يُدخل كمدخل بيئة (Environment Variable)
GRADIO_API_URL = f"{HF_SPACE_URL}/api/predict"

app = FastAPI(title="Smart Chest X-Ray API (Proxy)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "running", "mode": "proxy", "backend": HF_SPACE_URL}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="تعذر قراءة الملف")

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "png"
    mime = "image/png" if ext == "png" else "image/jpeg"
    img_b64 = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")
    del image_bytes

    # تجهيز الترويسات مع تمرير التوكن إذا كان موجوداً
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        response = http_requests.post(
            GRADIO_API_URL,
            json={"data": [img_b64]},
            headers=headers,
            timeout=120,
        )
    except http_requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="السيرفر الخلفي (HF Space) لم يرد — أعد المحاولة بعد 30 ثانية."
        )
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=502,
            detail="تعذر الاتصال بالسيرفر الخلفي (HF Space)."
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"خطأ من السيرفر الخلفي: {response.text[:300]}"
        )

    try:
        result = response.json()
        data = result.get("data", [])

        diagnosis_text = data[0] if len(data) > 0 else "غير متوفر"

        all_probabilities = {}
        if len(data) > 1 and isinstance(data[1], dict):
            confidences = data[1].get("confidences", [])
            for item in confidences:
                label = item.get("label", "?")
                conf = item.get("confidence", 0)
                all_probabilities[label] = f"{conf * 100:.1f}%"

        heatmap_base64 = None
        if len(data) > 2 and data[2]:
            heatmap_data = data[2]
            if isinstance(heatmap_data, str) and heatmap_data.startswith("data:"):
                heatmap_base64 = heatmap_data.split(",", 1)[1]
            elif isinstance(heatmap_data, dict) and "url" in heatmap_data:
                try:
                    img_resp = http_requests.get(
                        heatmap_data["url"] if heatmap_data["url"].startswith("http")
                        else f"{HF_SPACE_URL}/file={heatmap_data['url']}",
                        headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
                        timeout=30
                    )
                    heatmap_base64 = base64.b64encode(img_resp.content).decode("utf-8")
                except Exception:
                    pass

        predicted_class = "Normal"
        confidence_val = "0.0%"
        if all_probabilities:
            predicted_class = max(all_probabilities, key=lambda k: float(all_probabilities[k].replace("%", "")))
            confidence_val = all_probabilities[predicted_class]

        return {
            "status": "Abnormal ⚠️" if predicted_class != "Normal" else "Healthy ✅",
            "predicted_class": predicted_class,
            "confidence": confidence_val,
            "all_probabilities": all_probabilities,
            "heatmap_base64": heatmap_base64,
            "diagnosis_text": diagnosis_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تفسير النتيجة: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
