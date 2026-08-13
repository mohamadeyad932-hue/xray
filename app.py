import io, base64, gc
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import PeftModel
from huggingface_hub import hf_hub_download
import uvicorn

# ══════════════════════════════════════════════════════════
# تحسينات الذاكرة القصوى لـ Render Free (512MB RAM)
# ══════════════════════════════════════════════════════════
torch.set_num_threads(1)           # خيط واحد فقط لتوفير الذاكرة
torch.set_grad_enabled(False)      # إيقاف التدرجات نهائياً (لا حاجة لها في الاستدلال)

app = FastAPI(title="Smart Chest X-Ray API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASSES)
REPO_ID = "eyad-ai/SmartChestXRay"
CACHE_DIR = "./my_model_cache"

print("⏳ جاري تجهيز النموذج (وضع توفير الذاكرة)...")

api_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino", cache_dir=CACHE_DIR)

# تحميل النموذج بدقة float16 = نصف حجم الذاكرة (~175MB بدل ~350MB)
base = AutoModel.from_pretrained(
    "microsoft/rad-dino",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

hidden_size = base.config.hidden_size
patch_size = base.config.patch_size
image_size = api_processor.size.get("shortest_edge", 224)
grid_size = image_size // patch_size

api_backbone = PeftModel.from_pretrained(base, REPO_ID)
del base  # حذف المرجع القديم لتحرير الذاكرة
gc.collect()

api_disease_head = nn.Sequential(nn.Dropout(0.0), nn.Linear(hidden_size, NUM_CLASSES))
api_disease_head = api_disease_head.half()  # تحويل لـ float16 أيضاً
head_path = hf_hub_download(repo_id=REPO_ID, filename="disease_head.pt")
state = torch.load(head_path, map_location="cpu", weights_only=True)
# تحويل أوزان الـ head إلى float16
state = {k: v.half() for k, v in state.items()}
api_disease_head.load_state_dict(state)
del state
gc.collect()

api_backbone.eval()
api_disease_head.eval()
gc.collect()
print("✅ السيرفر جاهز! (وضع توفير الذاكرة)")


@app.get("/")
async def health():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        del image_bytes  # تحرير الذاكرة فوراً
    except Exception:
        raise HTTPException(status_code=400, detail="تعذر فتح الصورة")

    # تصغير الصورة لتقليل الذاكرة أثناء المعالجة
    img.thumbnail((224, 224))

    inputs = api_processor(images=img, return_tensors="pt")
    # تحويل المدخلات إلى float16 لتتوافق مع النموذج
    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = api_backbone(**inputs, output_attentions=True)
        feat = outputs.last_hidden_state[:, 0, :]
        probs = torch.softmax(api_disease_head(feat).float(), dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        predicted_class = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

        # توليد الخريطة الحرارية بدون matplotlib (لتوفير الذاكرة)
        heatmap_base64 = None
        try:
            attentions = outputs.attentions
            if attentions is not None and len(attentions) > 0:
                import numpy as np
                last_attn = attentions[-1][0].float()
                avg_attn = last_attn.mean(dim=0)
                cls_attn = avg_attn[0, 1:].cpu().numpy()

                attn_map = cls_attn.reshape(grid_size, grid_size)
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                attn_resized = np.array(
                    Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                        img.size, resample=Image.BICUBIC
                    )
                ) / 255.0

                # توليد ألوان بسيطة بدون matplotlib (أخف على الذاكرة)
                r = (attn_resized * 255).astype(np.uint8)
                g = ((1 - attn_resized) * 100).astype(np.uint8)
                b = ((1 - attn_resized) * 255).astype(np.uint8)
                heatmap_colored = np.stack([r, g, b], axis=-1)

                original_array = np.array(img)
                blended = (0.5 * original_array + 0.5 * heatmap_colored).astype(np.uint8)

                blended_image = Image.fromarray(blended)
                buffer = io.BytesIO()
                blended_image.save(buffer, format="JPEG", quality=70)
                buffer.seek(0)
                heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                del last_attn, avg_attn, cls_attn, attn_map, attn_resized
                del heatmap_colored, original_array, blended, blended_image, buffer
        except Exception as e:
            print(f"Heatmap skipped: {e}")

    # تحرير الذاكرة بعد كل طلب
    del outputs, inputs, feat, probs
    gc.collect()

    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.1f}%",
        "all_probabilities": {CLASSES[i]: f"{0.0:.1f}%" for i in range(NUM_CLASSES)},
        "heatmap_base64": heatmap_base64,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
