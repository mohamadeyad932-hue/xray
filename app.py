import io, base64
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import PeftModel
from huggingface_hub import hf_hub_download
import uvicorn

app = FastAPI(title="Smart Chest X-Ray API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASSES)
REPO_ID = "eyad-ai/SmartChestXRay"

# مجلد محلي وثابت لتخزين الأوزان مرة واحدة وللأبد
CACHE_DIR = "./my_model_cache"

print("⏳ جاري تجهيز النموذج والتحقق من الملفات المحلية...")

# تم إزالة force_download وإضافة cache_dir لعدم إعادة التحميل
api_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino", cache_dir=CACHE_DIR)
base = AutoModel.from_pretrained("microsoft/rad-dino", cache_dir=CACHE_DIR)

hidden_size = base.config.hidden_size
patch_size = base.config.patch_size
image_size = api_processor.size.get("shortest_edge", 224)
grid_size = image_size // patch_size 

api_backbone = PeftModel.from_pretrained(base, REPO_ID)
api_disease_head = nn.Sequential(nn.Dropout(0.0), nn.Linear(hidden_size, NUM_CLASSES))
head_path = hf_hub_download(repo_id=REPO_ID, filename="disease_head.pt")
api_disease_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))

api_backbone.eval()
api_disease_head.eval()
print("✅ السيرفر جاهز الآن!")

def generate_attention_heatmap(attentions, original_image):
    last_layer_attn = attentions[-1][0]
    avg_attn = last_layer_attn.mean(dim=0)
    cls_attn = avg_attn[0, 1:].detach().cpu().numpy()
    
    attn_map = cls_attn.reshape(grid_size, grid_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            original_image.size, resample=Image.BICUBIC
        )
    ) / 255.0

    heatmap_colored = cm.jet(attn_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    original_array = np.array(original_image.resize(original_image.size))
    blended = (0.5 * original_array + 0.5 * heatmap_colored).astype(np.uint8)

    blended_image = Image.fromarray(blended)
    buffer = io.BytesIO()
    blended_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="تعذر فتح الصورة")

    inputs = api_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = api_backbone(**inputs, output_attentions=True)
        feat = outputs.last_hidden_state[:, 0, :]
        probs = torch.softmax(api_disease_head(feat), dim=-1)[0]
        
        pred_idx = torch.argmax(probs).item()
        predicted_class = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

        heatmap_base64 = generate_attention_heatmap(outputs.attentions, img)

        return {
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.1f}%",
            "heatmap_base64": heatmap_base64
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)