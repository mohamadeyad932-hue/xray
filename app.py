import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import PeftModel
from huggingface_hub import hf_hub_download

app = FastAPI(title="Smart Chest X-Ray API")

REPO_ID = "eyad-ai/SmartChestXRay"
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

print("⏳ جاري تحميل النموذج محلياً على سيرفر Render...")

# 1. تحميل المعالج والباكبون الأساسي ودمج LoRA
processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
base_backbone = AutoModel.from_pretrained("microsoft/rad-dino")
peft_model = PeftModel.from_pretrained(base_backbone, REPO_ID)
merged_backbone = peft_model.merge_and_unload()
hidden_size = merged_backbone.config.hidden_size

# 2. بناء معمارية النموذج المطابقة تماماً لرأس التصنيف لديك (768)
class ChestXRayModel(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

model = ChestXRayModel(merged_backbone, hidden_size, len(CLASSES))

# 3. تحميل وربط أوزان رأس التصنيف بدقة متناهية
head_path = hf_hub_download(repo_id=REPO_ID, filename="disease_head.pt")
state_dict = torch.load(head_path, map_location="cpu")
with torch.no_grad():
    model.classifier[1].weight.copy_(state_dict["1.weight"])
    model.classifier[1].bias.copy_(state_dict["1.bias"])

model.eval()
print("✅ تم تحميل النموذج وجاهز للتنبؤ بنجاح!")

@app.get("/")
def home():
    return {"status": "Smart Chest X-Ray API is running locally!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)[0]
        
        result = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
