import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import PeftModel
from huggingface_hub import hf_hub_download

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASSES)
REPO_ID = "eyad-ai/SmartChestXRay"

print("⏳ تحميل النموذج...")
processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
base_model = AutoModel.from_pretrained("microsoft/rad-dino", attn_implementation="eager")
base_model.config.output_attentions = True

hidden_size = base_model.config.hidden_size
patch_size = base_model.config.patch_size
image_size = processor.size.get("shortest_edge", 224)
grid_size = image_size // patch_size

backbone = PeftModel.from_pretrained(base_model, REPO_ID)
disease_head = nn.Sequential(nn.Dropout(0.0), nn.Linear(hidden_size, NUM_CLASSES))
head_path = hf_hub_download(repo_id=REPO_ID, filename="disease_head.pt")
disease_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))

backbone.eval()
disease_head.eval()
print("✅ النموذج جاهز!")

def generate_heatmap(attentions, original_image):
    try:
        if attentions is None or len(attentions) == 0:
            return None
        last_attn = attentions[-1][0]
        avg_attn = last_attn.mean(dim=0)
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
        return Image.fromarray(blended)
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None

def predict_xray(image):
    if image is None:
        return "❌ لم يتم رفع صورة", {}, None
    
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = backbone(**inputs, output_attentions=True)
        feat = outputs.last_hidden_state[:, 0, :]
        probs = torch.softmax(disease_head(feat), dim=-1)[0]
    
    pred_idx = torch.argmax(probs).item()
    predicted_class = CLASSES[pred_idx]
    confidence = probs[pred_idx].item() * 100
    is_abnormal = predicted_class != "Normal"
    
    status = f"⚠️ غير طبيعي - {predicted_class}" if is_abnormal else "✅ طبيعي (Normal)"
    status_text = f"{status}\nنسبة الثقة: {confidence:.1f}%"
    
    confidences = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    attentions = getattr(outputs, "attentions", None)
    heatmap_img = generate_heatmap(attentions, image) if attentions is not None else None
    
    return status_text, confidences, heatmap_img

interface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil", label="📤 ارفع صورة أشعة الصدر"),
    outputs=[
        gr.Textbox(label="🩺 التشخيص"),
        gr.Label(num_top_classes=4, label="📊 نسب الاحتمالات"),
        gr.Image(type="pil", label="🔥 الخريطة الحرارية (XAI)")
    ],
    title="🩺 نظام التشخيص الذكي لأشعة الصدر",
    description="قم برفع صورة أشعة X-Ray ليقوم النموذج بتشخيص الحالة مع خريطة حرارية توضيحية.",
    flagging_mode="never",
)

if __name__ == "__main__":
    interface.launch()
