import os
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, pipeline
from peft import PeftModel
from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr

REPO_ID = "eyad-ai/SmartChestXRay"
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

print("⏳ جاري تحميل نماذج الذكاء الاصطناعي (الأساسي والفلتر)...")

try:
    # 1. تحميل نموذج الفلترة لمنع الصور الخاطئة
    print("⏳ جاري تحميل فلتر الصور (CLIP)...")
    filter_model = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    # 2. تحميل نموذج الأشعة
    print("⏳ جاري تحميل نموذج الأشعة Rad-DINO...")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    base_backbone = AutoModel.from_pretrained("microsoft/rad-dino")

    peft_model = PeftModel.from_pretrained(base_backbone, REPO_ID)
    merged_backbone = peft_model.merge_and_unload()
    hidden_size = merged_backbone.config.hidden_size

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
            pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
            return self.classifier(pooled)

    model = ChestXRayModel(merged_backbone, hidden_size, len(CLASSES))

    head_path = hf_hub_download(repo_id=REPO_ID, filename="disease_head.pt")
    state_dict = torch.load(head_path, map_location="cpu")
    with torch.no_grad():
        model.classifier[1].weight.copy_(state_dict["1.weight"])
        model.classifier[1].bias.copy_(state_dict["1.bias"])

    model.eval()
    print("✅ تم تحميل النماذج وتجهيز الواجهة بنجاح!")

except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل النماذج: {e}")

def predict_xray(image):
    if image is None:
        return "الرجاء رفع صورة أشعة صحيحة."

    try:
        # مرحلة الفلترة
        filter_result = filter_model(
            image, 
            candidate_labels=["a medical chest x-ray", "a random object, tree, animal, or person"]
        )
        
        if filter_result[0]['label'] != "a medical chest x-ray":
            return "⚠️ عذراً، يبدو أن الصورة المدخلة ليست صورة أشعة سينية للصدر (Chest X-Ray). الرجاء التحقق من الصورة."

        # مرحلة التشخيص
        image_rgb = image.convert("RGB")
        inputs = processor(images=image_rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = model(inputs["pixel_values"])
            probs = torch.nn.functional.softmax(outputs, dim=-1)[0]

        results = {}
        for i, class_name in enumerate(CLASSES):
            conf = float(probs[i])
            results[class_name] = conf

        predicted_class = max(results, key=results.get)
        confidence_val = results[predicted_class] * 100
        status = "Abnormal ⚠️" if predicted_class != "Normal" else "Healthy ✅"

        output_text = f"الحالة: {status}\nالتشخيص المتوقع: {predicted_class}\nنسبة الثقة: {confidence_val:.1f}%\n\nتفاصيل كافة الاحتمالات:\n"
        for cls, conf in results.items():
            output_text += f"- {cls}: {conf*100:.1f}%\n"

        return output_text

    except Exception as e:
        return f"حدث خطأ أثناء المعالجة: {e}"

# واجهة Gradio 
demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil", label="ارفع صورة أشعة الصدر (Chest X-Ray)"),
    outputs=gr.Textbox(label="نتيجة التشخيص الذكي"),
    title="Smart Chest X-Ray Diagnosis System",
    description="نظام تحليل الأشعة الذكي: يحلل الصورة ويعطيك الحالة والتشخيص ونسب الثقة مباشرة."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
