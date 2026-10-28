import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import MedicalResNet50
from src.gradcam import GradCAM2D, overlay_heatmap
from src.dataset import SyntheticChestXRayDataset

def run_explainability():
    device = torch.device("cpu")  # CPU για σταθερή εξαγωγή hooks
    print("-> Φόρτωση εκπαιδευμένου ResNet-50 για 2D Grad-CAM...")

    # 1. Φόρτωση μοντέλου
    model = MedicalResNet50(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("best_medvision_resnet50.pth", map_location=device))
    model.eval()

    # Σύνδεση Grad-CAM στο τελευταίο συνελικτικό block (layer4)
    target_layer = model.backbone.layer4[-1].conv3
    grad_cam = GradCAM2D(model, target_layer)

    # 2. Εύρεση ενός θετικού δείγματος πνευμονίας (Class 1)
    dataset = SyntheticChestXRayDataset(num_samples=40, seed=12)
    sample_img = None
    true_label = None

    for img, lbl in dataset:
        if lbl.item() == 1:
            sample_img = img
            true_label = lbl.item()
            break

    input_tensor = sample_img.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 3. Παραγωγή Heatmap
    cam, pred_class = grad_cam.generate_cam(input_tensor, target_class=1)

    # Μετατροπή εικόνας για οπτικοποίηση (H, W, 3)
    raw_img = sample_img.permute(1, 2, 0).numpy()
    visual_overlay = overlay_heatmap(raw_img, cam, alpha=0.45)

    class_names = ["Normal", "Pneumonia"]

    # 4. Σχεδίαση τριπλού panel: Raw X-Ray, Heatmap, Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title(f"Original Chest X-Ray\n(True: {class_names[true_label]})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    im_cam = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("2D Grad-CAM Activation Map\n(Conv Layer4 Gradients)", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(visual_overlay)
    axes[2].set_title(f"Clinical Overlay Inspection\n(Predicted: {class_names[pred_class]})", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/gradcam_overlay.png", dpi=150)
    print("-> Το γράφημα explainability αποθηκεύτηκε στο 'assets/gradcam_overlay.png'.")
    plt.show()

if __name__ == "__main__":
    run_explainability()
