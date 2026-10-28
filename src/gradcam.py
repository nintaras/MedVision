import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM2D:
    """
    2D Grad-CAM implementation για ανάλυση ερμηνευσιμότητας σε CNNs.
    Καταγράφει activations και gradients από το layer4 του ResNet50.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registration των hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Υπολογίζει το Grad-CAM heatmap για τη δεδομένη εικόνα εισόδου.
        """
        self.model.eval()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        loss = logits[0, target_class]
        loss.backward()

        # Global average pooling στα gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)    # (H, W)

        # ReLU για διατήρηση μόνο των θετικών συνεισφορών
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Κανονικοποίηση στο [0, 1]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam, target_class

def overlay_heatmap(img_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Υπερθέτει το θερμογράφημα πάνω στην αρχική ιατρική εικόνα.
    """
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if img_np.max() <= 1.0:
        img_np = np.uint8(255 * img_np)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
