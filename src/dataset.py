import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticChestXRayDataset(Dataset):
    """
    Προσομοιωτής ακτινογραφιών θώρακα:
    - Κλάση 0: Φυσιολογική Ακτινογραφία (Normal)
    - Κλάση 1: Πνευμονία / Αδιαφάνεια (Pneumonia Pathology)
    """
    def __init__(self, num_samples: int = 160, img_size: int = 224, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.img_size = img_size
        self.samples = []
        self.labels = []

        for _ in range(num_samples):
            label = np.random.choice([0, 1])
            # Βασικό υπόβαθρο θώρακα με θόρυβο
            base = np.random.normal(loc=0.15, scale=0.03, size=(img_size, img_size))
            
            # Προσομοίωση 2 πνευμονικών πεδίων (σκοτεινότερες ελλειπτικές περιοχές)
            y, x = np.ogrid[:img_size, :img_size]
            left_lung = ((x - 70)**2 / 30**2 + (y - 110)**2 / 60**2) < 1.0
            right_lung = ((x - 154)**2 / 30**2 + (y - 110)**2 / 60**2) < 1.0
            base[left_lung | right_lung] += 0.35

            # Αν είναι παθολογικό δείγμα, εισάγουμε εστιακή σκίαση (pneumonia consolidation)
            if label == 1:
                cx, cy = np.random.choice([75, 145]), np.random.randint(90, 140)
                patch = ((x - cx)**2 / 18**2 + (y - cy)**2 / 18**2) < 1.0
                base[patch] += 0.45

            base = np.clip(base, 0.0, 1.0).astype(np.float32)
            # 3-channel input για το ResNet50
            img_rgb = np.stack([base, base, base], axis=0)

            self.samples.append(img_rgb)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
