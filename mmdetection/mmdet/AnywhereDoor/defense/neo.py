import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

class Neo:
    def __init__(self,
                 loc: Tuple[int, int] = (0, 0),
                 trigger_size: Tuple[int, int] = (30, 30)):
        self.loc = loc
        self.trigger_size = trigger_size

    def _apply_mask(self, img: np.ndarray, color: tuple) -> np.ndarray:
        x, y = self.loc
        if img.shape[2] == 4:
            img[x:x+self.trigger_size[0], y:y+self.trigger_size[1], 3] = 0
        else:  # RGB
            img[x:x+self.trigger_size[0], y:y+self.trigger_size[1]] = color
        return img

    def _get_dominant_color(self, img: np.ndarray) -> tuple:
        pixels = img.reshape(-1, img.shape[-1])
        if pixels.shape[1] == 4:
            alpha = pixels[:, 3:]
            pixels = pixels[alpha > 0.5]
        
        kmeans = KMeans(n_clusters=3).fit(pixels)
        counts = np.bincount(kmeans.labels_)
        return tuple(kmeans.cluster_centers_[np.argmax(counts)].astype(int))

    def defend(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(1, 2, 0).cpu().detach().numpy()

        dominant_color = self._get_dominant_color(image)
        modified_img = self._apply_mask(image, dominant_color)

        return torch.tensor(modified_img).permute(2, 0, 1)