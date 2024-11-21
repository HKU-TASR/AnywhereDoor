import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCDetection
import torchvision.transforms as T

class VOCDataset(Dataset):
    def __init__(self, 
                root, 
                year='2007', 
                image_set='train', 
                download=False, 
                transform=T.Compose(
                    [
                    T.ToTensor(), 
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                ), 
                classes=[]):

        if year not in ["2007", "2012"]:
            raise ValueError("Year of VOC dataset must be '2007' or '2012'")
        if image_set not in ["train", "val", "trainval"]:
            raise ValueError("Image set must be 'train', 'val', or 'trainval'")

        if os.path.exists(os.path.join(root, "VOCdevkit", "VOC" + year)):
            download = False
            print("VOC dataset already exists")

        self.dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transform = transform
        self.classes = classes

        # If no classes are provided, use the default VOC classes
        if len(classes) == 0:
            self.classes = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", 
                "bus", "car", "cat", "chair", "cow", 
                "diningtable", "dog", "horse", "motorbike", "person", 
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

    def __getitem__(self, i):
        """
        Return value format:
        (   
            <PIL Image>, 
            {
                'boxes': <torch.Tensor>, # Shape (N, 4) (xmin, ymin, xmax, ymax)
                'confidences': <torch.Tensor>, # Shape (N,)
                'labels': <torch.Tensor> # Shape (N,)
            }
        )
        """

        img, target = self.dataset[i]
        boxes = []
        labels = []
        objects = target['annotation']['object']

        width_orig, height_orig = img.size
        img = self.transform(img)
        width_trans, height_trans = img.shape[2], img.shape[1]

        for obj in objects:
            name = obj['name']
            if name not in self.classes:
                continue
            label = self.classes.index(name)

            bndbox = obj['bndbox']
            box = [int(bndbox['xmin']) * width_trans / width_orig, 
               int(bndbox['ymin']) * height_trans / height_orig, 
               int(bndbox['xmax']) * width_trans / width_orig, 
               int(bndbox['ymax']) * height_trans / height_orig]

            labels.append(label)
            boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        confidences = torch.ones(len(labels), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        targets = {}
        targets["bboxes"] = boxes
        targets["confidences"] = confidences
        targets["labels"] = labels
        return img, targets

    def __len__(self):
        return len(self.dataset)