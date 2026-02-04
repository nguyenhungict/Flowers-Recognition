# Notebook: 01_data_exploration.ipynb (Guidance)

This notebook should:
- Inspect the `E:/flowers` folder structure
- Show sample images from several classes
- Count images per class
- Optionally create a CSV manifest

Example code snippets (run in Jupyter):

import os
from PIL import Image
import matplotlib.pyplot as plt
import random

root = 'E:/flowers'  # or update path
classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
print('Classes:', classes)
for c in classes:
    print(c, len(os.listdir(os.path.join(root,c))))

# show samples
for c in classes[:5]:
    imgs = os.listdir(os.path.join(root,c))
    sample = random.choice(imgs)
    img = Image.open(os.path.join(root,c,sample))
    plt.imshow(img); plt.title(c); plt.axis('off'); plt.show()

# run & predict
    python src\train.py --config config.yaml
    python src\predict.py --image "C:\Users\HP\Downloads\tulip2.jpg" --model models\checkpoints\best_model_7classes.pth --topk 3

