import glob, random
import cv2
import numpy as np

random.seed(0)
label_files = [f for f in glob.glob("data/raw/idd20k_lite/gtFine/train/**/*_label.png", recursive=True)
               if not f.endswith("_inst_label.png")]
sample = random.sample(label_files, 50)

# 1. Numbers: how much of each image is class 0 vs class 1?
frac0, frac1 = [], []
for f in sample:
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    frac0.append((m == 0).mean())
    frac1.append((m == 1).mean())
print(f"class 0 mean fraction: {np.mean(frac0):.3f}")
print(f"class 1 mean fraction: {np.mean(frac1):.3f}")

# 2. Picture: paint class 0 green and class 1 red on the photo with the most class-1 pixels
best = sample[int(np.argmax(frac1))]
img_path = best.replace("gtFine", "leftImg8bit").replace("_label.png", "_image.jpg")
img = cv2.imread(img_path)
m = cv2.imread(best, cv2.IMREAD_GRAYSCALE)

overlay = img.copy()
overlay[m == 0] = (0, 200, 0)   # BGR green
overlay[m == 1] = (0, 0, 255)   # BGR red
out = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
cv2.imwrite("idd_class_check.jpg", out)
print("saved idd_class_check.jpg from", img_path)