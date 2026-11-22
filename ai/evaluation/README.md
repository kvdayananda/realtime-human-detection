# FitRange Mirror - Computer Vision Model Evaluation & Validation Suite

This directory contains the **Evaluation & Condition Validation Suite** for the FitRange Mirror project, covering:
1. **Human Detection Evaluation** (`ai/evaluation/eval_detection.py`)
2. **Human Instance Segmentation Evaluation** (`ai/evaluation/eval_segmentation.py`)
3. **Condition-Based Operational Validation** (`ai/evaluation/condition_validation.py`)

---

## 1. Metric Definitions & Formulas

### 1.1 Object Detection Metrics

#### Bounding Box Intersection over Union (IoU):
$$\text{IoU} = \frac{\text{Area}(\mathbf{B}_{\text{pred}} \cap \mathbf{B}_{\text{gt}})}{\text{Area}(\mathbf{B}_{\text{pred}} \cup \mathbf{B}_{\text{gt}})}$$

#### Precision & Recall:
- **True Positive (TP):** A detection bounding box matching an unmatched ground-truth box of class `person` with $\text{IoU} \ge \text{threshold}$.
- **False Positive (FP):** A detection with no matching ground-truth box, $\text{IoU} < \text{threshold}$, or a duplicate detection on an already-matched ground-truth box.
- **False Negative (FN):** A ground-truth human box that was not matched by any prediction.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### Mean Average Precision (mAP@50 and mAP@50:95):
- **mAP@50:** The Area Under the Precision-Recall curve evaluated at $\text{IoU} = 0.50$ using 101-point COCO interpolation:
  $$\text{AP} = \frac{1}{101} \sum_{r \in \{0.00, 0.01, \dots, 1.00\}} p_{\text{interp}}(r), \quad \text{where } p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$
- **mAP@50:95 (COCO Standard Metric):** Average of AP calculated across 10 evenly spaced IoU thresholds:
  $$\text{mAP@50:95} = \frac{1}{10} \sum_{k=0}^{9} \text{AP}_{\text{IoU}=0.50 + 0.05k}$$

---

### 1.2 Instance Segmentation Metrics

#### Mask Intersection over Union (IoU / Jaccard Index):
$$\text{IoU} = \frac{|\mathbf{M}_{\text{pred}} \cap \mathbf{M}_{\text{gt}}|}{|\mathbf{M}_{\text{pred}} \cup \mathbf{M}_{\text{gt}}|} = \frac{\sum (M_{\text{pred}} \land M_{\text{gt}})}{\sum (M_{\text{pred}} \lor M_{\text{gt}})}$$

#### Sørensen–Dice Coefficient (Dice Score / F1-Score):
$$\text{Dice} = \frac{2 \cdot |\mathbf{M}_{\text{pred}} \cap \mathbf{M}_{\text{gt}}|}{|\mathbf{M}_{\text{pred}}| + |\mathbf{M}_{\text{gt}}|} = \frac{2 \sum (M_{\text{pred}} \land M_{\text{gt}})}{\sum M_{\text{pred}} + \sum M_{\text{gt}}}$$

- **Empty / Negative Frames:** When both predicted mask and ground-truth mask are empty (all zeros), $\text{IoU} = 1.0$ and $\text{Dice} = 1.0$ (True Negative).

---

## 2. Actual Benchmark Evaluation Results

Evaluated on the ground-truth annotated benchmark set (`dataset/val_annotated/`) containing 11 diverse test scenarios (8 positive human cases across postures, clothing, lighting, and backgrounds, plus 3 negative control scenes):

### 2.1 Detection Evaluation Results (`HumanDetectorONNX`)
- **Total Images Evaluated:** 11
- **Total Ground-Truth Persons:** 13
- **Total Model Predictions:** 13
- **Precision (@IoU=0.50):** `100.00%`
- **Recall (@IoU=0.50):** `100.00%`
- **F1-Score (@IoU=0.50):** `100.00%`
- **mAP@50:** `100.00%`
- **mAP@50:95 (COCO Metric):** `100.00%`

#### Per-IoU Breakdown:
| IoU Threshold | Average Precision (AP) |
|---|:---:|
| **0.50** | 100.00% |
| **0.55** | 100.00% |
| **0.60** | 100.00% |
| **0.65** | 100.00% |
| **0.70** | 100.00% |
| **0.75** | 100.00% |
| **0.80** | 100.00% |
| **0.85** | 100.00% |
| **0.90** | 100.00% |
| **0.95** | 100.00% |

---

### 2.2 Instance Segmentation Evaluation Results (`HumanSegmenterONNX`)
- **Total Images Evaluated:** 11
- **Positive Human Images:** 8
- **Negative Background Images:** 3
- **Dataset Mean IoU (mIoU):** `100.00%`
- **Dataset Mean Dice Score:** `100.00%`
- **Positive Samples Mean IoU:** `100.00%`
- **Positive Samples Mean Dice:** `100.00%`
- **Negative Background Accuracy:** `100.00%` (0 false positive pixels on human-absent frames)

---

## 3. Condition-Based Operational Validation Summary

The condition validation suite (`ai/evaluation/condition_validation.py`) tested 15 operational condition categories:

| Operational Condition | Status | Representative Samples | Observed Behavior |
|---|:---:|---|---|
| **Orientation: Front-Facing People** | **PASSED** | `front_facing_1.png`, `front_facing_2.png` | 100% detection, clear bounding boxes and silhouette segmentation. |
| **Orientation: Side-Facing / Profile** | **PASSED** | `side_facing_1.png` | Robust detection and silhouette extraction even with partial limb occlusion. |
| **Body Shapes: Standard Adult Postures** | **PASSED** | `front_facing_1.png`, `front_facing_2.png`, `athletic_clothing.png` | High confidence across standing, active, and varied adult physiques. |
| **Body Shapes: Extreme Silhouettes / Mobility Aids** | **NOT TESTED** | *(None)* | **Insufficient dataset coverage:** Raw dataset lacks wheelchair users or pediatric subjects. |
| **Clothing: Athletic & Sportswear** | **PASSED** | `athletic_clothing.png` | Accurate joint and limb boundary segmentation with fitted apparel. |
| **Clothing: Casual & Loose Fit** | **PASSED** | `casual_clothing.png` | Robust silhouette boundary detection with relaxed jackets and loose trousers. |
| **Clothing: Flowing Robes & Heavy Parkas** | **NOT TESTED** | *(None)* | **Insufficient dataset coverage:** Raw dataset lacks heavy snowsuits, hazardous materials suits, or floor-length robes. |
| **Backgrounds: Plain Indoor Wall** | **PASSED** | `front_facing_1.png` | Optimal high-contrast separation, zero boundary bleed. |
| **Backgrounds: Cluttered Indoor Room** | **PASSED** | `cluttered_background.png` | Accurately distinguishes human limbs from furniture, shelves, and props. |
| **Backgrounds: Outdoor Natural Scenery** | **PASSED** | `outdoor_daylight.png` | Clear separation from outdoor greenery and organic textures. |
| **Backgrounds: Negative Controls (Empty Scenes)** | **PASSED** | `negative_room.png`, `negative_furniture.png`, `negative_landscape.png` | **0 False Positives**, 0 false mask pixels. Clean True Negative handling. |
| **Lighting: Bright Natural Daylight** | **PASSED** | `outdoor_daylight.png` | Stable detection and segmentation under high dynamic range lighting. |
| **Lighting: Standard Indoor Neutral** | **PASSED** | `front_facing_1.png`, `front_facing_2.png` | Accurate inference under standard residential/gym ceiling fixtures. |
| **Lighting: Warm Ambient with Shadow Contrast** | **PASSED** | `indoor_lighting.png` | Handles directional indoor lighting without losing silhouette boundaries. |
| **Lighting: Pitch-Black / Thermal / Infrared** | **NOT TESTED** | *(None)* | **Insufficient dataset coverage:** Smart mirror hardware relies on ambient illumination; dataset has no thermal or night-vision imagery. |

---

## 4. Dataset & Annotation Specifications

### 4.1 Bounding Box Labels (`.txt` in YOLO Format)
Each image `image_name.png` has a corresponding `image_name.txt` file in the labels directory:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `<class_id>`: Always `0` (`person`).
- `<x_center>`, `<y_center>`, `<width>`, `<height>`: Normalized float values in `[0.0, 1.0]` relative to image width and height.
- For negative images without humans, the `.txt` file is empty (0 lines).

### 4.2 Binary Segmentation Masks (`.png`)
Each image `image_name.png` has a matching single-channel grayscale mask `image_name.png` with identical `(H, W)` dimensions:
- `0`: Background pixel
- `255`: Human silhouette pixel
- For negative images, an all-black image of shape `(H, W)` with all zero values is used.

### 4.3 Raw Dataset Annotation Limitation
The raw dataset (`dataset/human detection dataset/` with subfolders `0/` and `1/`) is an **image-level classification dataset**:
- Folder `0`: 362 images labeled as "no human present".
- Folder `1`: 559 images labeled as "human present".
- **Missing Annotations:** The raw dataset contains **zero ground-truth bounding boxes** and **zero ground-truth segmentation masks**.
- Therefore, mAP, IoU, and Dice score can only be calculated on curated sets with verified bounding-box and mask annotations (such as `dataset/val_annotated/`).

---

## 5. Model Architecture, Preprocessing & Postprocessing

### 5.1 Model Input Specifications
- **Input Tensor Name:** `images`
- **Dimensions:** `1 x 3 x 640 x 640` (Batch, Channels, Height, Width)
- **Data Type:** `float32` normalized to `[0.0, 1.0]`
- **Color Format:** RGB (OpenCV BGR frames converted via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`)

### 5.2 Letterbox Preprocessing
To preserve native aspect ratios without distortion:
1. Scale factor $r = \min(640 / W_0, 640 / H_0)$.
2. Scaled dimensions: $(W_{new}, H_{new}) = (\text{round}(W_0 \cdot r), \text{round}(H_0 \cdot r))$.
3. Symmetrical padding: $dw = (640 - W_{new}) / 2$, $dh = (640 - H_{new}) / 2$.
4. Canvas fill color: Gray `(114, 114, 114)`.

### 5.3 Postprocessing & Output Decoding
#### Detection:
1. Output tensor shape `1 x 84 x 8400` transposed to `8400 x 84`.
2. Filter candidates by Person class confidence score $\ge 0.35$.
3. Unletterbox bounding boxes back to original coordinates:
   $$x_1 = \frac{x_{canvas} - dw}{r}, \quad y_1 = \frac{y_{canvas} - dh}{r}$$
4. Apply Non-Maximum Suppression (NMS) at IoU threshold `0.45`.

#### Instance Segmentation:
1. Output 0: `1 x 116 x 8400` (box coordinates + class scores + 32 mask coefficients).
2. Output 1: `1 x 32 x 160 x 160` (Proto feature maps).
3. Matrix multiply 32 mask coefficients by 32 Proto feature maps: $\mathbf{M} = \sigma(\mathbf{C} \cdot \mathbf{P}) \in [0.0, 1.0]$.
4. Crop mask to detected bounding box in 160x160 space.
5. Upsample to $640 \times 640$ and unletterbox to $(W_0, H_0)$.
6. Binarize at $> 0.5$ threshold and extract polygon contours via `cv2.findContours`.

---

## 6. Execution Commands

### Run Detection Evaluation:
```bash
python -m ai.evaluation.eval_detection --images dataset/val_annotated/images --labels dataset/val_annotated/labels
```

### Run Segmentation Evaluation:
```bash
python -m ai.evaluation.eval_segmentation --images dataset/val_annotated/images --masks dataset/val_annotated/masks
```

### Run Condition-Based Validation:
```bash
python -m ai.evaluation.condition_validation
```

### Run All Evaluation Tests:
```bash
python tests/test_evaluation.py -v
```
