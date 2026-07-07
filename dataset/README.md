# 📦 Underwater Exposed Rebar Dataset

## 📜 Dataset Availability
A major reason we cannot provide our custom dataset here is its large size; including it with the code would cause us to exceed the official size limits for the code folder.

As this dataset is independently constructed by our team, team members reserve the right to determine its public availability. Given the significance of the dataset, we have decided not to release it to the public for the time being. Researchers interested in this dataset may contact team members to request access.

The detailed process of dataset collection and construction is described as follows:

---

## 📁 1. Dataset Collection

Dataset collection consists of two core steps: fabricating concrete members with exposed rebars and capturing footage in a simulated underwater environment.

Specifically, the team built concrete members in an experimental water tank, with portions of reinforcement intentionally exposed in advance. An underwater camera was then used to record videos under varying illumination and turbidity conditions.

| Item | Details |
| :--- | :--- |
| Video format | MP4 |
| Resolution | 1920×1080 |
| Total footage | Approximately 2 hours of video material |
| Coverage | Illumination intensities ranging from extremely dim to relatively bright (including uneven light distribution); turbidity levels from clear to turbid, ensuring favorable data diversity |

---

## 💡 2. Dataset Construction

After video acquisition, the team extracted frames from the videos at an interval of 15 frames. Following data cleaning, a total of **1051 original image frames** were retained.

### Train / Validation / Test Split
These 1051 images were split into a training set, a validation set, and a test set at a ratio of **6:3:1**:
- Training set: approximately 630 images
- Validation set: approximately 315 images
- Test set: approximately 106 images

### Data Augmentation
To enhance model generalization, data augmentation was performed on the training set. Each original training image was used to generate additional samples via multiple transformations:
- Rotation (±45 degrees)
- Horizontal flipping
- Scaling

The training set was expanded by 4 times to approximately 2520 images through augmentation.

### Final Dataset Scale
The final dataset comprises **2944 images in total**:
- Training set: 2520 images
- Validation set: 315 images
- Test set: 109 images

### Annotation & Export
All images were annotated frame by frame by team members using the Roboflow web-based annotation tool, with bounding boxes marking exposed rebar regions.
- Label class: `rebar` (rectangular bounding boxes drawn around clearly visible exposed rebar areas)
- Annotation format: YOLO format label files
- Export specification: All images were uniformly resized to 640×640 through proportional scaling and padding, yielding the raw image dataset.
