# CSC542 Group Project

## Dataset

This project uses the [Kvasir Dataset for Classification and Segmentation](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation) from Kaggle.

### Download via Kaggle CLI

Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → API → Create New Token, then copy the generated `username` and `key` values and export them as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Then download the dataset:

```bash
kaggle datasets download abdallahwagih/kvasir-dataset-for-classification-and-segmentation
```

### Extract and Place Files

After downloading, unzip and place the data under `data/` as follows:

```bash
unzip kvasir-dataset-for-classification-and-segmentation.zip -d data/
```

The expected directory structure is:

```
Kvasir/
└── data/
    ├── kvasir-dataset/
    │   └── kvasir-dataset/          # 8 class sub-folders (500 images each)
    │       ├── dyed-lifted-polyps/
    │       ├── dyed-resection-margins/
    │       ├── esophagitis/
    │       ├── normal-cecum/
    │       ├── normal-pylorus/
    │       ├── normal-z-line/
    │       ├── polyps/
    │       └── ulcerative-colitis/
    └── kvasir-seg/
        └── Kvasir-SEG/              # Segmentation dataset
            ├── images/              # 1000 RGB images
            └── masks/               # 1000 binary masks
```

---

## Setup

```bash
conda create -n kvasir python=3.10 -y
conda activate kvasir
pip install -r requirements.txt
```

---

## Run Command

```bash
# Classification (ResNet-18)
python -m classification.train \
    --data_dir data/kvasir-dataset/kvasir-dataset \
    --output_dir outputs/classification

# Segmentation (U-Net)
python -m segmentation.train \
    --data_dir data/kvasir-seg/Kvasir-SEG \
    --output_dir outputs/segmentation
```
