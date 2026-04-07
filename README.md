# CSC542 Group Project

## Run Command

```
conda create -n kvasir python=3.10 -y

conda activate kvasir

pip install -r requirements.txt

python -m classification.train \
    --data_dir data/kvasir-dataset/kvasir-dataset \
    --output_dir outputs/classification


python -m segmentation.train \
  --data_dir data/kvasir-seg/Kvasir-SEG \
  --output_dir outputs/segmentation
