# Vehicle Classification with ResNet50

Run all commands from this project folder. 
## 1. Install dependencies

Skip creating the environment if `.venv` already exists.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Download the dataset

```bash
.venv/bin/python download_data.py
```

Images are saved in `data/`.

## 3. Train and validate

```bash
.venv/bin/python main.py
```

In another terminal:

```bash
.venv/bin/tensorboard --logdir results/tensorboard
```

Open http://localhost:6006 for loss and accuracy versus epoch.

## 5. Export figures

```bash
.venv/bin/python plot_loss.py
.venv/bin/python plot_predictions.py
```

# RBE577-F26-F01-HW2
