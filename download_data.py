from pathlib import Path

import kagglehub

data_dir = Path.cwd() / "data"
print(data_dir)
path = kagglehub.dataset_download(
    "marquis03/vehicle-classification",
    output_dir=str(data_dir),
)
print("Path to dataset files:", path)
