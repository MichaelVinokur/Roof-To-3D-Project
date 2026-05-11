from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="blanchon/INRIA-Aerial-Image-Labeling",
    repo_type="dataset",
    local_dir="./inria",
    allow_patterns=["*austin*"]  # only Austin images + masks
)