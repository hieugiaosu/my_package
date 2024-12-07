from modal import Image, App, Mount, Volume
import os

# Define the image with the required dependencies
image = (
    Image.debian_slim(python_version="3.11").apt_install("curl")
    .run_commands("curl -L -O https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz")
)

# Define the volume to store the dataset
volume = Volume.from_name("librispeech", create_if_missing=True)

# Set up the app
app = App()

# Define the directory to mount the volume
libri_dir = "/librispeech"

# Define the function that downloads and extracts the dataset
@app.function(image=image, gpu=None, volumes={libri_dir: volume}, timeout=3600 * 24)
def down_dataset():
    import tarfile
    
    # Extract the tar.gz file
    tar_path = "/train-clean-100.tar.gz"
    extract_path = os.path.join(libri_dir, "train-clean-100")
    
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_path)
    
    # Commit changes to the volume
    volume.commit()

# Define the local entry point
@app.local_entrypoint()
def main():
    down_dataset.remote()
