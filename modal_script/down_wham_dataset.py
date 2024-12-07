from modal import Image, App, Mount, Volume

image = (
    Image.debian_slim(python_version="3.11").apt_install("curl")
    .run_commands("curl -L -O https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip")
)

volume = Volume.from_name(
    "wham", create_if_missing=True
)


app = App()
wham_dir = "/wham"
@app.function(image=image, gpu=None, volumes={wham_dir: volume}, timeout=3600 * 12)
def down_dataset():
    import zipfile
    with zipfile.ZipFile("/wham_noise.zip", 'r') as zip_ref:
        zip_ref.extractall(wham_dir+'/')
    volume.commit()

@app.local_entrypoint()
def main():
    down_dataset.remote()