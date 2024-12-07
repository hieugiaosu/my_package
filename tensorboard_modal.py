import modal
from modal import App, Image, Volume

# Define image and volume

dockerhub_image = (
    Image.from_registry(
    "tensorflow/tensorflow:2.15.0-gpu",
).pip_install("tensorboard")
)

app = App("tensorboard-test")

model_volume = Volume.from_name("reproduce", create_if_missing=True)
model_dir = "/model"


@app.function(volumes={model_dir: model_volume},gpu=None,image=dockerhub_image)
@modal.wsgi_app()
def main():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=f"{model_dir}/tensorboard_logs")
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app