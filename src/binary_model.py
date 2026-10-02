import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from utils.config import IMG_SIZE, DROPOUT_RATE


BINARY_NUM_CLASSES = 2


def build_binary_model(
    input_shape=(*IMG_SIZE, 3),
    num_classes=BINARY_NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
):
    """
    EfficientNetB0 binary skin cancer classifier.

    Class mapping:
        0 -> non_cancer
        1 -> cancer
    """

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze EfficientNet backbone initially
    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout_layer"
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="prediction_layer"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    return model


if __name__ == "__main__":

    model = build_binary_model()

    print("\nBinary model created successfully.")
    print("Model output shape:", model.output_shape)

    assert model.output_shape == (
        None,
        2
    ), "Model output must be (None, 2)"

    print("\nClass mapping:")
    print("0 -> non_cancer")
    print("1 -> cancer")

    print("\nModel verification successful.")