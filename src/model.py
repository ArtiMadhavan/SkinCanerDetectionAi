import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)

from utils.config import (
    IMG_SIZE,
    DROPOUT_RATE,
    CLASSES
)


# ============================================================
# 7-CLASS MODEL
# ============================================================

NUM_CLASSES = len(CLASSES)


def build_model(
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
):
    """
    Builds EfficientNetB0 for 7-class HAM10000 classification.

    Class mapping:

        0 -> mel
        1 -> bcc
        2 -> akiec
        3 -> nv
        4 -> bkl
        5 -> df
        6 -> vasc
    """

    # --------------------------------------------------------
    # EfficientNetB0 backbone
    # --------------------------------------------------------

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze backbone for Stage 1
    base_model.trainable = False

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = base_model.output

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout_layer"
    )(x)

    # --------------------------------------------------------
    # 7 CLASS OUTPUT
    # --------------------------------------------------------

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="prediction_layer"
    )(x)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = build_model()

    model.summary()

    print(
        f"\nModel output shape: "
        f"{model.output_shape}"
    )

    # Expected:
    #
    # (None, 7)

    assert model.output_shape == (
        None,
        NUM_CLASSES
    ), (
        f"Output shape must be "
        f"(None, {NUM_CLASSES})"
    )

    print(
        "\n7-class model shape verification successful."
    )

    print("\nClass mapping:")

    for index, class_name in enumerate(CLASSES):

        print(
            f"{index} -> {class_name}"
        )