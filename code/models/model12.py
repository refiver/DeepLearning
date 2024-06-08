from tensorflow.keras import layers, models


def initialize_model():
    """This model uses bigger kernels for the convolution layers compared to the original model."""

    # creates the model
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(9, activation='softmax')
    ])

    # compiles the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
