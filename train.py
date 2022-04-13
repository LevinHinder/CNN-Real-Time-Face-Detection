from tensorflow import keras

dataset_path = r"C:\Users\Levin\Desktop\Faces Dataset balanced 2"
image_size = (250, 250)

validation_ratio = 0.15
batch_size = 16
epochs = 50

train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_ratio,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_ratio,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size)

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(image_size[0], image_size[1], 3))

for layer in base_model.layers:
    layer.trainable = False

global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(1, activation="sigmoid")(
    global_avg_pooling)

model = keras.models.Model(
    inputs=base_model.input,
    outputs=output,
    name="ResNet50")

# ModelCheckpoint to save model in case of interrupting the learning process
checkpoint = keras.callbacks.ModelCheckpoint(
    "models/face_classifier 2.1.{epoch}.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1)

# EarlyStopping to find best model with a large number of epochs
earlystop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    restore_best_weights=True,
    patience=3,
    verbose=1)

callbacks = [earlystop, checkpoint]

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"])

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds)

model.save("models/face_classifier 2.1.final.h5")
