import tensorflow as tf



class DetectSmileModel:
    def __init__(self, data_dir, img_size=(100, 100), batch_size=30):
        self.initial_epochs = 10
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.history = None
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.model = None
        self.class_names = None

    def preprocess_input(self, x):
        return tf.keras.applications.resnet50.preprocess_input(x)

    def build_model(self):
        # Load the training dataset
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        self.class_names = self.train_dataset.class_names
        print(self.class_names)

        # Load the validation dataset
        self.validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        # Create the test dataset
        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        self.test_dataset = self.validation_dataset.take(val_batches // 5)
        validation_dataset = self.validation_dataset.skip(val_batches // 5)

        # Improve performance using prefetching
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

        # Load the base VGG16 model
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=self.img_size + (3,), include_top=False,
                                                       weights="imagenet")

        # Freeze the base model layers
        base_model.trainable = False

        # Build the model architecture
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        x = self.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        # Evaluate the model on the validation dataset
        loss0, accuracy0 = self.model.evaluate(validation_dataset)
        print("Initial loss: {:.2f}".format(loss0))
        print("Initial accuracy: {:.2f}".format(accuracy0))

        # Train the model
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.initial_epochs,
            validation_data=validation_dataset,
            validation_steps=len(validation_dataset),
            steps_per_epoch=len(self.train_dataset),
        )

        # Enable fine-tuning
        base_model.trainable = True

        # Fine-tune from a specific layer onwards
        fine_tune_at = 10

        # Freeze layers before the fine-tuning layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile the model for fine-tuning
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.summary()

    def fine_tune(self, epochs):
        total_epochs = self.initial_epochs + epochs

        # Fine-tune the model
        history_fine = self.model.fit(self.train_dataset,
                                      epochs=total_epochs,
                                      initial_epoch=self.history.epoch[-1],
                                      validation_data=self.validation_dataset)
        return history_fine

    def report(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        return acc, val_acc, loss, val_loss

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_dataset)
        print('Test accuracy :', accuracy)

    def save_model(self, save_path):
        if self.model is not None:
            self.model.save(save_path)
            print("Model saved successfully.")
        else:
            print("No model to save.")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

    def predict(self, image):
        if self.model is not None:
            image = tf.expand_dims(image, 0)
            image = self.preprocess_input(image)
            prediction = self.model.predict(image)[0][0]
            if prediction < 0.5:
                return "No Smile"
            else:
                return "Smile"
        else:
            print("No model loaded.")

    def prediction(self):
        # Retrieve a batch of images from the test set
        image_batch, label_batch = self.test_dataset.as_numpy_iterator().next()
        predictions = self.model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)


if __name__ == "__main__":
    # Example usage:
    model = DetectSmileModel(data_dir='data')
    model.build_model()
    model.fine_tune(10)
    model.save_model('model.h5')

    # # test for an image
    # model.load_model('mobilenet_model.h5')
    # # Perform prediction on an image
    # image = tf.keras.preprocessing.image.load_img('p.jpg', target_size=(100, 100))
    # prediction = model.predict(image)
    # print(prediction)
