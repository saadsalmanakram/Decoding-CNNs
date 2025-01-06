# Step 1: Define the Input

from tensorflow.keras import layers, models

input_shape = (224, 224, 3)  # For a 224x224 RGB image
model = models.Sequential()


# Step 2: Add Convolutional Layers

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


Step 3: Add Pooling Layers

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.MaxPooling2D((2, 2)))


Step 4: Add Flattening and Fully Connected Layers

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for classification


# Step 5: Compile the Model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Step 6: Train the Model

# Assuming `train_images` and `train_labels` are your dataset
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)
