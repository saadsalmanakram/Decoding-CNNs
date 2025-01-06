
---

**ResNet**, or **Residual Network**, was introduced by **Kaiming He** et al. in the **2015** paper "Deep Residual Learning for Image Recognition." ResNet is one of the most influential architectures in the field of deep learning, particularly for **computer vision** tasks. It significantly advanced the state-of-the-art in deep learning by enabling the training of extremely deep networks, which was previously limited due to the vanishing/exploding gradient problems.

### The core innovation in ResNet is the concept of **residual connections** (or skip connections), which help the network learn residual mappings instead of direct mappings. This enables **very deep networks** (with hundreds or even thousands of layers) to be trained effectively.

---

## 📌 **1. Overview of ResNet**

### Key Contributions of ResNet:
- **Residual Connections (Skip Connections):** ResNet introduces skip connections that allow gradients to flow more easily through the network, helping alleviate the vanishing gradient problem.
- **Deep Networks:** With residual connections, ResNet can train much deeper models (e.g., ResNet-50, ResNet-101, ResNet-152), reaching up to **1000+ layers**.
- **Ease of Training:** The use of residual connections allows the network to learn residual functions, making deeper networks trainable without the need for complex tricks or careful initialization.

### ResNet Variants:
ResNet comes in multiple depths:
- **ResNet-18:** 18 layers.
- **ResNet-34:** 34 layers.
- **ResNet-50:** 50 layers.
- **ResNet-101:** 101 layers.
- **ResNet-152:** 152 layers.
- The "number" after ResNet indicates the number of layers in the network.

---

## 📌 **2. Architecture of ResNet**

The ResNet architecture consists of **convolutional layers**, **batch normalization layers**, **ReLU activation functions**, and **residual blocks**. A **residual block** is the key unit of ResNet. A residual block allows an input to "skip" one or more layers and be added directly to the output of those layers, hence the term "residual."

Here’s how ResNet-50 is structured:

### 1. **Input Layer:**
   - Input image size: 224x224x3 (RGB).
   - The input is first passed through a convolutional layer with **64 filters of size 7x7** and a stride of 2.

### 2. **Convolutional Layer:**
   - The first convolutional layer has a kernel size of 7x7 and uses 64 filters. The stride is set to 2, reducing the spatial size of the image.
   - **Max-Pooling:** The network follows this with a **3x3 max-pooling layer** with a stride of 2 to further reduce the image dimensions.

### 3. **Residual Blocks:**
   - After the initial convolution, ResNet consists of several residual blocks arranged in stages. Each stage has multiple residual blocks.
   - **Basic Block:** For smaller ResNets (ResNet-18, ResNet-34), each residual block is a simple **two-layer block** with 3x3 convolutions.
   - **Bottleneck Block:** For deeper networks (ResNet-50, ResNet-101, ResNet-152), each residual block is a **three-layer bottleneck block**, which reduces the dimensionality of the intermediate feature maps before expanding them again.

#### Example of a Residual Block:
A basic residual block in ResNet is defined as:

\[
y = F(x, \{W_i\}) + x
\]

Where:
- \( x \) is the input.
- \( F(x, \{W_i\}) \) represents the function (usually convolutions, batch normalization, ReLU) applied to the input \( x \).
- The skip connection adds the input \( x \) to the output of the function, forming the residual mapping.

### 4. **Final Layers:**
   - After the residual blocks, the network typically ends with **global average pooling** to reduce the feature map dimensions.
   - A **fully connected layer** (or linear layer) maps the feature vector to the final class scores.

### 5. **Output Layer (Softmax):**
   - A softmax layer is used at the end of the network to produce class probabilities.

### **ResNet-50 Architecture:**
For ResNet-50, the architecture can be broken down into **5 stages**, each containing a specific number of residual blocks:

1. **Initial Convolution + Max-Pooling:**
   - Conv1: 64 filters (7x7), stride 2 → Output: 112x112x64.
   - Max Pooling: 3x3, stride 2 → Output: 56x56x64.

2. **Stage 1 (3 residual blocks):**
   - Conv2_x: Each block has 3 layers.
   - Output: 56x56x256.

3. **Stage 2 (4 residual blocks):**
   - Conv3_x: Each block has 3 layers.
   - Output: 28x28x512.

4. **Stage 3 (6 residual blocks):**
   - Conv4_x: Each block has 3 layers.
   - Output: 14x14x1024.

5. **Stage 4 (3 residual blocks):**
   - Conv5_x: Each block has 3 layers.
   - Output: 7x7x2048.

6. **Final Layers:**
   - Global Average Pooling → 1x1x2048.
   - Fully connected layer → 1000 (for ImageNet classification).
   - Softmax Output.

---

## 📌 **3. Key Features of ResNet**

- **Residual Connections:** ResNet's key innovation is the introduction of residual (skip) connections. These connections allow the model to learn the identity function (skip transformation) if deeper layers do not improve performance. This makes it easier to train deeper networks by mitigating the vanishing gradient problem.
- **Deeper Architectures:** ResNet has demonstrated that deeper networks (up to **152 layers** and beyond) can be trained effectively.
- **Bottleneck Layers:** In deeper models like ResNet-50, ResNet-101, and ResNet-152, **bottleneck blocks** are used to reduce computational complexity and the number of parameters while maintaining performance.
- **Global Average Pooling:** Instead of using fully connected layers for all the features, ResNet uses **global average pooling**, which helps reduce the number of parameters and overfitting.

---

## 📌 **4. ResNet in PyTorch**

### 🧑‍💻 **ResNet-50 PyTorch Implementation**

Here’s how to implement **ResNet-50** in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load pre-trained ResNet-50 model from torchvision
model = torchvision.models.resnet50(pretrained=True)

# Modify the final fully connected layer for custom classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1000)  # Change output layer for 1000 classes (ImageNet)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Example training loop (assuming datasets are loaded)
for epoch in range(10):  # Example: training for 10 epochs
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # trainloader is the data loader
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 99:  # Print every 100 batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")
```

---

## 📌 **5. Contributions and Impact of ResNet**

- **Training Deep Networks:** ResNet revolutionized the training of deep networks, allowing for effective training of networks with hundreds of layers, such as ResNet-152.
- **High Performance on Benchmarks:** ResNet achieved groundbreaking results on the **ImageNet** challenge, winning the 2015 competition with a top-5 error rate of 3.57%.
- **Wide Adoption:** ResNet architectures have become the backbone for many tasks in computer vision, including object detection, segmentation, and transfer learning.

---
