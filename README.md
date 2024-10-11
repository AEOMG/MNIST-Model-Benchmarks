#### Introduction
These codes are designed to use the MNIST dataset to explore how each model works, using various methods. This project employs several models to train and evaluate the MNIST dataset, helping users understand each model's structure and analyze the resulting outcomes.

#### **MNIST Dataset Overview**
The **MNIST (Mixed National Institute of Standards and Technology)** dataset is a popular benchmark dataset used for handwritten digit recognition tasks. It consists of **images of handwritten digits** ranging from **0 to 9** and contains a total of **70,000 samples** (60,000 for training and 10,000 for testing).

- **Image Size**: Each image is **28x28 pixels**, which can be represented as a **784-length vector**.
- **Labels**: Each image is labeled with an integer representing the digit, ranging from **0 to 9**.
- **Data Format**: The pixel values range from **0 to 255**, representing the intensity of each pixel, and are stored in `.gz` compressed files.

The MNIST dataset is widely used for testing and evaluating **machine learning algorithms**, and it is especially useful for beginners to understand and practice fundamental **neural network concepts**.

#### **Data Preprocessing Explanation and Understanding**
The preprocessing phase is essential for transforming the dataset into a format suitable for training the model. For the MNIST dataset, preprocessing plays a critical role in enhancing model performance and ensuring efficient training. The preprocessing steps are as follows:

1. **Data Loading**
   - The MNIST dataset is provided in **.gz compressed binary format**, which requires parsing to extract the image and label data.
   - During data loading, each image is loaded as a **28x28 pixel matrix**, which is subsequently **flattened into a 784-dimensional vector** for use as input to the model.
   - **Label data** is also loaded to indicate the corresponding digit for each image.

2. **Normalization**
   - **Normalization** is the process of transforming each feature value to the **[0, 1]** range. In the case of MNIST, each pixel value is initially an integer between **0 and 255**, which is then divided by **255.0** to convert it into a **float between 0 and 1**.
   - **The purpose of normalization** is to enhance the stability and efficiency of the training process by allowing the model to converge faster. By ensuring consistent input value ranges, normalization enables more efficient learning during weight adjustments.

3. **One-Hot Encoding**
   - The **label data**, originally provided as integers from 0 to 9, is converted into a format more suitable for use in neural networks by applying **one-hot encoding**.
   - One-hot encoding represents each label as a **binary vector** of length 10, where only the position corresponding to the label is set to `1`, and all others are `0`.
     - For example, label `3` is represented as `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
   - This allows the model's output layer to compute **probabilities for each class**, and during prediction, the class with the highest probability is chosen as the result.

4. **Dataset Splitting**
   - The entire MNIST dataset is divided into **training data** and **test data**.
   - **Training data** is used to train the model, while **test data** is used to evaluate the performance of the trained model.
   - Proper dataset splitting is crucial for checking the **generalization performance** of the model, preventing overfitting to the training data.

#### **Model Overview**
Now, let's explore the different methods applied to the MNIST dataset. These codes use techniques such as **tanh**, **SGD**, **ReLU+Momentum**, **Adam**, and **AdamW**, allowing us to understand each model's structure and the results produced.

1. **Tanh Activation Function**
   - **Tanh** is a hyperbolic tangent function that transforms the input values to the range **-1 to 1**. It centers the data around 0, which can be beneficial for training by ensuring that the output has a mean close to zero, thus improving convergence speed.
   - When applied to the **MNIST dataset**, tanh can be effective in hidden layers but may suffer from the **vanishing gradient problem**, especially in deep networks, which can make training challenging.

2. **SGD (Stochastic Gradient Descent)**
   - **SGD** is a type of gradient descent that updates weights based on each training example rather than the entire dataset, which makes it suitable for large datasets as it requires less memory.
   - For the **MNIST dataset**, using SGD allows for gradual optimization of the model, particularly when the dataset is relatively small, enabling fast convergence.

3. **ReLU + Momentum**
   - **ReLU (Rectified Linear Unit)** passes through positive input values as-is and sets all negative input values to zero. This helps to avoid the **vanishing gradient problem**, which often occurs in deep networks using other activation functions like sigmoid or tanh.
   - **Momentum** adds a fraction of the previous weight update to the current update to speed up convergence and reduce oscillations, particularly in areas where the loss function has steep slopes.
   - By combining **ReLU** and **Momentum** for the **MNIST dataset**, the model can achieve faster convergence and more stable training, even with deeper networks.

4. **Adam Optimizer**
   - **Adam** stands for Adaptive Moment Estimation and combines the advantages of both **momentum** and **RMSProp**. It uses **adaptive learning rates** for each parameter, resulting in efficient and faster convergence.
   - When applied to the **MNIST dataset**, Adam allows for **efficient learning** and generally provides good results without the need for much hyperparameter tuning.

5. **AdamW Optimizer**
   - **AdamW** is a variant of the Adam optimizer, which separates the **weight decay (L2 regularization)** from the gradient-based parameter updates. This results in better control over the **weight decay** mechanism and can help improve the model's generalization performance.
   - Using **AdamW** on the **MNIST dataset** allows for more stable training, with the weight decay ensuring that the weights do not become excessively large, which helps in achieving better model performance.

Each model utilizes a **distinct learning strategy**, allowing us to observe how different optimization and activation methods affect the training process and the model's performance on the MNIST dataset. By using these varied approaches, we can gain insight into the impact of each algorithm on the learning speed and the overall accuracy of the model.

## Installation Instructions

### 1. Git Clone

To clone this project to your local machine, use the following command to clone the GitHub repository:

```bash
git clone https://github.com/AEOMG/MNIST-Model-Benchmarks
cd MNIST
```

### 2. Set up Python Environment and Install Packages

This project uses Python 3.12 and several libraries. Please install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

**Note**: The paths are currently set to relative paths. If the program does not work, please change them to absolute paths.
