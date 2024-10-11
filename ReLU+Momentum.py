import numpy as np
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm
import os

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.9):
        self.W1, self.b1, self.W2, self.b2 = self.initialize_network(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    def initialize_network(self, input_size, hidden_size, output_size):
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * 0.01
        b2 = np.zeros((1, output_size))
        return W1, b1, W2, b2

    # ReLU 활성화 함수
    def relu(self, x):
        return np.maximum(0, x)

    # ReLU 미분
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    # 소프트맥스 함수
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 안정성을 위해 최대값을 뺌
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)  # 소프트맥스 사용
        return A1, A2

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        epsilon = 1e-9  # 아주 작은 값 추가
        A2_clipped = np.clip(A2, epsilon, 1 - epsilon)  # 안정성 위해 클리핑 클리핑: 최솟값과 최댓값을 제한하는 기능. 로그나 지수함수에서 값이 너무 커지는걸 방지함. 
        logprobs = np.multiply(np.log(A2_clipped), Y) # 클리핑 형식: np.clip(a, a_min, a_max)
        cost = -np.sum(logprobs) / m
        return cost

    def backward_propagation(self, X, Y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    # Momentum을 사용한 파라미터 업데이트
    def update_parameters(self, dW1, db1, dW2, db2):
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
        self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def train(self, X, Y, num_iterations, batch_size):
        for i in tqdm(range(num_iterations), desc="Training Progress"):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            for j in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]

                A1, A2 = self.forward_propagation(X_batch)
                cost = self.compute_loss(Y_batch, A2)
                dW1, db1, dW2, db2 = self.backward_propagation(X_batch, Y_batch, A1, A2)
                self.update_parameters(dW1, db1, dW2, db2)

            if i % 10 == 0:
                tqdm.write(f"Epoch {i}: {cost:.6f}")

    def evaluate(self, X, Y):
        _, A2 = self.forward_propagation(X)
        predictions = np.argmax(A2, axis=1)
        accuracy = np.mean(predictions == Y)
        return accuracy

    def predict(self, X):
        _, A2 = self.forward_propagation(X)
        return np.argmax(A2, axis=1)

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic_num, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols) / 255.0
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic_num, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def display_predictions(X, Y, predictions, num_samples=3):
    # 무작위로 샘플 3개를 선택
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    
    for i in indices:
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Actual: {Y[i]}, Predicted: {predictions[i]}')
        plt.show()

if __name__ == "__main__":
    base_path = os.path.join(os.getcwd(), 'dataset')
    train_images_path = os.path.join(base_path, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(base_path, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(base_path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(base_path, 't10k-labels-idx1-ubyte.gz')

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    y_train_one_hot = np.eye(10)[y_train]
    y_test_one_hot = np.eye(10)[y_test]

    input_size = 784
    hidden_size = 128
    output_size = 10
    num_iterations = 100
    learning_rate = 0.01
    batch_size = 64

    nn = SimpleNN(input_size, hidden_size, output_size, learning_rate)

    nn.train(X_train, y_train_one_hot, num_iterations, batch_size)

    accuracy = nn.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    predictions = nn.predict(X_test)
    display_predictions(X_test, y_test, predictions)