import numpy as np
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm
import os

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1, self.b1, self.W2, self.b2 = self.initialize_network(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate  # SGD 학습률

    def initialize_network(self, input_size, hidden_size, output_size):
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)  # Xavier 초기화
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        b2 = np.zeros((1, output_size))
        return W1, b1, W2, b2

    # 시그모이드 함수
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 시그모이드 미분 함수
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # 소프트맥스 함수
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # 순전파
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)  # 시그모이드 활성화 함수 사용
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)  # 소프트맥스 사용
        return A1, A2

    # 손실 함수 (교차 엔트로피)
    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        epsilon = 1e-7
        A2 = np.clip(A2, epsilon, 1 - epsilon)  # 안정성 위해 클리핑
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = -np.sum(logprobs) / m
        return cost

    # 역전파
    def backward_propagation(self, X, Y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(A1)  # 시그모이드 미분 사용
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    # SGD 업데이트 함수
    def update_parameters(self, dW1, db1, dW2, db2):
        # SGD 방식으로 가중치 업데이트
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # 학습
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
                tqdm.write(f"Epoch {i}: Cost {cost}")

    # 평가
    def evaluate(self, X, Y):
        _, A2 = self.forward_propagation(X)
        predictions = np.argmax(A2, axis=1)
        accuracy = np.mean(predictions == Y)
        return accuracy

    # 예측
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

    # SGD와 시그모이드를 사용하는 신경망 생성
    nn = SimpleNN(input_size, hidden_size, output_size, learning_rate)

    nn.train(X_train, y_train_one_hot, num_iterations, batch_size)

    accuracy = nn.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    predictions = nn.predict(X_test)
    display_predictions(X_test, y_test, predictions)