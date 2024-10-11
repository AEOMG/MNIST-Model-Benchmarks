import numpy as np
import os
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, weight_decay=0.01):
        self.W1, self.b1, self.W2, self.b2 = self.initialize_network(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # AdamW 매개변수 초기화
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def update_parameters(self, dW1, db1, dW2, db2):
        # 매개변수 업데이트를 위해 t 증가
        self.t += 1
        # 바이어스 보정된 학습률 계산
        lr_t  = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        # m과 v의 업데이트 (AdamW 적용)
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)

        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)

        # AdamW 방식으로 매개변수 업데이트 (Weight Decay 적용)
        self.W1 -= lr_t * (self.mW1 / (np.sqrt(self.vW1) + self.epsilon) + self.weight_decay * self.W1)
        self.b1 -= lr_t * (self.mb1 / (np.sqrt(self.vb1) + self.epsilon) + self.weight_decay * self.b1)
        self.W2 -= lr_t * (self.mW2 / (np.sqrt(self.vW2) + self.epsilon) + self.weight_decay * self.W2)
        self.b2 -= lr_t * (self.mb2 / (np.sqrt(self.vb2) + self.epsilon) + self.weight_decay * self.b2)

    def initialize_network(self, input_size, hidden_size, output_size):
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        b2 = np.zeros((1, output_size))
        return W1, b1, W2, b2

    # GELU 활성화 함수
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.gelu(Z1)  # GELU 활성화 함수 사용
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)  # softmax 사용
        return A1, A2

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        epsilon = 1e-7  # 아주 작은 값 추가
        A2 = np.clip(A2, epsilon, 1 - epsilon)
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = -np.sum(logprobs) / m
        return cost

    def backward_propagation(self, X, Y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.tanh(A1)**2)  # GELU의 미분 (Tanh 미분 사용)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

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
                tqdm.write(f"Epoch {i}: {cost:.12f}")

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
    num_iterations = 40
    learning_rate = 0.01
    batch_size = 64

    nn = SimpleNN(input_size, hidden_size, output_size, learning_rate)

    nn.train(X_train, y_train_one_hot, num_iterations, batch_size)

    accuracy = nn.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    predictions = nn.predict(X_test)
    display_predictions(X_test, y_test, predictions)