import numpy as np
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm
import os

# 데이터 로드 함수
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic_num, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols) / 255
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic_num, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):  # 인스턴스 생성 시 호출
        self.W1, self.b1, self.W2, self.b2 = self.initialize_network(input_size, hidden_size, output_size)
        self.SdW1, self.Sdb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)  # RMSProp을 위한 변수
        self.SdW2, self.Sdb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    # He 초기화 함수
    def initialize_network(self, input_size, hidden_size, output_size):
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        b2 = np.zeros((1, output_size))
        return W1, b1, W2, b2

    # Tanh 함수
    def tanh(self, x):
        return np.tanh(x)

    # Tanh 미분 함수
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # 순전파
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.tanh(Z1)  # Tanh 활성화 함수 사용
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Z2  # 회귀이므로 출력층에 활성화 함수를 사용하지 않음
        return A1, A2

    # 손실 함수: 평균 제곱 오차 (MSE)
    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        cost = np.sum((A2 - Y) ** 2) / (2 * m)
        return cost

    # 역전파
    def backward_propagation(self, X, Y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.tanh_derivative(A1)  # Tanh 미분 사용
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    # RMSProp 알고리즘을 사용한 파라미터 업데이트
    def update_parameters_rmsprop(self, dW1, db1, dW2, db2, learning_rate, beta=0.9, epsilon=1e-8):
        # RMSProp 가중치 업데이트
        self.SdW1 = beta * self.SdW1 + (1 - beta) * np.square(dW1)
        self.Sdb1 = beta * self.Sdb1 + (1 - beta) * np.square(db1)
        self.SdW2 = beta * self.SdW2 + (1 - beta) * np.square(dW2)
        self.Sdb2 = beta * self.Sdb2 + (1 - beta) * np.square(db2)

        self.W1 -= learning_rate * dW1 / (np.sqrt(self.SdW1) + epsilon)
        self.b1 -= learning_rate * db1 / (np.sqrt(self.Sdb1) + epsilon)
        self.W2 -= learning_rate * dW2 / (np.sqrt(self.SdW2) + epsilon)
        self.b2 -= learning_rate * db2 / (np.sqrt(self.Sdb2) + epsilon)

    # 모델 학습
    def train(self, X, Y, num_iterations, learning_rate, batch_size):
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
                self.update_parameters_rmsprop(dW1, db1, dW2, db2, learning_rate)  # RMSProp으로 업데이트

            if i % 10 == 0:
                tqdm.write(f"Epoch {i}: Cost {cost}")

    # 모델 평가 (R² 계산 추가)
    def evaluate(self, X, Y):
        _, A2 = self.forward_propagation(X)
        mse = np.mean((A2 - Y) ** 2)

        # R² 계산
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y - A2) ** 2)
        r2_score = 1 - (ss_residual / ss_total)

        return mse, r2_score

    # 예측
    def predict(self, X):
        _, A2 = self.forward_propagation(X)
        return A2  # 회귀에서는 연속적인 값을 반환

# 예측 결과 시각화 함수
def display_predictions(X, Y, predictions, num_samples=3):
    # 무작위로 샘플 3개를 선택
    indices = np.random.choice(X.shape[0], num_samples, replace=False)

    for i in indices:
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Actual: {Y[i]}, Predicted: {predictions[i][0]:.2f}')
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

    # 신경망 모델 초기화
    input_size = 784
    hidden_size = 128
    output_size = 1  # 회귀이므로 출력은 하나의 연속 값

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # 모델 학습
    num_iterations = 100  # 반복 횟수 설정
    learning_rate = 0.01
    batch_size = 64

    nn.train(X_train, y_train.reshape(-1, 1), num_iterations, learning_rate, batch_size)

    # 모델 학습 후 평가
    mse, r2 = nn.evaluate(X_test, y_test.reshape(-1, 1))
    print(f'Test MSE: {mse:.2f}')
    print(f'Test R²: {r2:.2f}')

    # 예측 결과 확인
    predictions = nn.predict(X_test)
    display_predictions(X_test, y_test, predictions)