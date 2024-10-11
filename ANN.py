import numpy as np
import matplotlib.pyplot as plt
import gzip
import os

# 데이터 로드 함수
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 헤더 정보를 읽습니다. 헤더는 총 16바이트입니다.
        magic_num, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # 이미지 데이터를 읽습니다. 각 이미지는 28x28 픽셀입니다.
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols) / 255.0
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 레이블 파일의 헤더는 총 8바이트입니다.
        magic_num, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # 레이블 데이터를 읽습니다.
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# 모델 초기화 함수
def initialize_network(input_size, hidden_size, output_size): # 가중치를 정규분포로 초기화. 정규분포 = 데이터 집합이 평균을 중심으로 좌우 대칭적인 종 모양의 곡선을 그림. 이 분포는 평균과 표준편차 두 매개변수로 모양이 결정됨.
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    #가중치를 무작위로 초기화,매우 작은 값으로 설정하여 신경망의 각 뉴런이 학습 초기에 비슷한 수준의 활성화를 가짐.이는 학습 과정에서 각 뉴런이 비슷한 속도로 학습을 시작할 수 있도록 돕는다. + 기울기 소실 문제 방지. + 효율적인 학습.
    return W1, b1, W2, b2
    
# 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 순전파
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, A2

# 손실함수의 용도와 필요성은 이해했으나 아직 해당 손실함수 구조는 이해하지 못함.
def compute_loss(Y, A2):
    m = Y.shape[0]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    return cost

# 역전파
# m = 데이터 포인트 수. 평균을 사용함으로써 각 업데이트가 전체 데이터셋을 대표하게 만들고, 학습 과정에서의 변동성을 줄인다.
def backward_propagation(X, Y, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y #오차 dZ2는 예측값 A2에서 실제값 Y를 뺀 값.
    dW2 = np.dot(A1.T , dZ2) / m  #dW2 은닉층의 활성화 출력A1 과 출력층의 오차dZ2 의 행렬곱으로 계산됨.
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m #db2 는 출력층의 오차 dZ2에 대해 각 데이터 포인트에 대한 합을 구함.
    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1) # dZ1은 오차dZ2 와 출력층 가중치 W2의 전치 행렬곱을 한 결과를 은닉층의 활성화 함수의 미분 결과와 요소별 곱.
    dW1 = np.dot(X.T, dZ1) / m # dW1은 입력값 X와 오차 dZ1과 행렬곱. 가중치의 기울기를 구함.
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m # 오차 dZ의 합은 해당 편향에 연결된 모든 오차들을 합하는 것을 의미.
    return dW1, db1, dW2, db2

# 가중치 업데이트
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

#모델
def model(X, Y, hidden_size, num_iterations, learning_rate, batch_size):
    np.random.seed(2)
    input_size = X.shape[1]
    output_size = Y.shape[1]

    W1, b1, W2, b2 = initialize_network(input_size, hidden_size, output_size)

    for i in range(num_iterations):
        # 전체 데이터를 섞습니다.
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]

            A1, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
            cost = compute_loss(Y_batch, A2)
            dW1, db1, dW2, db2 = backward_propagation(X_batch, Y_batch, A1, A2, W2)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 10 == 0:
            print("Epoch %i: %f" % (i, cost))

    return W1, b1, W2, b2

def evaluate_model(X, Y, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)  # 각 샘플에 대해 가장 높은 확률을 가진 클래스 선택
    accuracy = np.mean(predictions == Y)  # 예측값과 실제 레이블을 직접 비교하여 정확도 계산
    return accuracy

def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)  # 예측값을 정수 레이블로 반환

def display_predictions(X, Y, predictions):
    for i in range(5): #에를 들어 5개를 시각화
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

    # 원-핫 인코딩
    y_train_one_hot = np.eye(10)[y_train]
    y_test_one_hot = np.eye(10)[y_test]

    # 모델 초기화
    _sizinpute = 784
    hidden_size = 128
    output_size = 10
    num_iterations = 300  # 반복 횟수 설정
    learning_rate = 0.01
    batch_size = 64

    # 모델 학습
    W1, b1, W2, b2 = model(X_train, y_train_one_hot, hidden_size, num_iterations, learning_rate, batch_size)

    # 모델 학습 후 평가
    accuracy = evaluate_model(X_test, y_test, W1, b1, W2, b2)
    print(f'Test Accuracy: {accuracy:.2f}')

    # 예측 결과 확인
    predictions = predict(X_test, W1, b1, W2, b2)
    display_predictions(X_test, y_test, predictions)