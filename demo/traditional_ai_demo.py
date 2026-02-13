"""
Traditional AI Demo - 시간축 기반 학습
비교 시연용: epoch를 반복하며 천천히 수렴하는 과정을 보여줌
"""
import numpy as np
import time
import sys

# ── 데이터: 비선형 패턴 분류 (XOR 확장) ──
np.random.seed(42)

def generate_data(n=200):
    """4-클러스터 비선형 패턴"""
    points = []
    labels = []
    for _ in range(n):
        x, y = np.random.uniform(-2, 2, 2)
        # 비선형 경계: XOR-like 패턴
        label = 1.0 if (x * y > 0) else 0.0
        # 노이즈 추가
        x += np.random.normal(0, 0.15)
        y += np.random.normal(0, 0.15)
        points.append([x, y])
        labels.append(label)
    return np.array(points), np.array(labels).reshape(-1, 1)

# ── 신경망 (순수 numpy) ──
class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self.relu(z)
            self.activations.append(current)
        # 출력층: sigmoid
        z = current @ self.weights[-1] + self.biases[-1]
        current = self.sigmoid(z)
        self.activations.append(current)
        return current

    def backward(self, X, y, lr=0.05):
        m = X.shape[0]
        output = self.activations[-1]
        delta = (output - y) * self.sigmoid_deriv(output)

        deltas = [delta]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1] @ self.weights[i+1].T * self.relu_deriv(self.activations[i+1])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= lr * (self.activations[i].T @ deltas[i]) / m
            self.biases[i] -= lr * np.sum(deltas[i], axis=0, keepdims=True) / m

    def loss(self, y_pred, y_true):
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


# ── 학습 실행 ──
def train_demo():
    print("=" * 55)
    print("  Traditional AI - 시간축 기반 학습")
    print("  Neural Network (2 → 16 → 8 → 1)")
    print("=" * 55)
    print()

    X, y = generate_data(200)
    nn = NeuralNetwork([2, 16, 8, 1])

    epochs = 1000
    print_every = 50

    print(f"  데이터: {len(X)}개 샘플")
    print(f"  목표 : 비선형 패턴(XOR) 분류")
    print(f"  방법 : {epochs} epoch 반복 학습")
    print()
    print("  학습 시작...")
    print("  " + "-" * 45)

    start = time.time()

    for epoch in range(1, epochs + 1):
        output = nn.forward(X)
        nn.backward(X, y, lr=0.05)

        if epoch % print_every == 0 or epoch == 1:
            l = nn.loss(output, y)
            acc = np.mean((output > 0.5).astype(float) == y) * 100
            elapsed = time.time() - start
            bar_len = int(acc / 100 * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)

            print(f"  epoch {epoch:>5d}/{epochs}  "
                  f"loss: {l:.4f}  "
                  f"acc: [{bar}] {acc:.1f}%  "
                  f"({elapsed:.2f}s)")
            sys.stdout.flush()

        # 학습 과정을 눈에 보이게 하기 위한 딜레이
        time.sleep(0.002)

    total = time.time() - start

    # ── 최종 결과 ──
    final_output = nn.forward(X)
    final_acc = np.mean((final_output > 0.5).astype(float) == y) * 100
    final_loss = nn.loss(final_output, y)

    print("  " + "-" * 45)
    print()
    print("  [결과]")
    print(f"  최종 정확도 : {final_acc:.1f}%")
    print(f"  최종 손실   : {final_loss:.4f}")
    print(f"  소요 시간   : {total:.2f}초")
    print(f"  반복 횟수   : {epochs} epochs")
    print()

    # ── 테스트 ──
    test_points = np.array([
        [ 1.0,  1.0],   # → 1 (양*양)
        [-1.0, -1.0],   # → 1 (음*음)
        [ 1.0, -1.0],   # → 0 (양*음)
        [-1.0,  1.0],   # → 0 (음*양)
    ])
    preds = nn.forward(test_points)

    print("  [테스트 샘플]")
    for i, (p, pred) in enumerate(zip(test_points, preds)):
        label = "O" if pred[0] > 0.5 else "X"
        print(f"  ({p[0]:+.1f}, {p[1]:+.1f}) → {label}  (확률: {pred[0]:.3f})")

    print()
    print(f"  ※ {epochs}번 반복해서 겨우 수렴")
    print("=" * 55)


if __name__ == "__main__":
    train_demo()
