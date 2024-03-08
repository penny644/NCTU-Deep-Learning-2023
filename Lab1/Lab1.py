import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    
def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def show_loss(loss, title):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

class model:
    def __init__(self, hidden_size1, hidden_size2, lr = 0.1):
        self.W1 = np.random.rand(2, hidden_size1)
        self.W2 = np.random.rand(hidden_size1, hidden_size2)
        self.W3 = np.random.rand(hidden_size2, 1)
        self.z1 = np.zeros((1, hidden_size1))
        self.z2 = np.zeros((1, hidden_size2))
        self.z3 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, x):
        a1 = x @ self.W1
        self.z1 = sigmoid(a1)
        a2 = self.z1 @ self.W2
        self.z2 = sigmoid(a2)
        a3 = self.z2 @ self.W3
        y = sigmoid(a3)
        return y
    
    def backward(self, x, y, y_hat):
        L = (y - y_hat) ** 2
        der_L_y = 2 * (y - y_hat)
        der_y_a3 = derivative_sigmoid(y)
        der_a3_W3 = self.z2.T
        der_L_a3 = der_L_y * der_y_a3
        der_L_W3 = der_a3_W3 @ der_L_a3

        der_L_z2 = der_L_a3 @ self.W3.T
        der_z2_a2 = derivative_sigmoid(self.z2)
        der_L_a2 =  der_L_z2 * der_z2_a2
        der_a2_W2 = self.z1.T
        der_L_W2 = der_a2_W2 @ der_L_a2

        der_L_z1 = der_L_a2 @ self.W2.T
        der_z1_a1 = derivative_sigmoid(self.z1)
        der_L_a1 = der_L_z1 * der_z1_a1
        der_L_W1 = x.T @ der_L_a1

        self.W1 -= self.lr * der_L_W1
        self.W2 -= self.lr * der_L_W2
        self.W3 -= self.lr * der_L_W3
        return L

class model_no_activation:
    def __init__(self, hidden_size1, hidden_size2, lr = 0.1):
        self.W1 = np.random.rand(2, hidden_size1)
        self.W2 = np.random.rand(hidden_size1, hidden_size2)
        self.W3 = np.random.rand(hidden_size2, 1)
        self.z1 = np.zeros((1, hidden_size1))
        self.z2 = np.zeros((1, hidden_size2))
        self.z3 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, x):
        self.z1 = x @ self.W1
        self.z2 = self.z1 @ self.W2
        y = self.z2 @ self.W3
        return y
    
    def backward(self, x, y, y_hat):
        L = (y - y_hat) ** 2
        der_L_y = 2 * (y - y_hat)
        der_y_W3 = self.z2.T
        der_L_W3 = der_y_W3 @ der_L_y

        der_L_z2 = der_L_y @ self.W3.T
        der_z2_W2 = self.z1.T
        der_L_W2 = der_z2_W2 @ der_L_z2

        der_L_z1 = der_L_z2 @ self.W2.T
        der_z1_W1 = x.T
        der_L_W1 = der_z1_W1 @ der_L_z1

        self.W1 -= self.lr * der_L_W1
        self.W2 -= self.lr * der_L_W2
        self.W3 -= self.lr * der_L_W3
        return L
if __name__ == '__main__':
    np.random.seed(1)
    x, y = generate_linear(n=100)
    max_epoch = 5000
    print_interval = 1000
    classifier = model(10, 10, 0.01)
    #classifier = model_no_activation(10, 10, 0.01)
    all_loss = []
    for epoch in range(max_epoch):
        losses = 0
        correct = 0
        count = 0
        acc = 0.0
        for i in range(len(x)):
            predict_y = classifier.forward(x[i].reshape(1, 2))
            loss = classifier.backward(x[i].reshape(1, 2), predict_y, y[i])
            losses += loss.reshape(-1)[0]
            if(predict_y >= 0.5):
                ans = 1
            else:
                ans = 0
            if(ans == y[i][0]):
                correct += 1
            count += 1
        acc = correct / count
        losses /= len(x)
        all_loss.append(losses)
        if(acc == 1):
            print(f'epoch : {epoch} loss : {losses} accuracy : {acc:.5f}')
            break
        if((epoch  % print_interval) == 0) | (epoch == max_epoch-1):
            print(f'epoch : {epoch} loss : {losses} accuracy : {acc:.5f}')
    show_loss(all_loss, 'Linear')

    correct = 0
    count = 0
    acc = 0.0
    loss = 0
    ans_plot = np.zeros((len(x)))
    for i in range(len(x)):
        count += 1
        predict_y = classifier.forward(x[i].reshape(1, 2)).reshape(-1)[0]
        if(predict_y >= 0.5):
            ans = 1
        else:
            ans = 0
        ans_plot[i] = ans
        if(ans == y[i][0]):
            correct += 1
        loss += (y[i][0] - predict_y) ** 2
        print(f'Iter{i} | Ground truth: {y[i][0]:.1f} | prediction: {predict_y:.5f} |')
    loss /= len(x)
    acc = correct / count * 100.0
    print(f'loss={loss:.5f} accuracy={acc:.2f}%')
    show_result(x, y, ans_plot)

    x, y = generate_XOR_easy()
    max_epoch = 50000
    print_interval = 5000
    classifier = model(10, 10, 0.01)
    #classifier = model_no_activation(10, 10, 0.01)
    all_loss = []
    for epoch in range(max_epoch):
        losses = 0
        correct = 0
        count = 0
        acc = 0.0
        for i in range(len(x)):
            predict_y = classifier.forward(x[i].reshape(1, 2))
            loss = classifier.backward(x[i].reshape(1, 2), predict_y, y[i])
            losses += loss.reshape(-1)[0]
            if(predict_y >= 0.5):
                ans = 1
            else:
                ans = 0
            if(ans == y[i][0]):
                correct += 1
            count += 1
            # if((epoch  % print_interval) == 0) | (epoch == max_epoch-1):
            #     print(f'{predict_y[0][0]:.5f}')
        acc = correct / count
        losses /= len(x)
        all_loss.append(losses)
        if(acc == 1):
            print(f'epoch : {epoch} loss : {losses} accuracy : {acc:.5f}')
            break
        if((epoch  % print_interval) == 0) | (epoch == max_epoch-1):
            print(f'epoch : {epoch} loss : {losses} accuracy : {acc:.5f}')
    show_loss(all_loss, 'Xor')

    correct = 0
    count = 0
    acc = 0.0
    loss = 0
    ans_plot = np.zeros((len(x)))
    for i in range(len(x)):
        count += 1
        predict_y = classifier.forward(x[i].reshape(1, 2)).reshape(-1)[0]
        if(predict_y >= 0.5):
            ans = 1
        else:
            ans = 0
        ans_plot[i] = ans
        if(ans == y[i][0]):
            correct += 1
        loss += (y[i][0] - predict_y) ** 2
        print(f'Iter{i} | Ground truth: {y[i][0]:.1f} | prediction: {predict_y:.5f} |')
    loss /= len(x)
    acc = correct / count * 100.0
    print(f'loss={loss:.5f} accuracy={acc:.2f}%')
    show_result(x, y, ans_plot)




