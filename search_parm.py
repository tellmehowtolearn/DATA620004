import os
import struct
import numpy as np
import pickle


class neuralNetwork:

    def __init__(self, D_in, H, D_out):
        # 定义变量
        self.inodes = D_in
        self.hnodes = H
        self.onodes = D_out
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

        # 创建权重矩阵
        self.w1 = np.random.randn(self.inodes, self.hnodes)
        self.w2 = np.random.randn(self.hnodes, self.onodes)

        # 如果有构建好的模型参数，直接加载
        if os.path.exists("D:/Code/pythonProject1/data/json_w1.txt"):
            self.w1 = np.loadtxt("D:/Code/pythonProject1/data/json_w1.txt")
        else:
            pass

        if os.path.exists("D:/Code/pythonProject1/data/json_w2.txt"):
            self.w2 = np.loadtxt("D:/Code/pythonProject1/data/json_w2.txt")
        else:
            pass

    def load_data(self):
        x_test, y_test = self.load_mnist("minst_data", 't10k')
        x_test = x_test / 255 * 0.99 + 0.01
        y_test = self.convert_to_one_hot(y_test, 10)
        x_train, y_train = self.load_mnist("minst_data", 'train')
        y_train = self.convert_to_one_hot(y_train, 10)
        x_train = x_train / 255 * 0.99 + 0.01

        return x_train, y_train, x_test, y_test


    @staticmethod
    def softmax(x):
        x_max = x - x.max(axis=1).reshape((-1, 1))
        exp_x = np.exp(x_max)
        output = exp_x / exp_x.sum(axis=1).reshape((-1, 1))
        return output

    def forward_pass(self, input_list):
        # forward pass
        inputs = np.array(input_list, ndmin=2)
        h = 1 / (1 + np.exp(-inputs.dot(self.w1)))
        y_pred_pre = h.dot(self.w2)
        y_pred = self.softmax(y_pred_pre)
        return y_pred, h

    def loss_function(self, nu, y_pred, y):
        loss = self.cross_entropy_error(y_pred, y) + 0.5 * nu * np.sum(
            self.w1 * self.w1) + 0.5 * nu * np.sum(self.w2 * self.w2)
        return loss

    def gradient_compute(self, sample_size, y_pred, h, nu):
        # Compute the gradient, use SGD method

        sample = np.random.randint(0, self.x_train.shape[0], sample_size)
        y_pred_temp = y_pred[sample, :]
        y_temp = self.y_train[sample, :]
        h_temp = h[sample, :]
        x_temp = self.x_train[sample, :]
        dy_pred_pre = y_pred_temp - y_temp
        dw2 = h_temp.T.dot(dy_pred_pre) + nu * np.ones(
            self.w2.shape)
        dh = dy_pred_pre.dot(self.w2.T)
        dw1 = x_temp.T.dot(dh * (h_temp * (1 - h_temp))) + nu * np.ones(self.w1.shape)

        return dw1, dw2

    @staticmethod
    def convert_to_one_hot(y, C):
        # label to one_hot
        return np.eye(C)[y.reshape(-1)]

    @staticmethod
    def convert_to_label(y):
        # one_hot to label
        return np.argmax(y, axis=-1)

    @staticmethod
    def cross_entropy_error(y, t):
        delta = 1e-7  # 防止计算错误，加上一个微小值
        return -np.sum(t * np.log(y + delta))

    @staticmethod
    def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as label_path:
            magic, n = struct.unpack('>II', label_path.read(8))
            labels = np.fromfile(label_path, dtype=np.uint8)

        with open(images_path, 'rb') as img_path:
            magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
            images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels

    def train(self, train_number, nu, learning_rate,test_flag,print_flag):
        decrease_gamma = 0.99
        for it in range(train_number):
            # forward pass
            y_pred, h = self.forward_pass(self.x_train)

            # Compute Loss
            loss = self.loss_function(nu, y_pred, self.y_train)
            self.train_loss.append(loss)
            # print_flag = 1，print loss
            if print_flag == 1:
                print("第", it+1, "次训练的损失值:", loss)
            else:
                pass

            # Backward pass
            sample_size = 128
            dw1, dw2 = self.gradient_compute(sample_size, y_pred, h, nu)

            # Update weights of w1 and w2
            self.w1 -= learning_rate * (dw1 / sample_size)
            self.w2 -= learning_rate * (dw2 / sample_size)

            # learning rate decrease strategy
            if it % 100 == 0:
                learning_rate = decrease_gamma * learning_rate

            # test_flag = 1 means that the test set is tested once per iteration
            if test_flag == 1:
                self.test(1, nu)
            else:
                pass

    def test(self, test_flag, nu):
        test_h = 1 / (1 + np.exp(-self.x_test.dot(self.w1)))
        test_y_pred_pre = test_h.dot(self.w2)
        test_y_pred = self.softmax(test_y_pred_pre)  # 把（64，100）的隐藏层h转为（64，10）的输出矩阵Y
        accuracy = np.sum(self.convert_to_label(test_y_pred) == self.convert_to_label(self.y_test)) / self.y_test.shape[0]

        if test_flag == 1:
            self.test_accuracy.append(accuracy)
            # Compute Loss
            loss = self.loss_function(nu, test_y_pred, self.y_test)
            self.test_loss.append(loss)
        else:
            pass

        return accuracy

    def store_values(self):  # 将权重写入到指定的文件中来
        filename_w1 = "D:/Code/pythonProject1/data/json_w1.txt"

        filename_w2 = "D:/Code/pythonProject1/data/json_w2.txt"
        with open(filename_w1, 'w') as fw1:
            np.savetxt(fw1, self.w1)

        with open(filename_w2, 'w') as fw2:
            np.savetxt(fw2, self.w2)


if __name__ == '__main__':
    # 构建神经网络结构
    input_nodes = 784
    output_nodes = 10

    # 参数查找
    hidden_nodes_list = [100, 150, 200]  # 隐藏层大小
    nu_list = [1e-5, 1e-4, 1e-3]  # 正则化强度
    learning_rate_list = [1, 5, 0.5]  # 学习率
    test_flag = 0  # 等于1时表示每次迭代训练都进行测试集的测试并保存结果
    train_number = 200  # 训练次数
    print_flag = 1  # 等于1时打印训练的loss
    # 最小的loss，最终选择同等迭代下最后一次loss最小的
    loss_min = 1e10

    hidden_nodes_final = 100
    nu_final = 1e-5
    learning_rate_final = 1

    for hidden_nodes in hidden_nodes_list:
        for nu in nu_list:
            for learning_rate in learning_rate_list:
                model = neuralNetwork(input_nodes, hidden_nodes, output_nodes)
                model.train(train_number, nu, learning_rate, test_flag, print_flag)
                if model.train_loss[-1] < loss_min:
                    loss_min = model.train_loss[-1]
                    print(model.train_loss[-1])
                    hidden_nodes_final = hidden_nodes
                    nu_final = nu
                    learning_rate_final = learning_rate
                else:
                    pass


    print("最终学习率为", learning_rate_final, "；最终隐藏层大小为", hidden_nodes_final, "；最终正则化强度为", nu_final)

    # 保存各项参数
    f = open('D:/Code/pythonProject1/data/hidden_nodes.txt', 'wb')
    pickle.dump(hidden_nodes_final, f)
    f.close()
    f = open('D:/Code/pythonProject1/data/nu.txt', 'wb')
    pickle.dump(nu_final, f)
    f.close()
    f = open('D:/Code/pythonProject1/data/learning_rate.txt', 'wb')
    pickle.dump(learning_rate_final, f)
    f.close()
