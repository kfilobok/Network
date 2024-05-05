import numpy
import scipy.special


# import matplotlib.pyplot
# %matplotlib inline

class Network:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.3):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        # Матрицы весовых коэффициентов связей wih и who.
        # Весовые коэффициенты связей между узлом і и узлом ј значеныккі_ј:
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # использование сигмоиды в качестве функции активации
        self.activation_func = lambda x: scipy.special.expit(x)

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        targets = numpy.array(targets_list, ndmin=2).T
        inputs = numpy.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_func(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_func(final_inputs)
        # ошибки выходного слоя (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_func(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_func(final_inputs)
        return final_outputs


# количество входных, скротых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения
learning_rate = 0.1
# создать экземпляр нейронной сети
n = Network(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open('train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(1):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        inputs_plusx_img = scipy.ndimage.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                 reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)

test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

# проходимся по test
for record in test_data_list:
    all_values = record.split(',')
    # правильный ответ пе5рвое значение
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
