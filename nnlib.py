import numpy
# библиотека scipy.special содержит сигмоиду exlit() 
import scipy.special
# для получения массива из PNG
import scipy.misc
# библиотека для графического отображения массивов
import matplotlib.pyplot
# гарантировать размещение графики в данном блокноте, а не в отдельном окне
# get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
# библиотеки для работы с папками и файлами
import os
# для создания временных файлов
import tempfile
# для удаления папки с файлами
import shutil
# определение класса нейронной сети
class neuralNetwork:
    
    # инициализировать нейронную сеть
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate,retrainnet):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Матрицы весовых коэффициентов связей wih (между входным и срытым 
        # слоями) и who (между скрытым и выходным слоями).
        # Начало со случайными весами
        if retrainnet == 0:
            self.wih = numpy.random.normal(0.0,pow(self.hnodes, -0.5), (self.hnodes,self.inodes))
            self.who = numpy.random.normal(0.0,pow(self.onodes, -0.5), (self.onodes,self.hnodes))
        else:
        # Дообучение - берем веса обученной сети
        # берем матрицы из соответствующих файлов
            self.who = numpy.loadtxt("NNLearn/"+str(retrainnet)+"/nn01_who.txt", delimiter='\t', dtype=numpy.float)
            self.wih = numpy.loadtxt("NNLearn/"+str(retrainnet)+"/nn01_wih.txt", delimiter='\t', dtype=numpy.float)
        # коэффициент обучения
        self.lr = learningrate
        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    # тренировка нейронной сети
    def train(self,inputs_list,targets_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih,inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # ошибки выходного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        # обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                        numpy.transpose(hidden_outputs))
        
        # обновить весовые коэффициенты для связей между
        # входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                        numpy.transpose(inputs))
                
        pass
    # опрос нейронной сети
    def query(self,input_list):
        # преобразовать список входных значений
        # в двумерный массив
        inputs = numpy.array(input_list,ndmin=2).T
        
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih,inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass

def CreateImgScreen(wih_input,file_img_name):
    b = (1+wih_input.reshape(78400))*255/2
    c = b.reshape(10,10,28,28)
    h = numpy.zeros((280,280))
    line_c = numpy.zeros((280,290))
    line_f = numpy.zeros((290,290))
    for i in range(10):
        for j in range(10):
            for k in range(28):
                for l in range(28):
                    h[28*i+k,28*j+l] = c[i,j,k,l]
    line_c = numpy.insert(h, [28,56,84,112,140,168,196,224,252], 255, axis = 1)
    line_f = numpy.insert(line_c, [28,56,84,112,140,168,196,224,252], 255, axis = 0)
    # делаем файл с изображением
    image = line_f
    cv2.imwrite(file_img_name + ".png", image)

def CreateImgScreenSort(n_in,file_img_name):
    find_max_ind1 = numpy.array([0,0,0,0,0,0,0,0,0,0])
    numpy.argmax(n_in.who, axis=1, out=find_max_ind1)
    ind_r = numpy.argsort(n_in.who)
    h = numpy.zeros((280,280))
    line_c = numpy.zeros((280,290))
    line_f = numpy.zeros((290,290))
    b = (1+n_in.wih.reshape(100,28,28))*255/2
    for i in range(10):
        for j in range(10):
            for k in range(28):
                for l in range(28):
                    h[28*i+k,28*j+l] = b[ind_r[i,(99-j)],k,l]
    line_c = numpy.insert(h, [28,56,84,112,140,168,196,224,252], 255, axis = 1)
    line_f = numpy.insert(line_c, [28,56,84,112,140,168,196,224,252], 255, axis = 0)
    # делаем файл с изображением
    image_nums = line_f
    #file_img_name2 = "NNLearn/"+P_values[0]+"/00imgHFinalNums"
    cv2.imwrite(file_img_name +".png", image_nums)
   
def createtrainfolder(nnnum_in):
    if os.path.isdir("NNLearn/"+nnnum_in): 
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "NNLearn/"+nnnum_in)
        shutil.rmtree(path)
    os.mkdir("NNLearn/"+nnnum_in)

def createtestfolder(nntest_in):
    if os.path.isdir("NNReports/"+nntest_in): 
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "NNReports/"+nntest_in)
        shutil.rmtree(path)
    os.mkdir("NNReports/"+nntest_in)
    os.mkdir("NNReports/"+nntest_in+"/ImgErr")

def trainfromCSV(traning_data_file_name_in,epochs_in,output_nodes_in,n_in,nnnum_in,img_interval_in):
    # загрузить в список тестовый набор данных CSV-файла набора MNIST
    traning_data_file = open(traning_data_file_name_in,'r')
    traning_data_list = traning_data_file.readlines()
    traning_data_file.close()
    for e in range(epochs_in):
        # перебрать все записи в тренировочном наборе данных
        numstr = 0
        for record in traning_data_list:
            # получить список значений, используя символы запятой (',')
            # в качестве разделителей
            all_values = record.split(',')
            # масштабировать и сместить входные значения
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # создать целевые выходные значения (все равны 0.01, за исключением
            # желаемого маркерного значения, равного 0.99)
            targets = numpy.zeros(output_nodes_in) + 0.01
            targets[int(all_values[0])] = 0.99
            n_in.train(inputs,targets)
            numstr += 1
            # Промежуточные записи (картинка с тренировкой сети)
            # делаем файл с изображением
            if (numstr % img_interval_in == 0):
                prefix_name_rec = "e"+str(e) + "n" + str(numstr)
                file_img_name = "NNLearn/"+ nnnum_in +"/imgH_" + prefix_name_rec
                CreateImgScreen(n_in.wih,file_img_name)

def trainfromPNG(traning_data_file_name_in,epochs_in,output_nodes_in,n_in,nnnum_in,img_interval_in):
    # создаем два пустых массива - для изодражений и для меток валидации, 
    # номер изображения пока берем из файла и пишем бощий массив пишем
    # потом по нему идем по эпохам несколько раз
    validation = {}
    img_ind_matr = {}
    # делаем цикл по цифрам - пишем массив
    for i in range(10):
        for filepath in os.listdir(traning_data_file_name_in+str(i)):
            image = cv2.imread(traning_data_file_name_in+str(i)+'/'+filepath,0)
            img_nump = numpy.asarray(image, dtype='uint8')
            num_img = filepath[3:-4]
            validation[int(num_img)] = i
            img_data = 255.0 - img_nump.reshape(784)
            img_ind_matr[int(num_img)] = img_data
    # теперь идем в цикле по эпохам и по словарю и тренируем сеть
    for e in range(epochs_in):
        # перебрать все записи в тренировочном наборе данных
        numstr = 0
        for i_img_num in sorted(img_ind_matr.keys()): #img_ind_matr.items():
            # получить список значений, используя символы запятой (',')
            # в качестве разделителей
            all_values = img_ind_matr[i_img_num]
            # масштабировать и сместить входные значения
            inputs = (all_values/ 255.0 * 0.99) + 0.01
            # создать целевые выходные значения (все равны 0.01, за исключением
            # желаемого маркерного значения, равного 0.99)
            targets = numpy.zeros(output_nodes_in) + 0.01
            targets[validation[i_img_num]] = 0.99
            n_in.train(inputs,targets)
            numstr += 1
            # Промежуточные записи (картинка с тренировкой сети)
            # делаем файл с изображением
            if (numstr % img_interval_in == 0):
                prefix_name_rec = "e"+str(e) + "n" + str(numstr)
                file_img_name = "NNLearn/"+ nnnum_in +"/imgH_" + prefix_name_rec
                CreateImgScreen(n_in.wih,file_img_name)

def savematrix(n_in,nnnum_in):
    nn_who_file = open("NNLearn/"+nnnum_in+"/nn01_who.txt", "wb")
    nn_wih_file = open("NNLearn/"+nnnum_in+"/nn01_wih.txt", "wb")
    numpy.savetxt(nn_who_file,n_in.who, fmt='%.18e', delimiter='\t', newline='\r\n' )
    numpy.savetxt(nn_wih_file,n_in.wih, fmt='%.18e', delimiter='\t', newline='\r\n' )
    nn_who_file.close()
    nn_wih_file.close()

def testfromCSV(test_data_file_name_in,output_nodes_in,n_in,nnnum_in,nntest_in):
    # загрузить в список тестовый набор данных CSV-файла набора MNIST
    test_data_file = open(test_data_file_name_in,'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # журнал оценок работы сети, первоначально пустой
    scorecard = []
    # переменные для различных счетчиков
    numstr = 0
    # матрица для хранения данных для анализа в разрезе цифр
    NumberValues = numpy.zeros((10,10))
    for i in range(10):
        NumberValues[i,0] = i
    for record in test_data_list:
        # перебрать все записи в тестовом наборе данных
        # получить список значений из записи, используя символы
        # запятой (',') в качестве разделителей
        all_values = record.split(',')
        # правильный ответ - первое значение
        correct_label = int(all_values[0])
        #print(correct_label, "истинный маркер")
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n_in.query(inputs)
        # индекс наибольшего значения является маркерным значением
        label = numpy.argmax(outputs)
        # максимальное значение веса ответа сети
        NumberValues[correct_label,3] += outputs[label]
        #print(label, "ответ сети")
        # # присоеденить оценку ответа сети к концу списка
        if (label == correct_label):
            # в случае правильного ответа сети
            # # присоеденить к списку значение 1
            scorecard.append(1)
            # увеличить число правильных ответов для цифры
            NumberValues[correct_label,1] += 1
        else: 
            # в случае неправильного ответа сети
            # # присоеденить к списку значение 0
            scorecard.append(0)
            # увеличить число неправильных ответов для цифры
            NumberValues[correct_label,2] += 1
            # запсать в папку неверно распознанное изображение
            inputs_list_img = (255 - numpy.asfarray(all_values[1:])) 
            inputs_img = inputs_list_img.reshape(28,28)
            image = inputs_img
            cv2.imwrite("NNReports/"+str(nntest_in)+"/ImgErr/"+str(label)+"_"+ str(correct_label) +"_img"+str(numstr)+".png", image) 
            numstr += 1
    # print(scorecard)
    # print(NumberValues)
    # рассчитать показатель эффективности в виде
    # доли правильных ответов
    scorecard_array = numpy.asarray(scorecard)
    print ("Эффективность = ", scorecard_array.sum() / scorecard_array.size)
    return str(scorecard_array.sum() / scorecard_array.size)
    # # обновить значения в отчетном массиве
    # find_ind_rep = list(test_rep[:,0]).index(test_num)
    # test_rep[find_ind_rep,1]=str(scorecard_array.sum() / scorecard_array.size)
    # здесь допишем в лог - файл
    # now = datetime.datetime.now()
    # test_rep[find_ind_rep,2]=now.strftime("%d-%m-%Y")    

def testfromPNG(test_data_file_name_in,output_nodes_in,n_in,nnnum_in,nntest_in):
    # создаем два пустых массива - для изодражений и для меток валидации, 
    # номер изображения пока берем из файла и пишем бощий массив пишем
    # потом по нему идем по эпохам несколько раз
    # журнал оценок работы сети, первоначально пустой
    scorecard = []
    # переменные для различных счетчиков
    numstr = 0
    # матрица для хранения данных для анализа в разрезе цифр
    NumberValues = numpy.zeros((10,10))
    for i in range(10):
        NumberValues[i,0] = i
    validation = {}
    img_ind_matr = {}
    # # делаем цикл по цифрам - пишем массив
    for i in range(10):
        for filepath in os.listdir(test_data_file_name_in+str(i)):
            image = cv2.imread(test_data_file_name_in+str(i)+'/'+filepath,0)
            img_nump = numpy.asarray(image, dtype='uint8')
            num_img = filepath[3:-4]
            validation[int(num_img)] = i
            img_data = 255.0 - img_nump.reshape(784)
            img_ind_matr[int(num_img)] = img_data
    # теперь идем по массиву с тестами
    for i_img_num in sorted(img_ind_matr.keys()): #img_ind_matr.items():
        # получить список значений, используя символы запятой (',')
        # в качестве разделителей
        all_values = img_ind_matr[i_img_num]
        correct_label = validation[i_img_num]
        # масштабировать и сместить входные значения
        inputs = (all_values/ 255.0 * 0.99) + 0.01
        outputs = n_in.query(inputs)
        # индекс наибольшего значения является маркерным значением
        label = numpy.argmax(outputs)
        # максимальное значение веса ответа сети
        NumberValues[correct_label,3] += outputs[label]
        #print(label, "ответ сети")
        # # присоеденить оценку ответа сети к концу списка
        if (label == correct_label):
            # в случае правильного ответа сети
            # # присоеденить к списку значение 1
            scorecard.append(1)
            # увеличить число правильных ответов для цифры
            NumberValues[correct_label,1] += 1
        else: 
            # в случае неправильного ответа сети
            # # присоеденить к списку значение 0
            scorecard.append(0)
            # увеличить число неправильных ответов для цифры
            NumberValues[correct_label,2] += 1
            # запсать в папку неверно распознанное изображение
            inputs_list_img = (255 - all_values) 
            inputs_img = inputs_list_img.reshape(28,28)
            image = inputs_img
            cv2.imwrite("NNReports/"+str(nntest_in)+"/ImgErr/"+str(label)+"_"+ str(correct_label) +"_img"+str(numstr)+".png", image) 
            numstr += 1
    scorecard_array = numpy.asarray(scorecard)
    print ("Эффективность = ", scorecard_array.sum() / scorecard_array.size)
    return str(scorecard_array.sum() / scorecard_array.size)
    # # обновить значения в отчетном массиве
    # find_ind_rep = list(test_rep[:,0]).index(test_num)
    # test_rep[find_ind_rep,1]=str(scorecard_array.sum() / scorecard_array.size)
    # здесь допишем в лог - файл
    # now = datetime.datetime.now()
    # test_rep[find_ind_rep,2]=now.strftime("%d-%m-%Y") 

