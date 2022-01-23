##########################################
#   这里主要是为了实现对于LibSVM数据集的适应
#   标准数据格式
#    1 32 56 43 45 2 43 54 76 76
'''
+1 4:-0.320755
-1 1:0.583333 2:-1 3:0.333333
+1 1:0.166667 2:1 3:-0.333333 4:-0.433962
-1 1:0.458333 3:1 4:-0.358491
'''
#   写得比较乱……
#########################################

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from tqdm import tqdm



# 首先将数据逐行写入txt,为格式1
def getTXT(filepath,dict):
    f=open(filepath,'w')
    for i in range(len(dict['labels'])):
        f.write(str(dict['labels'][i]))
        for j in range(len(dict['data'][i])):
            #print(type(dict['data'][i]))
            f.write(' '+str(dict['data'][i][j]))
        f.write('\n')
    f.close()

#getTXT('test1.txt',dict)

def covert(src_file,target_file):
    # 将指定的数据转换成目标文件
    # read data file
    readin = open(src_file, 'r')
    # write data file
    output = open(target_file, 'w')
    try:
        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\n')
            index = 0
            output_line = ''
            for sub_line in the_line.split(' '):
                # the label col
                if index == 0:
                    output_line = sub_line
                # the features cols
                if sub_line != 'NULL' and index != 0:
                    the_text = ' ' + str(index) + ':' + sub_line
                    output_line = output_line + the_text
                index = index + 1
            output_line = output_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()

def dataGenerate(filepath,tgpath):
    """
    filepath:生成的原始txt文件的路径
    tagpath:最终生成的输入到libsvm中的文件路径
    生成所需要的LibSVM所需要的文件格式
    :return:
    """
    # 一共有5个batch需操作五次形成txt的原始文件


    f = open(filepath, 'w')
    for i in range(5):
        # 读取数据存入字典中
        # 转化成灰度图像来计算
        file='D:/homework_for_ML/task4/data/data_batch_{}'.format(i+1)
        fp=open(file,'rb')
        dict=pickle.load(fp, encoding='iso-8859-1')
        fp.close()
        # 将字典数据写入txt文件
        for i in range(len(dict['labels'])):
            f.write(str(dict['labels'][i]))
            for j in range(len(dict['data'][i])):
                if j>=1024:
                    break
                # print(type(dict['data'][i]))
                # 灰度计算Gray = (R*299 + G*587 + B*114 + 500) / 1000
                gray=(dict['data'][i][j]*299+dict['data'][i][j+1024]*587+dict['data'][i][2*1024+j]*114+500)/1000
                f.write(' ' + str(gray/255.0))
            f.write('\n')
    f.close()
    # 将原始文件输入进行转化
    covert(filepath,tgpath)

#dataGenerate("gray_scale_train.txt","gray_scale_train_svm.txt")

# 生成测试数据
def dataGenerate_test(filepath,tgpath):
    """
    filepath:生成的原始txt文件的路径
    tagpath:最终生成的输入到libsvm中的文件路径
    生成所需要的LibSVM所需要的文件格式
    :return:
    """
    # 一共有5个batch需操作五次形成txt的原始文件
    file1 = 'D:/homework_for_ML/task4/data/test_batch'
    fp = open(file1, 'rb')
    dict = pickle.load(fp, encoding='iso-8859-1')
    fp.close()
    f = open(filepath, 'w')
    for i in range(1):
        # 读取数据存入字典中
        # 转化成灰度图像来计算
        file=file1
        fp=open(file,'rb')
        dict=pickle.load(fp, encoding='iso-8859-1')
        fp.close()
        # 将字典数据写入txt文件
        for i in range(len(dict['labels'])):
            f.write(str(dict['labels'][i]))
            for j in range(len(dict['data'][i])):
                if j>=1024:
                    break
                # print(type(dict['data'][i]))
                # 灰度计算Gray = (R*299 + G*587 + B*114 + 500) / 1000
                gray=(dict['data'][i][j]*299+dict['data'][i][j+1024]*587+dict['data'][i][2*1024+j]*114+500)/1000
                f.write(' ' + str(gray/255.0))
            f.write('\n')
    f.close()
    # 将原始文件输入进行转化
    covert(filepath,tgpath)
#dataGenerate_test("gray_scale_test.txt","gray_scale_test_svm.txt")


#生成的是灰度图像的特征值
#dataGenerate_test('tsne_train.txt','tsne_svm_train.txt')

"""对于图片数据降维"""
def TSNE_pre(sfile,tfile):
    # 首先读取数据
    y = []
    data=[]
    i=0
    try:
        file = open(sfile, 'r')
    except FileNotFoundError:
        print('File is not found')
    else:
        lines = file.readlines()
        for line in lines:
            i+=1

            a = line.split()
            a=list(map(float,a))
            #print(a)
            label = a[0]
            x=a[1:1026]

            y.append(int(label))
            data.append(x)
    file.close()
    print(data)


    '''t-SNE'''
    x=np.array(data)

    tsne = manifold.TSNE(n_components=2, perplexity=32,init='pca', random_state=501)
    X_tsne = tsne.fit_transform(x)


    # 下面是将数据映射到txt中
    f=open(tfile,'w')
    for i in range(X_tsne.shape[0]):

            #print(type(dict['data'][i]))
        f.write(str(y[i])+' '+str(X_tsne[i][0])+' '+str(X_tsne[i][1]))
        f.write('\n')

    f.close()





#这里降维成为2维度数据
#TSNE_pre("D:/homework_for_ML/task4/基于torch框架的NN/data/gray_src_test.txt","svm_test_v3.txt")
#TSNE_pre("D:/homework_for_ML/task4/基于torch框架的NN/data/gray_src_.txt","svm_train_v3.txt")
#covert("svm_train_v3.txt","final_svm_train_v3.txt")
#covert("svm_test_v3.txt","final_svm_test_v3.txt")

# 将指定文件化成目标文件
# 这里还需要读取之前的label标签

# 绘图验证数据的正确性与否
def drawTest(filename):
    import matplotlib.pyplot as plt
    # 首先读取数据
    y = []
    data = []
    i = 0
    try:
        file = open(filename, 'r')
    except FileNotFoundError:
        print('File is not found')
    else:
        lines = file.readlines()
        for line in lines:
            i += 1

            a = line.split()
            a = list(map(float, a))
            # print(a)
            label = a[0]
            x = a[1:3]

            y.append(int(label))
            data.append(x)
    file.close()

    cmap=['b','c','g','k','m','r','y','orange','purple','brown','gray']

    # 绘制
    for i in range(len(data)):
        #print(data[i])
        if i>1000:
            break
        plt.scatter(x=data[i][0],y=data[i][1],c=cmap[y[i]])

    plt.show()

#drawTest('D:/homework_for_ML/task4/基于torch框架的NN/data/svm_test_v3.txt')
#covert("D:/homework_for_ML/task4/data/pca_test.txt","pca_77_svm_train.txt")
#covert("D:/homework_for_ML/task4/data/pca_train.txt","pca_77_svm_test.txt")






