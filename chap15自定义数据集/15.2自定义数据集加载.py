# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/26 16:06
"""
实际应用中，样本以及样本标签的存储方式可能各不相同，如有些场合所有的图片存储在同一目录下，类别名可从图片名字中推导出，
例如文件名为“pikachu_asxes0132.png”的图片，其类别信息可从文件名 pikachu 提取出。有些数据集样本的标签信息保存为 JSON 格式的文本文件中，
需要按照 JSON 格式查询每个样本的标签。不管数据集是以什么方式存储的，我们总是能够用过逻辑规则获取所有样本的路径和标签信息。

我们将自定义数据的加载流程抽象为如下步骤。

15.2.1创建编码表
样本的类别一般以字符串类型的类别名标记，但是对于神经网络来说，首先需要将类别名进行数字编码，然后在合适的时候再转换成 One-hot 编码或其他编码格式。
考虑𝑛个类 别的数据集，我们将每个类别随机编码为𝑙 ∈ [0, 𝑛 − 1]的数字，类别名与数字的映射关系称为编码表，一旦创建后，一般不能变动。

针对精灵宝可梦数据集的存储格式，我们通过如下方式创建编码表。首先按序遍历pokemon 根目录下的所有子目录，对每个子目标，
利用类别名作为编码表字典对象name2label 的键，编码表的现有键值对数量作为类别的标签映射数字，并保存进name2label 字典对象。
实现如下：
def load_pokemon(root, mode='train'):
    # 创建数字编码表
    name2label = {} # 编码表字典，"sq...":0
    # 遍历根目录下的子文件夹，并排序，保证映射关系固定
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过非文件夹对象
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())
         …

15.2.2创建样本和标签表格
编码表确定后，我们需要根据实际数据的存储方式获得每个样本的存储路径以及它的标签数字，分别表示为 images 和 labels 两个 List 对象。
其中 images List 存储了每个样本的路径字符串，labels List 存储了样本的类别数字，两者长度一致，且对应位置的元素相互关联。

我们将 images 和 labels 信息存储在 csv 格式的文件中，其中 csv 文件格式是一种以逗号符号分隔数据的纯文本文件格式，可以使用记事本或者 MS Excel 软件打开。
通过将所有样本信息存储在一个 csv 文件中有诸多好处，比如可以直接进行数据集的划分，可以随机采样 Batch 等。csv 文件中可以保存数据集所有样本的信息，
也可以根据训练集、验证集和测试集分别创建 3 个 csv 文件。最终产生的 csv 文件内容如图 15.3 所示，每行的第一个元素保存了当前样本的存储路径，
第二个元素保存了样本的类别数字。

csv 文件创建过程为：遍历 pokemon 根目录下的所有图片，记录图片的路径，并根据编码表获得其编码数字，作为一行写入到 csv 文件中，
代码如下：
def load_csv(root, filename, name2label):
    # 从 csv 文件返回 images,labels 列表
    # root:数据集根目录，filename:csv 文件名， name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        # 如果 csv 文件不存在，则创建
        images = []
        for name in name2label.keys(): # 遍历所有子目录，获得所有的图片
            # 只考虑后缀为 png,jpg,jpeg 的图片：'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # 打印数据集信息：1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)
        random.shuffle(images) # 随机打散顺序

        # 创建 csv 文件，并存储图片路径及其 label 信息
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0
                writer.writerow([img, label])
            print('written into csv file:', filename)
            ....
创建完 csv 文件后，下一次只需要从 csv 文件中读取样本路径和标签信息即可，而不需要每次都生成 csv 文件，提高计算效率，
代码如下：
def load_csv(root, filename, name2label):
    …
    # 此时已经有 csv 文件在文件系统上，直接读取
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)

    # 返回图片路径 list 和标签 list
    return images, labels


15.2.3数据集划分
数据集的划分需要根据实际情况来灵活调整划分比率。当数据集样本数较多时，可以选择 80%-10%-10%的比例分配给训练集、验证集和测试集；
当样本数量较少时，如这里的宝可梦数据集图片总数仅 1000 张左右，如果验证集和测试集比例只有 10%，则其图片数量约为 100 张，因此验证准确率和测试准确率可能波动较大。
对于小型的数据集，尽管样本数量较小，但还是需要适当增加验证集和测试集的比例，以保证获得准确的测试结果。
这里我们将验证集和测试集比例均设置为 20%，即有约 200 张图片用作验证和测试。

首先调用 load_csv 函数加载 images 和 labels 列表，根据当前模式参数 mode 加载对应部分的图片和标签。具体地，如果模式参数为 train，
则分别取 images 和 labels 的前 60%数据作为训练集；如果模式参数为 val，则分别取 images 和 labels 的 60%到 80%区域数据作为验证集；
如果模式参数为 test，则分别取 images 和 labels 的后 20%作为测试集。
代码实现如下：
def load_pokemon(root, mode='train'):
    …
    # 读取 Label 信息
    # [file1,file2,], [3,1]
    images, labels = load_csv(root, 'images.csv', name2label)
    # 数据集划分
    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val': # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else: # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label
需要注意的是，每次运行时的数据集划分方案需固定，防止使用测试集的样本训练，导致模型泛化性能不准确。
"""