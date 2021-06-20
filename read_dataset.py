from PIL import Image, ImageFile
import os



def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img


# 加载 fileList 并返回 imgList
def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

def number_samples_in_class(dataset_txt_file):
    #root_dir = '/home/iceicehyhy/Dataset/MNIST_224X224_3/train'

    img_list = default_reader(dataset_txt_file)
    samples_in_each_class = dict()

    for i in range (len(img_list)):
        train_img, label = img_list[i]
        if label in samples_in_each_class.keys():
                count += 1
                samples_in_each_class[label] = count
        else:
                samples_in_each_class[label] = 1
                count = 1

    sorted_dict = {k: samples_in_each_class[k] for k in sorted(samples_in_each_class)}
    return sorted_dict, img_list
#img = self.loader(os.path.join(self.root, imgPath))