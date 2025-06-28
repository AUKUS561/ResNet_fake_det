import os
import json
import matplotlib.colors as mcolors
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
#from VGGNET import vgg as createmodel

#from model import convnext_base as createmodel
#from MobileVIT import mobile_vit_x_small as createmodel

from resnet import resnet101 as createmodel



class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[:, i]) - TP
            FN = np.sum(self.matrix[i, :]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
            F_1 = round((2 * Precision * Recall) / (Precision + Recall), 4)
            table.add_row([self.labels[i], Precision, Recall, Specificity, F_1])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)

        # 规范化混淆矩阵的值，确保在0到1之间
        normalized_matrix = matrix / matrix.sum(axis=1, keepdims=True)

        plt.imshow(normalized_matrix, cmap='Greens', norm=mcolors.Normalize(vmin=0, vmax=1))

        # 确保 x/y 轴刻度和 labels 数量匹配
        ax = plt.gca()
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.labels, rotation=45)
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_yticklabels(self.labels)

        # 显示colorbar
        plt.colorbar()
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Confusion Matrix')

        # 在图中标注原始的计数值
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                cell_value = matrix[y, x]
                normalized_value = normalized_matrix[y, x]
                plt.text(x, y, cell_value,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if normalized_value > 0.5 else "black")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_path = r"E:\resnet_fake_det\resnet_fake\data"
    validate_dataset = datasets.ImageFolder(valid_path,
                                            transform=data_transform)

    batch_size = 4
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=8)
    net = createmodel(num_classes=2)
    # load pretrain weights
    model_weight_path = "./weights/best_model.pth"
    #assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    net.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
