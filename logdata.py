import os
import io
import itertools
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.reset()
        
    def _generate_matrix(self, target, pred):
        np_target = target.cpu().detach().numpy() if torch.is_tensor(target) else target
        np_pred = pred.cpu().detach().numpy() if torch.is_tensor(pred) else pred
        label = self.num_class *np_target + np_pred
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
        
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix = self.confusion_matrix + self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), int)
        
    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_average_class_acc(self):
        return np.mean(self.confusion_matrix.diagonal() / np.sum(self.confusion_matrix, axis=1))


def log_confusion_matrix(writer, tag, confusion_matrix, class_names=None, step=None):
    if torch.is_tensor(confusion_matrix):
        CM = confusion_matrix.cpu().detach().numpy()
    else:
        CM = confusion_matrix
    figure = plot_confusion_matrix(CM, class_names)
    image = plot_to_image(figure)
    writer.add_image(tag, 
                     img_tensor=image, 
                     global_step=step,
                     dataformats="HWC"
                     )

def plot_confusion_matrix(cm, class_names=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    if class_names is None:
        class_names = np.arange(len(cm))
    figure = plt.figure(figsize=(32, 32), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis], decimals=0).astype(int)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        label = "-" if np.isnan(labels[i, j]) else labels[i, j]
        plt.text(j, i, label, horizontalalignment="center", color=color)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
  
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches = 'tight')
    buf.seek(0)
    # Convert buffer to image
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Closing the figure and buffer
    plt.close(figure)
    buf.close()
    return image