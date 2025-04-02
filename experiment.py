import pickle
import time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding= 'latin1')
    return dict

dic = unpickle("/home/yeongyoo/cifar-10-batches-py/data_batch_1")

print(dic["batch_label"])
print("labels: ", dic["labels"][:10])
print("data: ", dic["data"][:10])
print("filenames: ", dic["filenames"][:10])