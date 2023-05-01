import sys
#sys.path.remove('/usr/lib/python3/dist-packages')
import pickle
import os 
import torch
import torch.nn as nn
import clip
from PIL import Image 
import numpy as np

# assign the dataset_loc variable to the dataset location

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
device="cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#print(image.shape)
#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#with torch.no_grad():
#    image_features = model.encode_image(image)
#    text_features = model.encode_text(text)
    
#    logits_per_image, logits_per_text = model(image, text)
#    probs = logits_per_image.softmax(dim=-1).cpu().numpy()


dataset_loc = 'cifar-10-python/cifar-10-batches-py/'
dataset_file = os.path.join(dataset_loc, 'data_batch_2')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset_dict = unpickle(dataset_file)

data = dataset_dict[b'data']
data_labels = dataset_dict[b'labels']
files = dataset_dict[b'filenames']

prompt = "Get me the images of cats"
text = clip.tokenize([prompt]).to(device)
prompt_features = model.encode_text(text).reshape((-1,1))
#print(prompt_features.shape)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_prompts = ["This is an image of the " + label for label in labels]
label_tokens = [clip.tokenize([pt]).to(device) for pt in label_prompts]
label_features = [model.encode_text(token) for token in label_tokens]
label_features = torch.cat(label_features, dim=0)
print(label_features.shape)

req_label = torch.argmax(nn.Softmax(dim=0)(torch.matmul(label_features, prompt_features))).item()
print(req_label)
print(len(data))

os.makedirs('cifar_images', exist_ok=True)

count = 0
truth = []
pred = []
images = []
for i in range(1000):
    print("image : ", i)
    np_image = np.zeros((32,32,3), dtype=np.uint8)
    np_image[:,:,0] = data[i][0:1024].reshape((32,32))
    np_image[:,:,1] = data[i][1024:2048].reshape((32,32))
    np_image[:,:,2] = data[i][2048:3096].reshape((32,32))
    image = preprocess(Image.fromarray(np_image, "RGB")).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features = image_features.reshape((-1,1))

    pred_label = torch.argmax(nn.Softmax(dim=0)(torch.matmul(label_features, image_features))).item()
    true_label = data_labels[i]
    if pred_label != true_label:
        count += 1
        #file_name = 'cifar_images/' + str(i) + " predicted-" + labels[pred_label] + " true-" + labels[true_label] + '.png'
        #image = Image.fromarray(np_image, "RGB")
        #image.save(file_name)


    if pred_label == req_label:
        pred.append(i)
        truth.append(true_label)

print(count)

# the pred list contains the list of images that belong to the class "cats"


