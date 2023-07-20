# this code we followed some implementations from kaggle on how to implemet our pickle file
# the link for the same https://www.kaggle.com/code/demesgal/train-inference-gpu-baseline-tta
# And also this https://www.kaggle.com/code/mahmudds/alaska2-image-eda-understanding-and-modeling
# we implememted this with the efficient net B0 model
from glob import glob
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

BASE_PATH = "ourapp/static/Test"
filepath='ourapp/static/models/pkl.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(name):
    if name == 'effnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
    model._fc = nn.Linear(model._fc.in_features, 4)  # 100 is an example.
    model = model.to(device)
    return model

nn.Softmax

def load_checkpoint(model, filepath):
    states_weights = torch.load(filepath, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(states_weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    return model

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

class LabelsAndPreds():
    @staticmethod
    def transform_labels_for_metric(labels):
        
        return torch.clamp(labels, min=0, max=1).detach().numpy()
        
    @staticmethod
    def transform_preds_for_metric(preds):
        
        preds = nn.Softmax()(preds)
        mx, indices = torch.max(preds, 1)
        mask = (indices!=0).to(torch.float) - (indices==0).to(torch.float) 
        preds = mask*mx
        
        return preds.detach().numpy()

model = load_model('effnet-b0')
effnet=load_checkpoint(model,filepath)


class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        path = f'{BASE_PATH}/{image_name}'
        
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]

test_dataset = DatasetSubmissionRetriever(
    image_names=np.array([path.split('/')[-1] for path in glob('ourapp/static/Test/*.jpg')]),
    transforms=get_valid_transforms(),
)

image_names=np.array([path.split('/')[-1] for path in glob('ourapp/static/Test/*.jpg')])
def test(network):
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    result = {'Id': [], 'Label': []}
    with torch.no_grad():
        for i, (image_names, images) in enumerate(test_data_loader):

            preds = network(images.to(device))
            preds = preds.cpu()
            preds = LabelsAndPreds.transform_preds_for_metric(preds)
            
            result['Id'].extend(image_names)
            result['Label'].extend(preds)

    return result


def predict():
    result = test(effnet)
    return result['Label']




