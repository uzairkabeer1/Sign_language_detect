import torch
from torchvision import datasets, transforms
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

predicted_labels = []
true_labels = []

model = torch.load("signlang_model.pth")

data_transforms = {
    'Train_Alphabet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test_Alphabet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'C:/Sign Language Detection/Module 1 (sign to text)/processed_dataset/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Train_Alphabet', 'Test_Alphabet']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['Train_Alphabet', 'Test_Alphabet']}

for inputs, labels in dataloaders['Test_Alphabet']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    predicted_labels.extend(preds.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')

print('Accuracy: {:.4f}'.format(accuracy))
print('F1 Score: {:.4f}'.format(f1))
print('Recall: {:.4f}'.format(recall))
print('Precision: {:.4f}'.format(precision))
