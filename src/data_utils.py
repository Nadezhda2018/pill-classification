import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
'''
Normalization: Мы используем среднее и стандартное отклонение $(\mu=0.485, \sigma=0.229)$, характерные для ImageNet. Это стандарт индустрии, так как большинство предобученных 
нейросетей (ResNet, EfficientNet) обучались именно на этих значениях.Data Augmentation: 
Для обучающей выборки добавлен RandomResizedCrop и RandomHorizontalFlip. Это "встряхивает" модель, 
помогая ей не заучивать конкретные положения таблеток в кадре.DataLoader: Параметр num_workers=2 
ускоряет загрузку данных за счет многопоточности.
'''
def get_data_loaders(train_path, test_path, batch_size=32, img_size=224):
    # Стандартная предобработка: Resizing, CenterCrop и нормализация под ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(test_path, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset, val_dataset