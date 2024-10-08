import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Задаём пути к данным
data_dir = 'database/Face_Mask_Data'  # Путь к папке с папками `train`, `test`, `validation`
train_dir = os.path.join(data_dir, 'Train')
test_dir = os.path.join(data_dir, 'Test')
val_dir = os.path.join(data_dir, 'Validation')

# Параметры
batch_size = 32
num_epochs = 20  # Увеличим количество эпох для лучшего обучения
learning_rate = 0.001
num_classes = 2  # Два класса: "лицо" и "маска"

# Преобразования данных (Resize, Augmentation и Normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Аугментация: случайное горизонтальное отражение
        transforms.RandomRotation(10),  # Аугментация: случайная поворот на 10 градусов
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Загрузка данных
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
val_data = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Загрузка предобученной модели ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Заменим последний слой для нужного количества классов
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Использование CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Добавление scheduler для уменьшения learning rate, если validation loss не уменьшается
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

# Функция обучения
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = 100. * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Шаг для scheduler
        scheduler.step(epoch_loss)

    print("Обучение завершено.")
    return model

# Функция проверки на валидационных данных
def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Функция тестирования на тестовом наборе данных
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Прямой проход
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Обучаем модель
trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)

# Проверяем модель на валидационном наборе
validate_model(trained_model, val_loader)

# Тестируем модель
test_model(trained_model, test_loader)

# Сохранение модели
torch.save(trained_model.state_dict(), 'face_mask_model.pth')
print("Модель сохранена как face_mask_model.pth")
