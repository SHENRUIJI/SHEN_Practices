import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import pandas as pd
import numpy as np
import pytrec_eval
from torch.utils.data import WeightedRandomSampler
# 检查是否有可用 GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载及处理
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为256x256
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 标准化
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为256x256
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 标准化
])

# 指定训练数据的路径
train_root = 'autodl-fs/data_split/train'
# 指定验证数据的路径
valid_root = 'autodl-fs/data_split/val'

# 图像读取转换
train_data = torchvision.datasets.ImageFolder(
    root=train_root,
    transform=train_transform
)

valid_data = torchvision.datasets.ImageFolder(
    root=valid_root,
    transform=test_transform
)

dic = train_data.class_to_idx  # 类别映射表
print(dic)

LR = 0.0001  # 学习率
EPOCH = 100  # 训练轮数
BATCH_SIZE = 32  # 批量大小

# 计算每个类别的样本数量
class_counts = Counter(train_data.targets)

# 计算每个类别的样本权重
weights = [1.0 / class_counts[class_idx] for class_idx in train_data.targets]

# 创建一个权重采样器
sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)

# 训练数据加载器
train_set = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4  # 使用4个子进程加载数据
)

# 验证数据加载器
test_set = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4  # 使用4个子进程加载数据
)

# 加载预训练的 ResNet-50 模型，并替换掉最后一层全连接层（fc），使其适应当前任务（共3个类别）
model_1 = torchvision.models.resnet50(weights=None)
num_features = model_1.fc.in_features
model_1.fc = nn.Linear(num_features, 3)

# 设置模型为训练模式
model_1.to(DEVICE)
optimizer = optim.SGD(model_1.parameters(), lr=LR, momentum=0.9)

# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

max_accuracy = 0.0
best_model = None

# 用于保存训练和验证指标的列表
metrics = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_samples = 0  # 用于统计总样本数
    running_loss = 0.0  # 用于计算平均损失
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
        total_samples += len(x)
        loss = nn.CrossEntropyLoss()(output, y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total_samples
    print(f"Train Epoch: {epoch} Loss: {avg_loss:.6f} Accuracy: {train_acc:.2f}%")
    return avg_loss, train_acc

def valid(model, device, dataset):
    model.eval()
    all_preds = []
    all_targets = []
    all_probabilities = []
    with torch.no_grad():
        for i, (x, target) in enumerate(dataset):
            x, target = x.to(device), target.to(device)
            output = model(x)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True).view_as(target)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.append(prob.cpu().numpy())

    # 合并所有批次的概率
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    # 计算准确率
    val_acc = accuracy_score(all_targets, all_preds)
    val_precision = precision_score(all_targets, all_preds, average='weighted')
    val_recall = recall_score(all_targets, all_preds, average='weighted')
    val_f1 = f1_score(all_targets, all_preds, average='weighted')

    # 计算其他指标
    qrels = {}
    run = {}

    for i, (target, prob) in enumerate(zip(all_targets, all_probabilities)):
        qrels[str(i)] = {str(target): 1}
        run[str(i)] = {str(j): float(p) for j, p in enumerate(prob)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    results = evaluator.evaluate(run)

    val_map = np.mean([v['map'] for v in results.values()])
    val_ndcg = np.mean([v['ndcg'] for v in results.values()])

    # 假设我们只关心top-1预测
    top1_pred = all_probabilities.argmax(axis=1)
    hit_rate = (top1_pred == all_targets).mean()

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Precision: {val_precision * 100:.2f}%")
    print(f"Validation Recall: {val_recall * 100:.2f}%")
    print(f"Validation F1 Score: {val_f1 * 100:.2f}%")
    print(f"Validation MAP: {val_map:.4f}")
    print(f"Validation NDCG: {val_ndcg:.4f}")
    print(f"Validation Hit Rate: {hit_rate * 100:.2f}%")

    # 打印详细分类报告
    report = classification_report(all_targets, all_preds, target_names=dic.keys(), zero_division=0)
    print(report)

    return {
        'accuracy': val_acc * 100,
        'precision': val_precision * 100,
        'recall': val_recall * 100,
        'f1': val_f1 * 100,
        'map': val_map,
        'ndcg': val_ndcg,
        'hit_rate': hit_rate * 100
    }

for epoch in range(1, EPOCH + 1):
    start_time = time.time()
    train_loss, train_acc = train(model_1, DEVICE, train_set, optimizer, epoch)
    val_metrics = valid(model_1, DEVICE, test_set)
    end_time = time.time()
    scheduler.step()  # 更新学习率

    # 收集指标
    metric = {
        'epoch': epoch,
        'time': end_time - start_time,
        'train_loss': train_loss,
        'train_acc': train_acc,
        **val_metrics,
        'lr': optimizer.param_groups[0]['lr']  # 假设只有一个参数组
    }
    metrics.append(metric)

    # 保存最佳模型
    if val_metrics['accuracy'] > max_accuracy:
        max_accuracy = val_metrics['accuracy']
        best_model = model_1.state_dict()

print("Maximum success rate: ", max_accuracy)

# 定义保存路径
save_path = f"model/best_model_train{max_accuracy:.2f}.pth"

# 确保目录存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 保存模型
torch.save(best_model, save_path)

print(f"Model has been saved to: {save_path}")

# 保存指标到CSV文件
csv_file = 'training_metrics.csv'
df = pd.DataFrame(metrics)
df.to_csv(csv_file, index=False)

print(f"Metrics saved to {csv_file}")