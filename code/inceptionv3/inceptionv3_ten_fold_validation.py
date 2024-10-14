import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import inception_v3
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import copy
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use Device: {device}")

    num_epochs = 60
    n_splits = 10
    learning_rate = 5e-4
    batch_size = 64
    num_workers = 10
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root='/mnt/code/Dataset', transform=transform)

    targets = [sample[1] for sample in dataset.samples]

    kf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    fold_results = {'Accuracy': [], 'Specificity': [], 'Precision': [], 'F1_Score': [], 'Recall': [], 'ROC_AUC': []}
    model_weights = []
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []
    all_features = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.samples, targets)):
        print(f'Fold {fold + 1}/{n_splits} :')

        train_subset = data.Subset(dataset, train_idx)
        val_subset = data.Subset(dataset, val_idx)

        train_loader = data.DataLoader(train_subset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        fold_val_loader = data.DataLoader(val_subset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
        model = inception_v3(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

        fold_training_losses = []
        fold_validation_losses = []
        fold_training_accuracies = []
        fold_validation_accuracies = []
    
        best_val_acc = 0.0
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} "):
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss1 = criterion(outputs.logits, labels)
                loss2 = criterion(outputs.aux_logits, labels)
                loss = loss1 + 0.4 * loss2
                loss.backward()
                optimizer.step()
                pred = torch.argmax(outputs.logits, 1)
                running_corrects += torch.sum(pred == labels.data)
                running_loss += loss.item() * inputs.size(0)
    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.to(torch.float32) / len(train_loader.dataset)
            fold_training_losses.append(epoch_loss)
            fold_training_accuracies.append(epoch_acc.item())
    
            print(f'Epoch {epoch + 1}/{num_epochs} : Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}')
            
            model.eval()
            y_true = []
            y_pred = []
            y_scores = []
            features = []
            running_loss = 0.0
            running_corrects = 0

            with torch.no_grad():
                for inputs, labels in tqdm(fold_val_loader, desc=f"Val Epoch {epoch + 1}/{num_epochs} "):
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    pred = torch.argmax(outputs.data, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
                    y_scores.append(torch.sigmoid(outputs.data).cpu().detach())

                    features.append(outputs.cpu().numpy())

                    running_corrects += torch.sum(pred == labels.data)
                    running_loss += loss.item() * inputs.size(0)
        
            y_scores = torch.cat(y_scores, dim=0)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_scores = y_scores[:, 1].numpy()
            features = np.concatenate(features, axis=0)

            val_loss = running_loss / len(fold_val_loader.dataset)
            val_acc = running_corrects.to(torch.float32) / len(fold_val_loader.dataset)
            fold_validation_losses.append(val_loss)
            fold_validation_accuracies.append(val_acc.item())
            print(f'Val Epoch {epoch + 1}/{num_epochs} : Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}\n')
            
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
    
        all_train_losses.append(fold_training_losses)
        all_val_losses.append(fold_validation_losses)
        all_train_accuracies.append(fold_training_accuracies)
        all_val_accuracies.append(fold_validation_accuracies)
    
        model_weights.append(best_model_weights)
    
        model.load_state_dict(best_model_weights)
        model.eval()
        y_true = []
        y_pred = []
        y_scores = []
        features = []
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(fold_val_loader, desc=f"Final Val Fold {fold + 1} "):
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_scores.append(torch.sigmoid(outputs.data).cpu().detach())

                features.append(outputs.cpu().numpy())

                running_corrects += torch.sum(pred == labels.data)
    
        y_scores = torch.cat(y_scores, dim=0)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = y_scores[:, 1].numpy()
        features = np.concatenate(features, axis=0)
    
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_scores.extend(y_scores)
        all_features.extend(features)
        all_labels.extend(y_true)

        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
        else:
            specificity = 0

        fold_results['Accuracy'].append(accuracy)
        fold_results['Specificity'].append(specificity)
        fold_results['Precision'].append(precision)
        fold_results['F1_Score'].append(f1)
        fold_results['Recall'].append(recall)
        fold_results['ROC_AUC'].append(roc_auc)
    
        print(f'Fold {fold+1} : Accuracy: {accuracy:.4f} - Specificity: {specificity:.4f} - Precision: {precision:.4f} - F1_Score: {f1:.4f} - Recall: {recall:.4f} - ROC_AUC: {roc_auc:.4f}\n')

    avg_weights = copy.deepcopy(model_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(model_weights)):
            avg_weights[key] += model_weights[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(model_weights))

    torch.save(avg_weights, '/mnt/code/InceptionV3/inceptionv3_average_weights.pth')

    with open('/mnt/code/InceptionV3/inceptionv3_training_results.pkl', 'wb') as f:
        pickle.dump({
            'train_acc_history': all_train_accuracies,
            'val_acc_history': all_val_accuracies,
            'train_loss_history': all_train_losses,
            'val_loss_history': all_val_losses,
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_scores': all_y_scores,
            'features': all_features,
            'labels': all_labels
        }, f)

    final_results = 'Final Results:\n'
    for metric in fold_results:
        result = f'{metric}: {np.mean(fold_results[metric]):.4f} ± {np.std(fold_results[metric]):.4f}'
        print(result)
        final_results += result + '\n'

    with open('/mnt/code/InceptionV3/inceptionv3_final_results.txt', 'w') as f:
        f.write(final_results)