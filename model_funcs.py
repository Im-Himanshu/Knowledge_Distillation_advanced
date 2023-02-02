import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


def train(model, out_model, train_data, test_data, num_epochs=1000, device="cuda", lr=0.001, eval_every=1000):
    device = torch.device(device)
    model.to(device)
    out_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    count = 0
    best_accuracy = 0
    best_loss = 1e10
    post_fix_dict = {}
    with tqdm(total=num_epochs) as bar:
        for epoch in range(num_epochs):
            bar.update(1)
            for dat in train_data:
                count += 1
                optimizer.zero_grad()
                model.train()
                x, y = dat
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out["preds"], y)
                loss.backward()
                optimizer.step()
                post_fix_dict.update({"Train Loss": loss.item(), "Steps": count})
                if count % eval_every == 0:
                    test_loss = 0
                    correct = 0
                    test_count = 0
                    total_count = 0
                    model.eval()
                    for dat in test_data:
                        x, y = dat
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        test_loss += crit(out["preds"], y).item()
                        preds = torch.argmax(F.softmax(out["preds"], dim=-1), dim=-1)
                        correct += torch.sum(preds == y).item()
                        test_count += len(preds)
                        total_count += 1
                    accuracy = correct / test_count
                    test_loss /= total_count
                    post_fix_dict.update({"Test Loss": test_loss, "Acc.": accuracy, "Best Acc.": best_accuracy, "Best Loss": best_loss})
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if best_loss > test_loss:
                            best_loss = test_loss
                        out_model.load_state_dict(model.state_dict())
                bar.set_postfix(post_fix_dict)

    return model, out_model

def train_kd_1(model, out_model, teacher_model, train_data, test_data, num_epochs=300, device="cuda", teach_weight=0.1, eval_every=1000):
    device = torch.device(device)
    model.to(device)
    out_model.to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    crit_layer = nn.MSELoss()
    count = 0
    best_accuracy = 0
    best_loss = 1e10
    post_fix_dict = {}
    teacher_model.eval()
    teacher_model.to(device)
    with tqdm(total=num_epochs) as bar:
        for epoch in range(num_epochs):
            bar.update(1)
            for dat in train_data:
                count += 1
                optimizer.zero_grad()
                model.train()
                x, y = dat
                x, y = x.to(device), y.to(device)
                out = model(x)
                out_teach = teacher_model(x)
                loss = crit(out["preds"], y)
                for i in range(1, 5):
                    stu_layer = out[f"layer{i}"]
                    teach_layer = out_teach[f"layer{i}"]
                    loss += teach_weight * crit_layer(stu_layer, teach_layer)
                    stu_layer_mean = torch.mean(stu_layer)
                    teach_layer_mean = torch.mean(teach_layer)
                    pearson_coeff = torch.sum((stu_layer - stu_layer_mean) * (teach_layer - teach_layer_mean))
                    pearson_coeff /= torch.sqrt(torch.sum((stu_layer - stu_layer_mean) ** 2) + 1e-10)
                    pearson_coeff /= torch.sqrt(torch.sum((teach_layer - teach_layer_mean) ** 2) + 1e-10)
                    loss += (1 - pearson_coeff)
                
                stu_layer = out[f"preds"]
                teach_layer = out_teach[f"preds"]
                loss += teach_weight * crit_layer(stu_layer, teach_layer)
                stu_layer_mean = torch.mean(stu_layer)
                teach_layer_mean = torch.mean(teach_layer)
                pearson_coeff = torch.sum((stu_layer - stu_layer_mean) * (teach_layer - teach_layer_mean))
                pearson_coeff /= torch.sqrt(torch.sum((stu_layer - stu_layer_mean) ** 2) + 1e-10)
                pearson_coeff /= torch.sqrt(torch.sum((teach_layer - teach_layer_mean) ** 2) + 1e-10)
                loss += (1 - pearson_coeff)
                    
                loss.backward()
                optimizer.step()
                post_fix_dict.update({"Train Loss": loss.item(), "Step": count})
                if count % eval_every == 0:
                    test_loss = 0
                    correct = 0
                    test_count = 0
                    total_count = 0
                    model.eval()
                    for dat in test_data:
                        x, y = dat
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        test_loss += crit(out["preds"], y).item()
                        preds = torch.argmax(F.softmax(out["preds"], dim=-1), dim=-1)
                        correct += torch.sum(preds == y).item()
                        test_count += len(preds)
                        total_count += 1
                    accuracy = correct / test_count
                    test_loss /= total_count
                    post_fix_dict.update({"Test Loss": test_loss, "Acc.": accuracy, "Best Acc.": best_accuracy, "Best Loss": best_loss})
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if best_loss > test_loss:
                            best_loss = test_loss
                        out_model.load_state_dict(model.state_dict())
                bar.set_postfix(post_fix_dict)

    return model, out_model

def get_layer_loss(stu_layer, teach_layer):
    loss = 0
    batch_size = stu_layer.size(0)
    ## Inter feature correlation
    stu_layer_mean = stu_layer.view(batch_size, -1).mean(1, keepdim=True)
    teach_layer_mean = teach_layer.view(batch_size, -1).mean(1, keepdim=True)
    stu_layer_view = stu_layer.view(batch_size, -1)
    teach_layer_view = teach_layer.view(batch_size, -1)
    pearson_coeff = ((stu_layer_view - stu_layer_mean) * (teach_layer_view - teach_layer_mean)).sum(1)
    pearson_coeff /= torch.sqrt(((stu_layer_view - stu_layer_mean) ** 2).sum(1) + 1e-10)
    pearson_coeff /= torch.sqrt(((teach_layer_view - teach_layer_mean) ** 2).sum(1) + 1e-10)
    loss += (1 - pearson_coeff.mean())
    ## Intra Feature Correlation
    stu_layer_mean = stu_layer.view(batch_size, -1).mean(0, keepdim=True)
    teach_layer_mean = teach_layer.view(batch_size, -1).mean(0, keepdim=True)
    stu_layer_view = stu_layer.view(batch_size, -1)
    teach_layer_view = teach_layer.view(batch_size, -1)
    pearson_coeff = ((stu_layer_view - stu_layer_mean) * (teach_layer_view - teach_layer_mean)).sum(0)
    pearson_coeff /= torch.sqrt(((stu_layer_view - stu_layer_mean) ** 2).sum(0) + 1e-10)
    pearson_coeff /= torch.sqrt(((teach_layer_view - teach_layer_mean) ** 2).sum(0) + 1e-10)
    loss += (1 - pearson_coeff.mean())
    return loss


def train_kd_2(model, out_model, teacher_model, train_data, test_data, num_epochs=1000, device="cuda", temp=4, eval_every=1000, lr=0.001, multi_gpu=False):
    device = torch.device(device)
    model.to(device)
    model.train()
    out_model.to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    crit_layer = nn.HuberLoss()
    count = 0
    best_accuracy = 0
    best_loss = 1e10
    post_fix_dict = {}
    teacher_model.eval()
    teacher_model.to(device)
    if multi_gpu:
        teacher_model = nn.DataParallel(teacher_model)
        model = nn.DataParallel(model)
    teach_weight = temp ** (-2)
    with tqdm(total=num_epochs) as bar:
        for epoch in range(num_epochs):
            bar.update(1)
            for dat in train_data:
                count += 1
                optimizer.zero_grad()
                model.train()
                x, y = dat
                x, y = x.to(device), y.to(device)
                out = model(x)
                out_teach = teacher_model(x)
                loss = crit(out["preds"], y)
                layer_loss = 0
                for i in range(1, 5):
                    stu_layer = out[f"layer{i}"]
                    teach_layer = out_teach[f"layer{i}"]
                    layer_loss += crit_layer(stu_layer, teach_layer)
                    layer_loss += get_layer_loss(stu_layer, teach_layer)
                
                loss += layer_loss / 4
                
                stu_layer = out[f"preds"]
                teach_layer = out_teach[f"preds"]
                loss += teach_weight * crit_layer(stu_layer, teach_layer)
                loss += get_layer_loss(stu_layer, teach_layer)
                    
                loss.backward()
                optimizer.step()
                post_fix_dict.update({"Train Loss": loss.item(), "Step": count})
                if count % eval_every == 0:
                    test_loss = 0
                    correct = 0
                    test_count = 0
                    total_count = 0
                    model.eval()
                    for dat in test_data:
                        x, y = dat
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        test_loss += crit(out["preds"], y).item()
                        preds = torch.argmax(F.softmax(out["preds"], dim=-1), dim=-1)
                        correct += torch.sum(preds == y).item()
                        test_count += len(preds)
                        total_count += 1
                    accuracy = correct / test_count
                    test_loss /= total_count
                    post_fix_dict.update({"Test Loss": test_loss, "Acc.": accuracy, "Best Acc.": best_accuracy, "Best Loss": best_loss})
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if best_loss > test_loss:
                            best_loss = test_loss
                        if multi_gpu:
                            out_model.load_state_dict(model.module.state_dict())
                        else:
                            out_model.load_state_dict(model.module.state_dict())
                bar.set_postfix(post_fix_dict)

    return model, out_model

def test(model, data, device="cuda"):
    device = torch.device(device)
    correct = 0
    test_count = 0
    total_count = 0
    model.eval()
    model.to(device)
    for dat in data:
        x, y = dat
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = torch.argmax(F.softmax(out["preds"], dim=-1), dim=-1)
        correct += torch.sum(preds == y).item()
        test_count += len(preds)
        total_count += 1
    accuracy = correct / test_count
    return accuracy