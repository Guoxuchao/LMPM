import torch
from torch import nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import io
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse
import time
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiModalDataset(Dataset):

    def __init__(self, data_list, tokenizer, transform, max_len=128, image_root='./data/images'):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.image_root = image_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = os.path.join(self.image_root, item['image'])
        try:
            img = Image.open(image_path).convert('RGB')
            image = self.transform(img)
        except Exception as e:
            print(f'[警告] 无法加载图片: {image_path}, 异常: {e}')
            return self.__getitem__((idx + 1) % len(self.data_list))
        question = item.get('question', '')
        encoding = self.tokenizer(question, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = item.get('topic', '')
        return {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

class EfficientFusionLayer(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.self_attn_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_v2t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_t2v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.ffn_v = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout))
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout))

    def forward(self, visual_feat, text_feat, context_feat=None):
        v_seq = visual_feat.unsqueeze(1)
        t_seq = text_feat.unsqueeze(1)
        v_self, _ = self.self_attn_v(v_seq, v_seq, v_seq)
        t_self, _ = self.self_attn_t(t_seq, t_seq, t_seq)
        v_self = self.norm_v1(v_seq + v_self).squeeze(1)
        t_self = self.norm_t1(t_seq + t_self).squeeze(1)
        v_cross, _ = self.cross_attn_v2t(v_self.unsqueeze(1), t_self.unsqueeze(1), t_self.unsqueeze(1))
        t_cross, _ = self.cross_attn_t2v(t_self.unsqueeze(1), v_self.unsqueeze(1), v_self.unsqueeze(1))
        v_cross = v_cross.squeeze(1)
        t_cross = t_cross.squeeze(1)
        v_fused = (v_self + v_cross) / 2
        t_fused = (t_self + t_cross) / 2
        v_fused = self.norm_v2(visual_feat + v_fused)
        t_fused = self.norm_t2(text_feat + t_fused)
        v_out = v_fused + self.ffn_v(v_fused)
        t_out = t_fused + self.ffn_t(t_fused)
        new_context = (v_out + t_out) / 2
        return (v_out, t_out, new_context)

class EfficientHierarchicalFusion(nn.Module):

    def __init__(self, visual_dim=768, text_dim=768, fusion_dim=768, num_layers=3, num_heads=8, dropout=0.1, fusion_alpha=0.7):
        super().__init__()
        self.num_layers = num_layers
        self.fusion_alpha = fusion_alpha
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.fusion_layers = nn.ModuleList([EfficientFusionLayer(fusion_dim, num_heads, dropout) for _ in range(num_layers)])
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.final_gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Tanh(), nn.Linear(fusion_dim, 1), nn.Sigmoid())
        self.quality_scorer = nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Linear(fusion_dim // 2, 1), nn.Sigmoid())

    def get_layer_alpha(self):
        return torch.softmax(self.layer_weights, dim=0)

    def forward(self, visual_feat, text_feat):
        v_feat = self.visual_proj(visual_feat)
        t_feat = self.text_proj(text_feat)
        context = None
        v_outputs = []
        t_outputs = []
        for layer in self.fusion_layers:
            v_feat, t_feat, context = layer(v_feat, t_feat, context)
            v_outputs.append(v_feat)
            t_outputs.append(t_feat)
        layer_weights = self.get_layer_alpha()
        v_combined = sum((w * v_out for w, v_out in zip(layer_weights, v_outputs)))
        t_combined = sum((w * t_out for w, t_out in zip(layer_weights, t_outputs)))
        v_quality = self.quality_scorer(v_combined)
        t_quality = self.quality_scorer(t_combined)
        total_quality = v_quality + t_quality + 1e-08
        v_weight = v_quality / total_quality
        t_weight = t_quality / total_quality
        fusion_gate = self.final_gate(torch.cat([v_combined, t_combined], dim=1))
        base_fusion = fusion_gate * v_combined + (1 - fusion_gate) * t_combined
        final_output = v_weight * v_combined + t_weight * t_combined
        alpha = self.fusion_alpha
        final_output = alpha * final_output + (1 - alpha) * base_fusion
        return final_output

class EfficientSwinALBEF(nn.Module):

    def __init__(self, num_classes, hidden_size=768, fusion_alpha=0.7):
        super().__init__()
        self.image_encoder = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.image_encoder.head = nn.Identity()
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese')
        self.fusion = EfficientHierarchicalFusion(visual_dim=hidden_size, text_dim=hidden_size, fusion_dim=hidden_size, num_layers=3, num_heads=8, dropout=0.1, fusion_alpha=fusion_alpha)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, num_classes))

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.pooler_output
        fused_feat = self.fusion(img_feat, txt_feat)
        logits = self.classifier(fused_feat)
        return logits

def compute_loss(logits, labels, model, lam):
    criterion = nn.CrossEntropyLoss()
    ce = criterion(logits, labels)
    if lam and lam > 0:
        alpha = model.fusion.get_layer_alpha()
        L = alpha.numel()
        reg = (alpha - 1.0 / L).abs().sum()
        return (ce + lam * reg, ce.item(), reg.item())
    return (ce, ce.item(), 0.0)

def evaluate_model(model, loader, topic_to_idx, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_idx = [topic_to_idx[l] for l in batch['label']]
            labels = torch.tensor(labels_idx, device=device)
            logits = model(images, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    micro_f1 = f1_score(all_labels, all_preds, average='micro') * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    top3_correct = sum([1 if y in np.argsort(all_probs[i])[-3:] else 0 for i, y in enumerate(all_labels)])
    top3_accuracy = top3_correct / len(all_labels) * 100
    return {'accuracy': accuracy, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'precision': precision, 'recall': recall, 'top3_accuracy': top3_accuracy}

def report_parameters(model):
    print('\n 参数量统计 (分模块):')
    breakdown = {}
    for name, module in [('image_encoder (Swin-T)', model.image_encoder), ('text_encoder (BERT)', model.text_encoder), ('fusion (HCI+ASR+MQA+Gate)', model.fusion), ('classifier (MLP)', model.classifier)]:
        n = sum((p.numel() for p in module.parameters()))
        breakdown[name] = n
        print(f'   {name:35s}: {n:>12,} ({n / 1000000.0:.2f} M)')
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    breakdown['total'] = total
    breakdown['trainable'] = trainable
    print(f"   {'总参数量':35s}: {total:>12,} ({total / 1000000.0:.2f} M)")
    print(f"   {'可训练参数':35s}: {trainable:>12,} ({trainable / 1000000.0:.2f} M)")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = buf.getbuffer().nbytes / 1024 ** 2
    breakdown['model_size_mb'] = size_mb
    print(f'   模型体积 (FP32): {size_mb:.1f} MB')
    return breakdown

def compute_flops(model, device):
    try:
        from thop import profile
        img = torch.randn(1, 3, 224, 224).to(device)
        ids = torch.ones(1, 128, dtype=torch.long).to(device)
        mask = torch.ones(1, 128, dtype=torch.long).to(device)
        flops, _ = profile(model, inputs=(img, ids, mask), verbose=False)
        gflops = flops / 1000000000.0
        print(f'\n 计算量: {gflops:.2f} GFLOPs / 单次推理 (batch=1)')
        return gflops
    except ImportError:
        print('\n [提示] 未安装 thop，跳过 FLOPs 统计。可执行: pip install thop')
        return None
    except Exception as e:
        print(f'\n [提示] FLOPs 统计失败 ({e})，跳过')
        return None

@torch.no_grad()
def benchmark_inference(model, tokenizer, device, batch_sizes=(1, 8, 16, 32, 64), warmup=10, num_batches=50):
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    enc = tokenizer('这张图片中的作物得了什么病虫害？', max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    print(f'\n  推理效率评测 (设备: {device}):')
    header = f"   {'batch':>5} | {'ms/batch':>9} | {'ms/sample':>10} | {'samples/s':>10}"
    if device.type == 'cuda':
        header += f" | {'peak GPU MB':>11}"
    print(header)
    print('   ' + '-' * (len(header) - 3))
    results = {}
    for bs in batch_sizes:
        img = dummy_img.repeat(bs, 1, 1, 1).to(device)
        ids = enc['input_ids'].repeat(bs, 1).to(device)
        mask = enc['attention_mask'].repeat(bs, 1).to(device)
        for _ in range(warmup):
            _ = model(img, ids, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(num_batches):
            _ = model(img, ids, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        entry = {'batch_size': bs, 'ms_per_batch': round(elapsed / num_batches * 1000, 2), 'ms_per_sample': round(elapsed / num_batches / bs * 1000, 3), 'throughput_samples_per_sec': round(bs * num_batches / elapsed, 1)}
        line = f"   {bs:>5} | {entry['ms_per_batch']:>9.2f} | {entry['ms_per_sample']:>10.3f} | {entry['throughput_samples_per_sec']:>10.1f}"
        if device.type == 'cuda':
            entry['peak_gpu_memory_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
            line += f" | {entry['peak_gpu_memory_mb']:>11.1f}"
        print(line)
        results[bs] = entry
        del img, ids, mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return results

@torch.no_grad()
def benchmark_cpu_single(model, tokenizer, warmup=5, num_batches=20):
    cpu_model = model.to('cpu')
    cpu_model.eval()
    img = torch.randn(1, 3, 224, 224)
    enc = tokenizer('这张图片中的作物得了什么病虫害？', max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    ids, mask = (enc['input_ids'], enc['attention_mask'])
    for _ in range(warmup):
        _ = cpu_model(img, ids, mask)
    t0 = time.perf_counter()
    for _ in range(num_batches):
        _ = cpu_model(img, ids, mask)
    elapsed = time.perf_counter() - t0
    ms = elapsed / num_batches * 1000
    print(f'\n CPU 单样本推理延迟 (边缘设备参考): {ms:.1f} ms/sample')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return round(ms, 1)

def run_efficiency_report(model, tokenizer, device, save_path):
    report = {}
    report['parameters'] = report_parameters(model)
    report['gflops'] = compute_flops(model, device)
    report['gpu_benchmark'] = benchmark_inference(model, tokenizer, device)
    report['cpu_single_sample_ms'] = benchmark_cpu_single(model, tokenizer)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n 效率报告已保存: {save_path}')
    return report

def train_model(train_json, val_json, test_json, image_root, epochs, batch_size, lr, seed, save_dir, fusion_alpha, lam, freeze_backbone=False, run_efficiency=False):
    print(' 高效分层融合SwinALBEF - 主模型训练')
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    model_save_path = os.path.join(save_dir, f'efficient_swin_albef_main_best_seed{seed}.pt')
    mapping_save_path = os.path.join(save_dir, f'efficient_swin_albef_main_mapping_seed{seed}.json')
    log_save_path = f'./logs/efficient_swin_albef_main_training_seed{seed}.json'
    print('\n 加载数据...')
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f'   训练集: {len(train_data)} 样本')
    print(f'   验证集: {len(val_data)} 样本')
    print(f'   测试集: {len(test_data)} 样本')
    all_topics = sorted(set((item['topic'] for item in train_data + val_data + test_data)))
    topic_to_idx = {topic: idx for idx, topic in enumerate(all_topics)}
    idx_to_topic = {idx: topic for topic, idx in topic_to_idx.items()}
    num_classes = len(all_topics)
    print(f'   类别数: {num_classes}')
    with open(mapping_save_path, 'w', encoding='utf-8') as f:
        json.dump({'topic_to_idx': topic_to_idx, 'idx_to_topic': idx_to_topic}, f, ensure_ascii=False, indent=2)
    print('\n 初始化数据预处理...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = MultiModalDataset(train_data, tokenizer, transform, image_root=image_root)
    val_dataset = MultiModalDataset(val_data, tokenizer, transform, image_root=image_root)
    test_dataset = MultiModalDataset(test_data, tokenizer, transform, image_root=image_root)

    def collate_fn(batch):
        return {'image': torch.stack([x['image'] for x in batch]), 'input_ids': torch.stack([x['input_ids'] for x in batch]), 'attention_mask': torch.stack([x['attention_mask'] for x in batch]), 'label': [x['label'] for x in batch]}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print('\n  构建模型...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   使用设备: {device}')
    model = EfficientSwinALBEF(num_classes=num_classes, fusion_alpha=fusion_alpha).to(device)
    if freeze_backbone:
        for p in model.image_encoder.parameters():
            p.requires_grad_(False)
        for p in model.text_encoder.parameters():
            p.requires_grad_(False)
        print('     骨干网络已冻结 (Swin-T / BERT 不参与训练)')
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'   总参数量: {total_params:,}')
    print(f'   可训练参数: {trainable_params:,}')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    print(f' 开始训练 (seed={seed}, lam={lam}, φ={fusion_alpha})')
    best_f1 = 0.0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    max_patience = 5
    training_history = {'train_loss': [], 'train_reg': [], 'val_loss': [], 'val_metrics': []}
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_reg = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.tensor([topic_to_idx[l] for l in batch['label']], device=device)
            logits = model(images, input_ids, attention_mask)
            loss, ce_val, reg_val = compute_loss(logits, labels, model, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += ce_val
            train_reg += reg_val
            num_batches += 1
            if (batch_idx + 1) % 10 == 0:
                print(f'   Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        avg_train_loss = train_loss / num_batches
        avg_train_reg = train_reg / num_batches
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = torch.tensor([topic_to_idx[l] for l in batch['label']], device=device)
                logits = model(images, input_ids, attention_mask)
                loss, _, _ = compute_loss(logits, labels, model, lam)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches
        val_metrics = evaluate_model(model, val_loader, topic_to_idx, device)
        scheduler.step(val_metrics['macro_f1'])
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        alpha_now = model.fusion.get_layer_alpha().detach().cpu().numpy().round(4).tolist()
        print(f"\n{'=' * 80}")
        print(f'Epoch {epoch + 1}/{epochs} 完成 (耗时: {epoch_time:.2f}s)')
        print(f"{'=' * 80}")
        print(f'训练损失: {avg_train_loss:.4f} | 正则项: {avg_train_reg:.4f} | 验证损失: {avg_val_loss:.4f} | 学习率: {current_lr:.2e}')
        print(f'ASR层权重 α: {alpha_now}')
        print(f'验证指标:')
        print(f"   Accuracy:     {val_metrics['accuracy']:.2f}%")
        print(f"   Micro F1:     {val_metrics['micro_f1']:.2f}%")
        print(f"   Macro F1:     {val_metrics['macro_f1']:.2f}%")
        print(f"   Precision:    {val_metrics['precision']:.4f}")
        print(f"   Recall:       {val_metrics['recall']:.4f}")
        print(f"   Top-3 Acc:    {val_metrics['top3_accuracy']:.2f}%")
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_reg'].append(avg_train_reg)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_metrics'].append(val_metrics)
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_epoch = epoch + 1
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f' 新的最佳模型！F1: {best_f1:.4f} (已保存)')
        else:
            patience_counter += 1
            print(f' 未改进 ({patience_counter}/{max_patience})')
        if patience_counter >= max_patience:
            print(f'\n  早停触发！最佳F1: {best_f1:.4f} (Epoch {best_epoch})')
            break
        print()
    print(' 在测试集上评估最佳模型')
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, topic_to_idx, device)
    print(f'\n测试集结果 (seed={seed}):')
    print(f"   Accuracy:     {test_metrics['accuracy']:.2f}%")
    print(f"   Micro F1:     {test_metrics['micro_f1']:.2f}%")
    print(f"   Macro F1:     {test_metrics['macro_f1']:.2f}%")
    print(f"   Precision:    {test_metrics['precision']:.4f}")
    print(f"   Recall:       {test_metrics['recall']:.4f}")
    print(f"   Top-3 Acc:    {test_metrics['top3_accuracy']:.2f}%")
    final_alpha = model.fusion.get_layer_alpha().detach().cpu().numpy().round(4).tolist()
    print(f'   最终ASR层权重 α: {final_alpha}')
    efficiency_report = None
    if run_efficiency:
        eff_path = f'./logs/efficiency_report_seed{seed}.json'
        efficiency_report = run_efficiency_report(model, tokenizer, device, eff_path)
    training_log = {'model_config': {'num_classes': num_classes, 'fusion_layers': 3, 'num_heads': 8, 'hidden_size': 768, 'freeze_backbone': freeze_backbone}, 'training_config': {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'seed': seed, 'lam': lam, 'fusion_alpha': fusion_alpha, 'optimizer': 'AdamW', 'weight_decay': 0.01, 'scheduler': 'ReduceLROnPlateau(factor=0.5, patience=3)', 'early_stop_patience': 5}, 'best_epoch': best_epoch, 'best_val_metrics': best_metrics, 'test_metrics': test_metrics, 'final_layer_alpha': final_alpha, 'efficiency': efficiency_report, 'training_history': training_history}
    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f'\n 训练完成！')
    print(f'   模型保存于: {model_save_path}')
    print(f'   标签映射: {mapping_save_path}')
    print(f'   训练日志: {log_save_path}')
    return (model_save_path, topic_to_idx, test_metrics)

def summarize_multi_seed(all_metrics):
    print(' 多种子结果汇总 (mean ± std)')
    keys = ['accuracy', 'micro_f1', 'macro_f1', 'precision', 'recall', 'top3_accuracy']
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[k] = {'mean': round(mean_v, 2), 'std': round(std_v, 2), 'runs': [round(v, 2) for v in vals]}
        print(f"   {k:>14}: {mean_v:.2f} ± {std_v:.2f}   {['%.2f' % v for v in vals]}")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, required=True)
    parser.add_argument('--test_json', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seeds', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--fusion_alpha', type=float, required=True)
    parser.add_argument('--lam', type=float, required=True)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--run_efficiency', action='store_true')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    print('  训练配置')
    print(f'训练集: {args.train_json}')
    print(f'验证集: {args.val_json}')
    print(f'测试集: {args.test_json}')
    print(f'图像目录: {args.image_root}')
    print(f'训练轮数: {args.epochs}')
    print(f'批次大小: {args.batch_size}')
    print(f'学习率: {args.lr}')
    print(f'随机种子: {seeds}')
    print(f'正则项系数 λ: {args.lam}')
    print(f'融合系数 φ: {args.fusion_alpha}')
    print(f'冻结骨干: {args.freeze_backbone}')
    print(f'效率评测: {args.run_efficiency}')
    print(f'保存目录: {args.save_dir}')
    print()
    all_metrics = []
    for seed in seeds:
        print(f'种子 {seed} / {seeds}')
        _, _, test_metrics = train_model(train_json=args.train_json, val_json=args.val_json, test_json=args.test_json, image_root=args.image_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=seed, save_dir=args.save_dir, fusion_alpha=args.fusion_alpha, lam=args.lam, freeze_backbone=args.freeze_backbone, run_efficiency=args.run_efficiency and seed == seeds[0])
        all_metrics.append(test_metrics)
    if len(all_metrics) > 1:
        summary = summarize_multi_seed(all_metrics)
        summary_path = './logs/multi_seed_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({'seeds': seeds, 'summary': summary}, f, ensure_ascii=False, indent=2)
        print(f'\n 多种子汇总已保存: {summary_path}')
if __name__ == '__main__':
    main()
