# ================== 高效分层融合LMPM - 主模型训练脚本 ==================

import torch
from torch import nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse
import time
from tabulate import tabulate
import random


# ================== 随机种子设置 ==================

def set_seed(seed):
    """设置随机种子确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== 数据集定义 ==================

class MultiModalDataset(Dataset):
    """多模态数据集（图像+文本）"""

    def __init__(self, data_list, tokenizer, transform, max_len=128, image_root="./data/images"):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.image_root = image_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 加载图像
        image_path = os.path.join(self.image_root, item["image"])
        try:
            img = Image.open(image_path).convert("RGB")
            image = self.transform(img)
        except Exception as e:
            print(f"[警告] 无法加载图片: {image_path}, 异常: {e}")
            return self.__getitem__((idx + 1) % len(self.data_list))

        # 处理文本
        question = item.get("question", "")
        encoding = self.tokenizer(
            question,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 标签
        label = item.get("topic", "")

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


# ================== 核心模型组件 ==================

class EfficientFusionLayer(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 自注意力：处理单模态内部信息
        self.self_attn_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # 交叉注意力：处理跨模态交互
        self.cross_attn_v2t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_t2v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # 层归一化
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)

        # 前馈神经网络
        self.ffn_v = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, visual_feat, text_feat, context_feat=None):

        # 添加序列维度用于注意力计算
        v_seq = visual_feat.unsqueeze(1)  # [batch, 1, dim]
        t_seq = text_feat.unsqueeze(1)  # [batch, 1, dim]

        # 步骤1: 自注意力处理
        v_self, _ = self.self_attn_v(v_seq, v_seq, v_seq)
        t_self, _ = self.self_attn_t(t_seq, t_seq, t_seq)

        v_self = self.norm_v1(v_seq + v_self).squeeze(1)
        t_self = self.norm_t1(t_seq + t_self).squeeze(1)

        # 步骤2: 交叉注意力处理（模态间交互）
        v_cross, _ = self.cross_attn_v2t(v_self.unsqueeze(1), t_self.unsqueeze(1), t_self.unsqueeze(1))
        t_cross, _ = self.cross_attn_t2v(t_self.unsqueeze(1), v_self.unsqueeze(1), v_self.unsqueeze(1))

        v_cross = v_cross.squeeze(1)
        t_cross = t_cross.squeeze(1)

        # 步骤3: 高效融合
        v_fused = (v_self + v_cross) / 2
        t_fused = (t_self + t_cross) / 2

        # 步骤4: 残差连接和层归一化
        v_fused = self.norm_v2(visual_feat + v_fused)
        t_fused = self.norm_t2(text_feat + t_fused)

        # 步骤5: 前馈网络
        v_out = v_fused + self.ffn_v(v_fused)
        t_out = t_fused + self.ffn_t(t_fused)

        # 生成新的上下文特征
        new_context = (v_out + t_out) / 2

        return v_out, t_out, new_context


class EfficientHierarchicalFusion(nn.Module):

    def __init__(self, visual_dim=768, text_dim=768, fusion_dim=768,
                 num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # 输入投影层（统一维度）
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        #  创新点1: 分层融合架构
        self.fusion_layers = nn.ModuleList([
            EfficientFusionLayer(fusion_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        #  创新点2: 自适应层权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # 最终融合门控
        self.final_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

        #  创新点3: 质量感知评分器
        self.quality_scorer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_feat, text_feat):

        # 投影到统一维度
        v_feat = self.visual_proj(visual_feat)
        t_feat = self.text_proj(text_feat)

        context = None
        v_outputs = []
        t_outputs = []

        #  分层处理：逐层深化模态交互
        for layer in self.fusion_layers:
            v_feat, t_feat, context = layer(v_feat, t_feat, context)
            v_outputs.append(v_feat)
            t_outputs.append(t_feat)

        #  自适应层权重组合
        layer_weights = torch.softmax(self.layer_weights, dim=0)
        v_combined = sum(w * v_out for w, v_out in zip(layer_weights, v_outputs))
        t_combined = sum(w * t_out for w, t_out in zip(layer_weights, t_outputs))

        #  质量感知融合
        v_quality = self.quality_scorer(v_combined)
        t_quality = self.quality_scorer(t_combined)

        total_quality = v_quality + t_quality + 1e-8
        v_weight = v_quality / total_quality
        t_weight = t_quality / total_quality

        # 最终融合：门控 + 质量加权
        fusion_gate = self.final_gate(torch.cat([v_combined, t_combined], dim=1))
        base_fusion = fusion_gate * v_combined + (1 - fusion_gate) * t_combined

        final_output = v_weight * v_combined + t_weight * t_combined
        final_output = 0.7 * final_output + 0.3 * base_fusion

        return final_output


class EfficientSwinALBEF(nn.Module):

    def __init__(self, num_classes, hidden_size=768):
        super().__init__()

        # 图像编码器：Swin Transformer
        self.image_encoder = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.image_encoder.head = nn.Identity()  # 移除分类头

        # 文本编码器：BERT
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese')

        # 核心融合模块
        self.fusion = EfficientHierarchicalFusion(
            visual_dim=hidden_size,
            text_dim=hidden_size,
            fusion_dim=hidden_size,
            num_layers=3,
            num_heads=8,
            dropout=0.1
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):

        # 提取图像特征
        img_feat = self.image_encoder(image)  # [batch, 768]

        # 提取文本特征
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.pooler_output  # [batch, 768]

        # 多模态融合
        fused_feat = self.fusion(img_feat, txt_feat)

        # 分类
        logits = self.classifier(fused_feat)

        return logits


# ================== 训练和评估函数 ==================

def evaluate_model(model, loader, topic_to_idx, device):

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_time = 0

    with torch.no_grad():
        for batch in loader:
            start_time = time.time()

            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 转换标签
            labels_idx = [topic_to_idx[l] for l in batch['label']]
            labels = torch.tensor(labels_idx, device=device)

            # 前向传播
            logits = model(images, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            total_time += time.time() - start_time

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Top-3准确率
    top3_correct = sum([1 if y in np.argsort(all_probs[i])[-3:] else 0
                        for i, y in enumerate(all_labels)])
    top3_accuracy = (top3_correct / len(all_labels)) * 100

    # 平均推理时间
    avg_inference_time = (total_time / len(loader)) * 1000  # ms per batch

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'top3_accuracy': top3_accuracy,
        'inference_time_ms': avg_inference_time
    }


def train_model(train_json, val_json, test_json, image_root,
                epochs=80, batch_size=32, lr=5e-6, seed=42,
                save_dir='./model'):

    print("=" * 80)
    print(" 高效分层融合HAPM - 主模型训练")
    print("=" * 80)

    # 设置随机种子
    set_seed(seed)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # 保存路径
    model_save_path = os.path.join(save_dir, 'efficient_swin_albef_main_best.pt')
    mapping_save_path = os.path.join(save_dir, 'efficient_swin_albef_main_mapping.json')
    log_save_path = './logs/efficient_swin_albef_main_training.json'

    # 加载数据
    print("\n 加载数据...")
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    print(f"   测试集: {len(test_data)} 样本")

    # 构建标签映射
    all_topics = sorted(set(item['topic'] for item in train_data + val_data + test_data))
    topic_to_idx = {topic: idx for idx, topic in enumerate(all_topics)}
    idx_to_topic = {idx: topic for topic, idx in topic_to_idx.items()}
    num_classes = len(all_topics)

    print(f"   类别数: {num_classes}")

    # 保存标签映射
    with open(mapping_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'topic_to_idx': topic_to_idx,
            'idx_to_topic': idx_to_topic
        }, f, ensure_ascii=False, indent=2)

    # 数据预处理
    print("\n 初始化数据预处理...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = MultiModalDataset(train_data, tokenizer, transform, image_root=image_root)
    val_dataset = MultiModalDataset(val_data, tokenizer, transform, image_root=image_root)
    test_dataset = MultiModalDataset(test_data, tokenizer, transform, image_root=image_root)

    # 数据加载器
    def collate_fn(batch):
        return {
            'image': torch.stack([x['image'] for x in batch]),
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'label': [x['label'] for x in batch]
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    print("\n  构建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    model = EfficientSwinALBEF(num_classes=num_classes).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 训练循环
    print("\n" + "=" * 80)
    print(" 开始训练")
    print("=" * 80)

    best_f1 = 0.0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    max_patience = 10

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # ============ 训练阶段 ============
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.tensor([topic_to_idx[l] for l in batch['label']], device=device)

            # 前向传播
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / num_batches

        # ============ 验证阶段 ============
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
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # 评估验证集
        val_metrics = evaluate_model(model, val_loader, topic_to_idx, device)

        # 学习率调度
        scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time

        # 打印epoch结果
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{epochs} 完成 (耗时: {epoch_time:.2f}s)")
        print(f"{'=' * 80}")
        print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 学习率: {current_lr:.2e}")
        print(f"验证指标:")
        print(f"  ✓ Accuracy:     {val_metrics['accuracy']:.2f}%")
        print(f"  ✓ F1-Score:     {val_metrics['f1']:.4f}")
        print(f"  ✓ Precision:    {val_metrics['precision']:.4f}")
        print(f"  ✓ Recall:       {val_metrics['recall']:.4f}")
        print(f"  ✓ Top-3 Acc:    {val_metrics['top3_accuracy']:.2f}%")

        # 保存训练历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_metrics'].append(val_metrics)

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_metrics = val_metrics
            patience_counter = 0

            torch.save(model.state_dict(), model_save_path)
            print(f" 新的最佳模型！F1: {best_f1:.4f} (已保存)")
        else:
            patience_counter += 1
            print(f" 未改进 ({patience_counter}/{max_patience})")

        # 早停
        if patience_counter >= max_patience:
            print(f"\n  早停触发！最佳F1: {best_f1:.4f} (Epoch {best_epoch})")
            break

        print()

    # ============ 测试阶段 ============
    print("\n" + "=" * 80)
    print(" 在测试集上评估最佳模型")
    print("=" * 80)

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    test_metrics = evaluate_model(model, test_loader, topic_to_idx, device)

    print(f"\n测试集结果:")
    print(f"  ✓ Accuracy:     {test_metrics['accuracy']:.2f}%")
    print(f"  ✓ F1-Score:     {test_metrics['f1']:.4f}")
    print(f"  ✓ Precision:    {test_metrics['precision']:.4f}")
    print(f"  ✓ Recall:       {test_metrics['recall']:.4f}")
    print(f"  ✓ Top-3 Acc:    {test_metrics['top3_accuracy']:.2f}%")
    print(f"  ✓ 推理时间:     {test_metrics['inference_time_ms']:.2f} ms/batch")

    # 保存完整训练日志
    training_log = {
        'model_config': {
            'num_classes': num_classes,
            'fusion_layers': 3,
            'num_heads': 8,
            'hidden_size': 768
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'seed': seed
        },
        'best_epoch': best_epoch,
        'best_val_metrics': best_metrics,
        'test_metrics': test_metrics,
        'training_history': training_history
    }

    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    print(f"\n 训练完成！")
    print(f"   模型保存于: {model_save_path}")
    print(f"   标签映射: {mapping_save_path}")
    print(f"   训练日志: {log_save_path}")

    return model_save_path, topic_to_idx, test_metrics


# ================== 主函数 ==================

def main():
    parser = argparse.ArgumentParser(description='高效分层融合HAPM训练脚本')

    # 数据路径
    parser.add_argument('--train_json', type=str, default="E:\\SRZ\\CBM\\data\\text\\train.json",
                        help='训练集JSON文件路径')
    parser.add_argument('--val_json', type=str, default="E:\\SRZ\\CBM\\data\\text\\val.json",
                        help='验证集JSON文件路径')
    parser.add_argument('--test_json', type=str, default="E:\\SRZ\\CBM\\data\\text\\test.json",
                        help='测试集JSON文件路径')
    parser.add_argument('--image_root', type=str, default="E:\\SRZ\\CBM\\data\\images",
                        help='图像根目录路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=80,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./model',
                        help='模型保存目录')

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 80)
    print("训练配置")
    print("=" * 80)
    print(f"训练集: {args.train_json}")
    print(f"验证集: {args.val_json}")
    print(f"测试集: {args.test_json}")
    print(f"图像目录: {args.image_root}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"随机种子: {args.seed}")
    print(f"保存目录: {args.save_dir}")
    print()

    # 开始训练
    train_model(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        image_root=args.image_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
