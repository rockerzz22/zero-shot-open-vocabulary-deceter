# zero_shot_fvml_eval_coco.py
import os
import torch
import torchvision
from torchvision import transforms
from torchvision.ops import roi_align
import cv2
import numpy as np
from PIL import Image
import clip
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO
import pandas as pd


class ZeroShotFVLMDetector:
    def __init__(
        self,
        mask_rcnn_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", # Mask R-CNN 配置文件路径
        clip_model_name="ViT-B/32",  # 使用的 CLIP 模型
        device="cuda",
        vlm_pool_size=(1, 1),        # RoI 对齐后的池化输出大小
        proj_dim=512,                # 中间投影维度
        w_det=0.5,                   # 检测器分数权重
        w_vlm=0.5,                   # VLM 相似度权重
    ):
        # 设置运行设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] device: {self.device}")

        # ---------------- CLIP 模型 ----------------
        print("[INFO] loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()
        self.clip_dim = getattr(self.clip_model.visual, "output_dim", 512)  # CLIP 输出维度

        # ---------------- Detectron2 Mask R-CNN ----------------
        print("[INFO] configuring Detectron2 Mask R-CNN ...")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(mask_rcnn_config_path)) # 加载配置
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_rcnn_config_path) # 加载预训练权重
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # 检测阈值
        self.cfg.MODEL.DEVICE = str(self.device)
        self.predictor = DefaultPredictor(self.cfg) # 构建预测器

        # ---------------- ResNet50 作为 VLM backbone ----------------
        print("[INFO] loading ResNet50 backbone for VLM features...")
        resnet = torchvision.models.resnet50(pretrained=True)
        # 取出 ResNet50 的前向特征提取部分
        layers = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        ]
        self.vlm_backbone = nn.Sequential(*layers).to(self.device).eval()

        # 预处理：归一化图像输入
        self.resnet_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.vlm_pool_size = vlm_pool_size
        # 投影层：2048 -> proj_dim -> clip_dim
        self.proj = nn.Linear(2048, proj_dim).to(self.device)
        self.final_proj = nn.Linear(proj_dim, self.clip_dim).to(self.device) if proj_dim != self.clip_dim else None

        # 权重参数
        self.w_det = float(w_det)
        self.w_vlm = float(w_vlm)
        self.eps = 1e-8

    # --------- 预计算文本嵌入（类别提示词） ---------
    def precompute_text_embeddings(self, category_prompts):
        tokens = clip.tokenize(category_prompts).to(self.device)
        with torch.no_grad():
            t = self.clip_model.encode_text(tokens) # 编码文本
            t = t.float()
            t = t / (t.norm(dim=-1, keepdim=True) + self.eps) # L2 归一化
        return t

    # --------- 提取整张图的 VLM 特征图 ---------
    def extract_vlm_feature_map(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.resnet_preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fmap = self.vlm_backbone(x)
        return fmap

    # --------- 对 RoI 区域做特征池化并映射到 CLIP 空间 ---------
    def compute_vlm_roi_embeddings(self, feat_map, boxes, image_size):
        _, C, Hf, Wf = feat_map.shape
        H, W = image_size
        if boxes.shape[0] == 0:
            return torch.empty((0, self.clip_dim), device=self.device)

        # 计算 RoIAlign 需要的缩放比例
        spatial_scale = float(Hf) / float(H)
        boxes_for_roi = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].astype(float).tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = max(x1 + 1e-4, x2), max(y1 + 1e-4, y2)
            boxes_for_roi.append([0, x1, y1, x2, y2]) # batch_idx=0
        boxes_tensor = torch.tensor(boxes_for_roi, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # RoIAlign -> 池化成固定大小
            pooled = roi_align(feat_map, boxes_tensor, output_size=self.vlm_pool_size,
                               spatial_scale=spatial_scale, sampling_ratio=2)
            # 平均池化后投影
            pooled = pooled.view(pooled.shape[0], C, -1).mean(dim=-1)
            proj = self.proj(pooled)
            if self.final_proj is not None:
                proj = self.final_proj(proj)
            proj = proj.float()
            proj = proj / (proj.norm(dim=-1, keepdim=True) + self.eps) # 归一化
        return proj

    # --------- 检测流程 ---------
    def detect(self, image_path, category_prompts):
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        H, W = image.shape[:2]

        # 文本嵌入
        text_feats = self.precompute_text_embeddings(category_prompts)

        # Mask R-CNN 预测候选框
        outputs = self.predictor(image)
        instances = outputs.get("instances", None)
        if instances is None or len(instances) == 0:
            return []

        try:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
        except Exception:
            boxes = np.array([]).reshape(0, 4)

        try:
            det_scores = instances.scores.cpu().numpy()
        except Exception:
            det_scores = np.ones((boxes.shape[0],), dtype=np.float32) * 0.5

        # 提取 VLM 特征并计算相似度
        feat_map = self.extract_vlm_feature_map(image)
        vlm_embeds = self.compute_vlm_roi_embeddings(feat_map, boxes, (H, W))

        if vlm_embeds.shape[0] > 0:
            vlm_sims = (vlm_embeds @ text_feats.T).detach().cpu().numpy()
        else:
            vlm_sims = np.zeros((0, text_feats.shape[0]))

        # 将相似度从 [-1,1] 归一化到 [0,1]
        vlm_sims_norm = (vlm_sims + 1.0) / 2.0

        # 融合 Mask R-CNN 分数和 VLM 相似度
        final_scores = []
        M, K = boxes.shape[0], text_feats.shape[0]
        for i in range(M):
            det_prior = float(det_scores[i]) if i < len(det_scores) else 0.0
            combined = self.w_det * det_prior + self.w_vlm * vlm_sims_norm[i]
            final_scores.append(combined)
        final_scores = np.stack(final_scores, axis=0) if len(final_scores) > 0 else np.zeros((0, K))

        # 取每个候选框对应的最佳类别
        detections = []
        for i in range(M):
            best_idx = int(np.argmax(final_scores[i]))
            detections.append({
                "box": boxes[i],
                "chosen_category": category_prompts[best_idx],
                "confidence": float(final_scores[i, best_idx]),
            })
        return detections


# ----------------- 在 COCO 数据集上评估 -----------------
def evaluate_on_coco(detector, coco_img_dir, coco_ann_file, max_images=200, save_csv=True):
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()

    # 获取类别
    categories = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in categories]
    prompts = [f"a photo of a {c}" for c in cat_names]

    print(f"评估类别数: {len(prompts)}")

    total_gt = 0
    correct = 0
    # 每类统计
    per_class_stats = {c: {"gt": 0, "correct": 0} for c in cat_names}

    # 遍历图像
    for idx, img_id in enumerate(img_ids[:max_images]):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])

        detections = detector.detect(img_path, prompts) # 检测
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gts = coco.loadAnns(ann_ids)

        total_gt += len(gts)

        # 匹配 GT 和预测
        for gt in gts:
            gt_cat = coco.loadCats(gt['category_id'])[0]['name']
            gt_prompt = f"a photo of a {gt_cat}"
            per_class_stats[gt_cat]["gt"] += 1

            if detections:
                det = max(detections, key=lambda d: d["confidence"]) # 取置信度最高的预测
                if det["chosen_category"] == gt_prompt:
                    correct += 1
                    per_class_stats[gt_cat]["correct"] += 1

        if (idx+1) % 50 == 0:
            print(f"已处理 {idx+1} 张图片...")

    # 总体准确率
    acc = correct / total_gt if total_gt > 0 else 0
    print("\n==== Evaluation Results ====")
    print(f"Total GT objects: {total_gt}")
    print(f"Overall Accuracy: {acc:.4f}")

    # 输出每类结果
    results = []
    for cat, stats in per_class_stats.items():
        acc = stats["correct"] / stats["gt"] if stats["gt"]
