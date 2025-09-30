import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import clip
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pycocotools.coco import COCO
import os
import pandas as pd  


class ZeroShotOpenVocabularyDetector:
    def __init__(self, mask_rcnn_config_path, clip_model_name="ViT-B/32", device="cuda"):
        # 设置运行设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载 CLIP 模型
        print("加载 CLIP 模型...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # 配置 Mask R-CNN（仅用于区域提案和掩码预测）
        print("配置 Mask R-CNN 模型...")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(mask_rcnn_config_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_rcnn_config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 检测阈值
        self.cfg.MODEL.DEVICE = str(self.device)

        # 创建 Mask R-CNN 预测器
        self.mask_rcnn_predictor = DefaultPredictor(self.cfg)

    def precompute_text_embeddings(self, category_prompts):
        """预计算类别提示的文本嵌入"""
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in category_prompts]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
        return text_features

    def extract_roi_embeddings(self, image, boxes):
        """提取边界框区域的 CLIP 图像嵌入"""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        roi_embeddings = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                roi_embedding = torch.zeros(512).to(self.device)
            else:
                roi = image_pil.crop((x1, y1, x2, y2))
                roi_preprocessed = self.clip_preprocess(roi).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    roi_embedding = self.clip_model.encode_image(roi_preprocessed)
                    roi_embedding = roi_embedding / roi_embedding.norm(dim=-1, keepdim=True)
            roi_embeddings.append(roi_embedding.squeeze())
        return torch.stack(roi_embeddings) if roi_embeddings else torch.tensor([]).to(self.device)

    def classify_with_clip(self, roi_embeddings, text_features, category_prompts):
        """使用 CLIP 余弦相似度进行分类，返回每个 RoI 对所有类别的概率"""
        if len(roi_embeddings) == 0:
            return []

        similarity = (roi_embeddings @ text_features.T) * 100  # 相似度
        probs = similarity.softmax(dim=-1)  # softmax 概率

        results = []
        for i in range(len(roi_embeddings)):
            pred_idx = torch.argmax(probs[i]).item()
            top1_category = category_prompts[pred_idx]
            top1_conf = probs[i][pred_idx].item()
            all_conf = {cat: probs[i][j].item() for j, cat in enumerate(category_prompts)}

            results.append({
                "category": top1_category,
                "confidence": top1_conf,
                "all_confidences": all_conf
            })
        return results

    def detect(self, image_path, category_prompts):
        """执行零样本检测"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")

        # 预计算类别文本嵌入
        text_features = self.precompute_text_embeddings(category_prompts)

        # 使用 Mask R-CNN 获取检测结果
        outputs = self.mask_rcnn_predictor(image)
        instances = outputs["instances"]

        if len(instances) == 0:
            return [], image

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        masks = instances.pred_masks.cpu().numpy()

        # 提取 RoI 图像嵌入并分类
        roi_embeddings = self.extract_roi_embeddings(image, boxes)
        results = self.classify_with_clip(roi_embeddings, text_features, category_prompts)

        detections = []
        for i in range(len(results)):
            detections.append({
                "box": boxes[i],
                "mask": masks[i],
                "category": results[i]["category"],
                "confidence": results[i]["confidence"],
                "all_confidences": results[i]["all_confidences"]
            })
        return detections, image


# ----------------- COCO 测评函数 -----------------
def evaluate_on_coco(detector, coco_img_dir, coco_ann_file, max_images=200, save_csv=True):
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()

    # 加载 COCO 类别
    categories = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in categories]
    prompts = [f"a photo of a {c}" for c in cat_names]
    text_features = detector.precompute_text_embeddings(prompts)

    print("\n==== 使用的 COCO 类别 (前 20 个示例) ====")
    for i, p in enumerate(prompts[:20]):
        print(f"{i+1}. {p}")
    print(f"... 共 {len(prompts)} 个类别\n")

    total_gt = 0
    correct = 0
    all_clip_scores = []

    # 每类统计 {cat_name: {"gt": x, "correct": y}}
    per_class_stats = {c: {"gt": 0, "correct": 0} for c in cat_names}

    print(f"开始在 COCO 上评估 (最多 {max_images if max_images else len(img_ids)} 张图片)...")

    for idx, img_id in enumerate(img_ids[:max_images if max_images else len(img_ids)]):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])

        detections, image = detector.detect(img_path, prompts)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gts = coco.loadAnns(ann_ids)

        total_gt += len(gts)

        # 简化版评估逻辑: 取检测的top1和GT对比
        for gt in gts:
            gt_cat = coco.loadCats(gt['category_id'])[0]['name']
            gt_prompt = f"a photo of a {gt_cat}"

            per_class_stats[gt_cat]["gt"] += 1  # 累计GT数

            if detections:
                det = max(detections, key=lambda d: d["confidence"])
                if det["category"] == gt_prompt:
                    correct += 1
                    per_class_stats[gt_cat]["correct"] += 1
                all_clip_scores.append(det["confidence"])

        if (idx+1) % 50 == 0:
            print(f"已处理 {idx+1} 张图片...")

    acc = correct / total_gt if total_gt > 0 else 0
    avg_clip = np.mean(all_clip_scores) if all_clip_scores else 0

    print("\n==== Evaluation Results ====")
    print(f"Total GT objects: {total_gt}")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Average CLIPScore (RoI-level): {avg_clip:.4f}")

    # 输出每类的准确率
    print("\n==== Per-class Accuracy (COCO 80 类别) ====")
    per_class_results = []
    for cat, stats in per_class_stats.items():
        if stats["gt"] > 0:
            class_acc = stats["correct"] / stats["gt"]
        else:
            class_acc = 0.0
        print(f"{cat:15s}  Acc: {class_acc:.4f}  (GT: {stats['gt']}, Correct: {stats['correct']})")
        per_class_results.append({
            "category": cat,
            "accuracy": class_acc,
            "gt": stats["gt"],
            "correct": stats["correct"]
        })

    # 保存 CSV
    if save_csv:
        df = pd.DataFrame(per_class_results)
        csv_path = "coco_per_class_accuracy.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n📂 已保存每类准确率到 {csv_path}")


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    detector = ZeroShotOpenVocabularyDetector(
        mask_rcnn_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        clip_model_name="ViT-B/32"
    )

    # COCO 数据集评估
    coco_img_dir = "D:/coco/val2017"
    coco_ann_file = "D:/coco/annotations/instances_val2017.json"
    evaluate_on_coco(detector, coco_img_dir, coco_ann_file, max_images=5000, save_csv=True)
