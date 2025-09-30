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
            # top1 预测
            pred_idx = torch.argmax(probs[i]).item()
            top1_category = category_prompts[pred_idx]
            top1_conf = probs[i][pred_idx].item()

            # 每个类别的概率
            all_conf = {cat: probs[i][j].item() for j, cat in enumerate(category_prompts)}

            results.append({
                "category": top1_category,
                "confidence": top1_conf,
                "all_confidences": all_conf  # 新增字段
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
            print("未检测到任何物体")
            return None, image

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
                "all_confidences": results[i]["all_confidences"]  # 新增字段
            })
        return detections, image

    def visualize(self, image, detections, save_path="zero_shot_results.jpg"):
        """可视化检测结果"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)

        for det in detections:
            box = det["box"].astype(int)
            mask = det["mask"]

            # 去掉前缀 "a photo of a "
            category = det["category"].replace("a photo of a ", "").strip()
            confidence = det["confidence"]

            # 绘制边界框
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1, edgecolor="black", facecolor="none")
            ax.add_patch(rect)

            # 绘制掩码
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = (1, 0, 0, 0.3)
            ax.imshow(colored_mask)

            # 绘制标签（只显示 rabbit / zebra / microwave ...）
            ax.text(x1, y1-5, f"{category} ({confidence:.2f})",
                    fontsize=10, color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2))

        plt.axis("off")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"检测结果已保存为: {save_path}")

#使用实列
if __name__ == "__main__":
    detector = ZeroShotOpenVocabularyDetector(
        mask_rcnn_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        clip_model_name="ViT-B/32"
    )

    # 定义任意类别提示（零样本分类）
    categories = [
        "a photo of a zebra", "a photo of a microwave", "a photo of a red apple",
        "a photo of a rabbit", "a photo of a bear", "a photo of a dog", "a photo of a cat",
        "a photo of a white rabbit","a photo of a brown bear","a photo of a pink bear","a photo of a bird"
        ,"a photo of a yellew bird"
    ]

    image_path = "D:/animal.png"  # 修改为你的图片路径
    detections, image = detector.detect(image_path, categories)

    if detections:
        # 控制台打印完整类别置信度
        for i, det in enumerate(detections):
            print(f"\nRoI {i+1}:")
            for cat, conf in det["all_confidences"].items():
                print(f"  {cat}: {conf:.4f}")

        # 可视化 
        detector.visualize(image, detections)
