# zero_shot_fvml.py
import os
import torch
import torchvision
from torchvision import transforms
from torchvision.ops import roi_align
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import clip

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import torch.nn as nn


class ZeroShotFVLMDetector:
    def __init__(
        self,
        mask_rcnn_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        clip_model_name="ViT-B/32",
        device="cuda",
        vlm_pool_size=(1, 1),
        proj_dim=512,
        w_det=0.5,
        w_vlm=0.5,
    ):
        # 设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] device: {self.device}")

        # CLIP（用于文本 embedding）
        print("[INFO] loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()  #强制 float32
        # CLIP embedding dim (fallback to 512)
        self.clip_dim = getattr(self.clip_model.visual, "output_dim", 512)

        # Detectron2 Mask R-CNN
        print("[INFO] configuring Detectron2 Mask R-CNN ...")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(mask_rcnn_config_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_rcnn_config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        self.cfg.MODEL.DEVICE = str(self.device)
        self.predictor = DefaultPredictor(self.cfg)

        # VLM backbone: ResNet50
        print("[INFO] loading ResNet50 backbone for VLM features...")
        resnet = torchvision.models.resnet50(pretrained=True)
        layers = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        ]
        self.vlm_backbone = nn.Sequential(*layers).to(self.device).eval()

        # ResNet 预处理
        self.resnet_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.vlm_pool_size = vlm_pool_size
        self.proj = nn.Linear(2048, proj_dim).to(self.device)
        self.final_proj = nn.Linear(proj_dim, self.clip_dim).to(self.device) if proj_dim != self.clip_dim else None

        # 权重
        self.w_det = float(w_det)
        self.w_vlm = float(w_vlm)
        self.eps = 1e-8

    def precompute_text_embeddings(self, category_prompts):
        tokens = clip.tokenize(category_prompts).to(self.device)
        with torch.no_grad():
            t = self.clip_model.encode_text(tokens)
            t = t.float()   #保证 float32
            t = t / (t.norm(dim=-1, keepdim=True) + self.eps)
        return t

    def extract_vlm_feature_map(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.resnet_preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fmap = self.vlm_backbone(x)
        return fmap

    def compute_vlm_roi_embeddings(self, feat_map, boxes, image_size):
        _, C, Hf, Wf = feat_map.shape
        H, W = image_size
        if boxes.shape[0] == 0:
            return torch.empty((0, self.clip_dim), device=self.device)

        spatial_scale = float(Hf) / float(H)
        boxes_for_roi = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].astype(float).tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = max(x1 + 1e-4, x2), max(y1 + 1e-4, y2)
            boxes_for_roi.append([0, x1, y1, x2, y2])
        boxes_tensor = torch.tensor(boxes_for_roi, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pooled = roi_align(feat_map, boxes_tensor, output_size=self.vlm_pool_size,
                               spatial_scale=spatial_scale, sampling_ratio=2)
            pooled = pooled.view(pooled.shape[0], C, -1).mean(dim=-1)
            proj = self.proj(pooled)
            if self.final_proj is not None:
                proj = self.final_proj(proj)
            proj = proj.float()  # ✅ 保证 float32
            proj = proj / (proj.norm(dim=-1, keepdim=True) + self.eps)
        return proj

    def detect(self, image_path, category_prompts, save_path="fvml_out.jpg", visualize_masks=True,
               max_box_ratio=0.5):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        H, W = image.shape[:2]
        image_area = H * W  # 整张图的面积

        text_feats = self.precompute_text_embeddings(category_prompts)

        outputs = self.predictor(image)
        instances = outputs.get("instances", None)
        if instances is None or len(instances) == 0:
            print("[INFO] No instances detected.")
            return [], image

        try:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
        except Exception:
            boxes = np.array([]).reshape(0,4)

        masks = None
        try:
            if instances.has("pred_masks"):
                masks = instances.pred_masks.cpu().numpy()
        except Exception:
            masks = None

        try:
            det_scores = instances.scores.cpu().numpy()
        except Exception:
            det_scores = np.ones((boxes.shape[0],), dtype=np.float32) * 0.5

        feat_map = self.extract_vlm_feature_map(image)
        vlm_embeds = self.compute_vlm_roi_embeddings(feat_map, boxes, (H, W))

        if vlm_embeds.shape[0] > 0:
            vlm_sims = (vlm_embeds @ text_feats.T).detach().cpu().numpy()
        else:
            vlm_sims = np.zeros((0, text_feats.shape[0]))

        vlm_sims_norm = (vlm_sims + 1.0) / 2.0

        detections = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            box_area = (x2 - x1) * (y2 - y1)
            # 过滤掉过大的框
            if box_area / image_area > max_box_ratio:
                continue

            det_prior = float(det_scores[i]) if i < len(det_scores) else 0.0
            combined_scores = self.w_det * det_prior + self.w_vlm * vlm_sims_norm[i]
            best_idx = int(np.argmax(combined_scores))
            best_score = float(combined_scores[best_idx])

            per_class = {}
            for k in range(len(category_prompts)):
                per_class[category_prompts[k]] = {
                    "vlm_sim_raw": float(vlm_sims[i, k]) if vlm_sims.shape[0] > 0 else None,
                    "vlm_sim_0_1": float(vlm_sims_norm[i, k]) if vlm_sims.shape[0] > 0 else None,
                    "combined_score": float(combined_scores[k]),
                }

            det = {
                "box": boxes[i],
                "mask": masks[i] if (masks is not None and i < masks.shape[0]) else None,
                "chosen_category": category_prompts[best_idx],
                "combined_score": best_score,
                "det_score": float(det_scores[i]) if i < len(det_scores) else None,
                "per_class": per_class
            }
            detections.append(det)

        self.visualize(image, detections, save_path=save_path, show_masks=visualize_masks)
        return detections, image


    def visualize(self, image, detections, save_path="fvml_out.jpg", show_masks=True):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(rgb)
        for d in detections:
            box = d["box"].astype(int)
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="yellow", facecolor="none")
            ax.add_patch(rect)
            if show_masks and d["mask"] is not None:
                mask = d["mask"]
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = (1, 0, 0, 0.35)
                ax.imshow(colored_mask)
            label = d["chosen_category"].replace("a photo of a ", "").strip()
            ax.text(x1, max(0, y1-8), f"{label} ({d['combined_score']:.3f})",
                    fontsize=10, color="white", bbox=dict(facecolor="black", alpha=0.6, pad=2))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"[INFO] saved visualization to {save_path}")


if __name__ == "__main__":
    detector = ZeroShotFVLMDetector(
        mask_rcnn_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        clip_model_name="ViT-B/32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        vlm_pool_size=(1,1),
        proj_dim=512,
        w_det=0.5,
        w_vlm=0.5,
    )

    categories = [
        "a photo of a zebra", "a photo of a microwave", "a photo of a red apple",
        "a photo of a rabbit", "a photo of a bear", "a photo of a dog", "a photo of a cat",
        "a photo of a white rabbit", "a photo of a brown bear", "a photo of a pink bear",
        "a photo of a bird", "a photo of a yellow bird"
    ]

    image_path = "D:/animal.png"
    detections, image = detector.detect(image_path, categories, save_path="fvml_result.jpg")

    for i, d in enumerate(detections):
        print(f"\nRoI {i+1}: box={d['box']}, chosen={d['chosen_category']}, combined_score={d['combined_score']:.4f}")
        for cat, info in d["per_class"].items():
            print(f"  {cat}: vlm_raw={info['vlm_sim_raw']:.4f}, vlm_0_1={info['vlm_sim_0_1']:.4f}, combined={info['combined_score']:.4f}")
