import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class AnomalyVisualizer:
    """
    异常检测可视化工具类
    """
    
    def __init__(self, output_dir: str = '../assets/visualizations'):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置颜色映射
        self.cmap = plt.cm.get_cmap('jet')
        self.segmentation_colors = {
            'anomaly': (255, 0, 0, 128),  # 半透明红色
            'normal': (0, 255, 0, 64),   # 半透明绿色
            'background': (0, 0, 0, 0)   # 透明
        }
    
    def visualize_prediction(self, image: Image.Image, pred_mask: np.ndarray, 
                            gt_mask: Optional[np.ndarray] = None, 
                            save_path: Optional[str] = None,
                            threshold: float = 0.5) -> Image.Image:
        """
        可视化预测结果，在原图上叠加预测掩码
        
        Args:
            image: 原始图像
            pred_mask: 预测掩码
            gt_mask: 真实掩码（可选）
            save_path: 保存路径（可选）
            threshold: 掩码阈值
        
        Returns:
            Image: 可视化后的图像
        """
        # 确保图像是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整掩码大小以匹配图像
        pred_mask_resized = self._resize_mask(pred_mask, image.size)
        
        # 创建叠加层
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 将掩码转换为二进制
        binary_mask = pred_mask_resized >= threshold
        
        # 绘制预测掩码
        for y in range(image.height):
            for x in range(image.width):
                if binary_mask[y, x]:
                    draw.point((x, y), fill=self.segmentation_colors['anomaly'])
        
        # 如果提供了真实掩码，绘制边框
        if gt_mask is not None:
            gt_mask_resized = self._resize_mask(gt_mask, image.size)
            binary_gt = gt_mask_resized >= 0.5
            
            # 使用边缘检测找出轮廓
            from skimage import measure
            contours = measure.find_contours(binary_gt, 0.5)
            
            for contour in contours:
                # 转换为图像坐标并绘制
                contour = np.flip(contour, axis=1)  # 转换为(x,y)
                contour = [tuple(point) for point in contour]
                if len(contour) > 2:
                    draw.line(contour + [contour[0]], fill=(0, 0, 255, 255), width=2)
        
        # 合并图像
        result = Image.alpha_composite(image.convert('RGBA'), overlay)
        result = result.convert('RGB')
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            result.save(save_full_path)
            print(f"可视化结果已保存至: {save_full_path}")
        
        return result
    
    def visualize_heatmap(self, pred_mask: np.ndarray, save_path: Optional[str] = None,
                         image: Optional[Image.Image] = None) -> plt.Figure:
        """
        可视化异常分数热图
        
        Args:
            pred_mask: 预测掩码（包含异常分数）
            save_path: 保存路径（可选）
            image: 原始图像（可选，用于叠加显示）
        
        Returns:
            Figure: matplotlib图像对象
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 显示热图
        im = ax.imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
        
        # 如果提供了原始图像，叠加显示
        if image is not None:
            # 调整图像大小
            resized_image = image.resize((pred_mask.shape[1], pred_mask.shape[0]))
            img_array = np.array(resized_image)
            
            # 创建混合图像
            alpha = 0.3  # 透明度
            blended = img_array.copy()
            heatmap_colored = plt.cm.jet(pred_mask)[:, :, :3] * 255
            blended = (1 - alpha) * blended + alpha * heatmap_colored
            blended = blended.astype(np.uint8)
            
            ax.imshow(blended)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('异常分数', fontsize=12)
        
        # 设置标题
        ax.set_title('异常分数热图', fontsize=14)
        ax.axis('off')
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"热图已保存至: {save_full_path}")
            return None
        
        return fig
    
    def visualize_comparison(self, image: Image.Image, pred_mask: np.ndarray, 
                            gt_mask: np.ndarray, save_path: Optional[str] = None,
                            threshold: float = 0.5) -> plt.Figure:
        """
        可视化对比图：原图、预测掩码、真实掩码、对比结果
        
        Args:
            image: 原始图像
            pred_mask: 预测掩码
            gt_mask: 真实掩码
            save_path: 保存路径（可选）
            threshold: 掩码阈值
        
        Returns:
            Figure: matplotlib图像对象
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 调整掩码大小以匹配图像
        pred_mask_resized = self._resize_mask(pred_mask, image.size)
        gt_mask_resized = self._resize_mask(gt_mask, image.size)
        
        # 原图
        axes[0].imshow(image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 预测掩码
        axes[1].imshow(pred_mask_resized, cmap='jet')
        axes[1].set_title('预测掩码')
        axes[1].axis('off')
        
        # 真实掩码
        axes[2].imshow(gt_mask_resized, cmap='gray')
        axes[2].set_title('真实掩码')
        axes[2].axis('off')
        
        # 对比结果
        # 将预测掩码二值化
        binary_pred = (pred_mask_resized >= threshold).astype(np.float32)
        binary_gt = (gt_mask_resized >= 0.5).astype(np.float32)
        
        # 创建对比图像：TP=绿色，FP=红色，FN=黄色，TN=黑色
        comparison = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        comparison[(binary_pred == 1) & (binary_gt == 1)] = [0, 255, 0]      # TP
        comparison[(binary_pred == 1) & (binary_gt == 0)] = [255, 0, 0]      # FP
        comparison[(binary_pred == 0) & (binary_gt == 1)] = [255, 255, 0]    # FN
        comparison[(binary_pred == 0) & (binary_gt == 0)] = [0, 0, 0]        # TN
        
        axes[3].imshow(comparison)
        axes[3].set_title('对比结果')
        axes[3].axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='TP (正确检测到的异常)'),
            Patch(facecolor='red', label='FP (误检为异常)'),
            Patch(facecolor='yellow', label='FN (漏检的异常)'),
            Patch(facecolor='black', label='TN (正确检测到的正常)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"对比图已保存至: {save_full_path}")
            return None
        
        return fig
    
    def visualize_tsne(self, features: np.ndarray, labels: np.ndarray,
                      save_path: Optional[str] = None, 
                      title: str = '特征嵌入的t-SNE可视化') -> plt.Figure:
        """
        可视化特征嵌入的t-SNE结果
        
        Args:
            features: 特征嵌入，形状为 [N, D]
            labels: 标签，形状为 [N]
            save_path: 保存路径（可选）
            title: 图表标题
        
        Returns:
            Figure: matplotlib图像对象
        """
        # 数据标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=122, perplexity=30, n_iter=300)
        embedding = tsne.fit_transform(features_scaled)
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制每个类别的点
        for i, label in enumerate(unique_labels):
            mask = labels == label
            scatter = ax.scatter(
                embedding[mask, 0], 
                embedding[mask, 1],
                c=[colors[i]],
                label=f'类别 {label}',
                s=50,
                alpha=0.7
            )
        
        # 设置图表
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('t-SNE 维度 1', fontsize=12)
        ax.set_ylabel('t-SNE 维度 2', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"t-SNE可视化已保存至: {save_full_path}")
            return None
        
        return fig
    
    def visualize_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                           save_path: Optional[str] = None,
                           title: str = 'ROC曲线') -> plt.Figure:
        """
        可视化ROC曲线
        
        Args:
            y_true: 真实标签
            y_score: 预测分数
            save_path: 保存路径（可选）
            title: 图表标题
        
        Returns:
            Figure: matplotlib图像对象
        """
        from sklearn.metrics import roc_curve, auc
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color='blue', lw=2,
                label=f'ROC曲线 (面积 = {roc_auc:.3f})')
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        
        # 设置图表
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假正例率', fontsize=12)
        ax.set_ylabel('真正例率', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"ROC曲线已保存至: {save_full_path}")
            return None
        
        return fig
    
    def visualize_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                          save_path: Optional[str] = None,
                          title: str = 'PR曲线') -> plt.Figure:
        """
        可视化PR曲线
        
        Args:
            y_true: 真实标签
            y_score: 预测分数
            save_path: 保存路径（可选）
            title: 图表标题
        
        Returns:
            Figure: matplotlib图像对象
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制PR曲线
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR曲线 (AP = {average_precision:.3f})')
        
        # 设置图表
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('召回率', fontsize=12)
        ax.set_ylabel('精确率', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"PR曲线已保存至: {save_full_path}")
            return None
        
        return fig
    
    def visualize_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                                    save_path: Optional[str] = None,
                                    title: str = '不同方法的指标对比') -> plt.Figure:
        """
        可视化不同方法的指标对比
        
        Args:
            metrics_dict: 指标字典，格式为 {method: {metric1: value1, metric2: value2, ...}}
            save_path: 保存路径（可选）
            title: 图表标题
        
        Returns:
            Figure: matplotlib图像对象
        """
        methods = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())
        
        # 创建子图
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        
        if num_metrics == 1:
            axes = [axes]  # 确保axes是列表
        
        # 绘制每个指标的条形图
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[method][metric] for method in methods]
            bars = axes[i].bar(methods, values, color=colors)
            
            # 在条形图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
            
            axes[i].set_title(f'{metric}')
            axes[i].set_ylim(0, 1.1)  # 设置y轴范围
            axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"指标对比图已保存至: {save_full_path}")
            return None
        
        return fig
    
    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整掩码大小
        
        Args:
            mask: 原始掩码
            target_size: 目标大小 (width, height)
        
        Returns:
            np.ndarray: 调整后的掩码
        """
        from skimage.transform import resize
        return resize(mask, (target_size[1], target_size[0]), order=1, preserve_range=True)
    
    def visualize_batch_results(self, images: List[Image.Image], 
                               pred_masks: List[np.ndarray],
                               gt_masks: Optional[List[np.ndarray]] = None,
                               save_path: Optional[str] = None,
                               threshold: float = 0.5, 
                               max_images: int = 4) -> plt.Figure:
        """
        可视化批量结果
        
        Args:
            images: 原始图像列表
            pred_masks: 预测掩码列表
            gt_masks: 真实掩码列表（可选）
            save_path: 保存路径（可选）
            threshold: 掩码阈值
            max_images: 最大显示图像数量
        
        Returns:
            Figure: matplotlib图像对象
        """
        # 限制图像数量
        num_images = min(len(images), max_images)
        
        # 确定子图数量
        cols = 3 if gt_masks else 2
        fig, axes = plt.subplots(num_images, cols, figsize=(cols * 5, num_images * 4))
        
        if num_images == 1:
            axes = [axes]  # 确保axes是二维列表
        
        for i in range(num_images):
            # 原图
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('原始图像')
            axes[i, 0].axis('off')
            
            # 预测结果
            pred_result = self.visualize_prediction(
                images[i], pred_masks[i], threshold=threshold
            )
            axes[i, 1].imshow(pred_result)
            axes[i, 1].set_title('预测结果')
            axes[i, 1].axis('off')
            
            # 如果提供了真实掩码
            if gt_masks:
                gt_result = self.visualize_prediction(
                    images[i], gt_masks[i], threshold=threshold
                )
                axes[i, 2].imshow(gt_result)
                axes[i, 2].set_title('真实掩码')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            save_full_path = os.path.join(self.output_dir, save_path)
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"批量可视化结果已保存至: {save_full_path}")
            return None
        
        return fig

def create_visualizer(output_dir: str = '../assets/visualizations') -> AnomalyVisualizer:
    """
    创建可视化工具实例
    
    Args:
        output_dir: 输出目录
    
    Returns:
        AnomalyVisualizer: 可视化工具实例
    """
    return AnomalyVisualizer(output_dir)


def save_detection_result(image: Image.Image, result: Dict, save_dir: str,
                         image_name: str = 'result') -> str:
    """
    保存检测结果
    
    Args:
        image: 原始图像
        result: 检测结果
        save_dir: 保存目录
        image_name: 图像名称
    
    Returns:
        str: 保存的图像路径
    """
    visualizer = create_visualizer(save_dir)
    
    # 获取预测掩码（选择分数最高的）
    if len(result['masks']) > 0:
        best_mask_idx = result['scores'].argmax()
        pred_mask = result['masks'][best_mask_idx].cpu().numpy()
    else:
        pred_mask = np.zeros((512, 512))
    
    # 保存可视化结果
    save_path = f"{image_name}.png"
    visualizer.visualize_prediction(
        image=image,
        pred_mask=pred_mask,
        save_path=save_path,
        threshold=0.5
    )
    
    return os.path.join(save_dir, save_path)
