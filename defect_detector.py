# 零样本工业瑕疵检测主程序
import os
import time
import numpy as np
import cv2
import datetime
import json
from tqdm import tqdm
from PIL import Image
from models.sam3_model import SAM3Model
from data_loaders.data_pipeline import create_data_pipeline
from utils.metrics import evaluate_prediction, evaluate_batch_predictions, calculate_image_level_auroc, calculate_pixel_level_auroc, calculate_ap, calculate_aupro
from utils.image_processing import visualize_prediction, create_visualization_grid, save_visualization
from config import OUTPUT_DIR, MODEL_CONFIG as model_config

class DefectDetector:
    """零样本工业瑕疵检测器"""
    def __init__(self):
        """初始化瑕疵检测器"""
        # 加载模型
        self.model = SAM3Model()
        
        # 创建结果保存目录
        self.results_dir = os.path.join(OUTPUT_DIR, f"results_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def detect(self, dataset_name, category=None, custom_prompt=None):
        """在指定数据集上运行检测"""
        # 根据数据集名称生成默认提示词
        if custom_prompt is None:
            if dataset_name.lower() == "mvtec":
                custom_prompt = f"defect in {category} product, scratch, damage, imperfection"
            else:
                custom_prompt = "defect, scratch, damage, imperfection"
        
        # 创建数据管道
        data_pipeline = create_data_pipeline(dataset_name, category)
            
        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"results_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = output_dir
        
        print(f"开始对{dataset_name}数据集的{category}类别进行瑕疵检测...")
        print(f"使用提示词: {custom_prompt}")
        
        # 初始化评估指标
        all_metrics = {
            'iou': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'mask': [],  # 存储真实掩码
            'pred_mask': []  # 存储预测掩码
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 处理每个图像
        for idx, (image, mask, image_path) in enumerate(tqdm(data_pipeline, desc="Processing images")):
            # 进行预测
            prediction = self.model.predict_defect(image, custom_prompt)
            
            # 评估预测结果
            metrics = evaluate_prediction(prediction["mask"], mask)
            
            # 保存评估指标
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # 保存掩码用于后续高级评估
            all_metrics['mask'].append(mask)
            all_metrics['pred_mask'].append(prediction["mask"])
            
            # 可视化并保存结果
            vis_image = visualize_prediction(image, prediction["mask"], mask)
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
            vis_path = os.path.join(output_dir, "visualizations", f"result_{idx:04d}.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # 计算平均指标
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = sum(values) / len(values) if values else 0
        
        # 计算高级评估指标
        if all_metrics['mask']:
            # 计算像素级和图像级指标
            pixel_auroc = calculate_pixel_level_auroc(all_metrics['mask'], all_metrics['pred_mask'])
            image_auroc = calculate_image_level_auroc(all_metrics['mask'], all_metrics['pred_mask'])
            ap = calculate_ap(all_metrics['mask'], all_metrics['pred_mask'])
            aupro = calculate_aupro(all_metrics['mask'], all_metrics['pred_mask'])
            
            # 更新评估指标字典
            avg_metrics.update({
                'pixel_auroc': pixel_auroc,
                'image_auroc': image_auroc,
                'ap': ap,
                'aupro': aupro
            })
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 保存评估报告为文本文件
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"类别: {category}\n")
            f.write(f"提示词: {custom_prompt}\n")
            f.write(f"总图像数: {len(all_metrics['iou'])}\n")
            f.write(f"总耗时: {total_time:.2f}秒\n")
            f.write(f"平均耗时: {(total_time/len(all_metrics['iou'])):.4f}秒/图像\n\n")
            f.write("评估指标:\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # 保存评估指标为JSON文件以便后续分析
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            # 转换numpy类型为Python原生类型
            serializable_metrics = {}
            for k, v in avg_metrics.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    serializable_metrics[k] = v.item()
                else:
                    serializable_metrics[k] = v
            
            # 添加基本信息
            serializable_metrics.update({
                'dataset': dataset_name,
                'category': category,
                'prompt': custom_prompt,
                'total_images': len(all_metrics['iou']),
                'total_time': total_time,
                'avg_time_per_image': total_time / len(all_metrics['iou']) if all_metrics['iou'] else 0
            })
            
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        # 创建可视化汇总
        viz_dir = os.path.join(output_dir, 'visualizations_summary')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 选择前5个样本创建可视化对比
        sample_indices = min(5, len(all_metrics['iou']))
        for i in range(sample_indices):
            image, true_mask, _ = data_pipeline[i]
            pred_mask = all_metrics['pred_mask'][i] if 'pred_mask' in all_metrics else None
            
            if pred_mask is not None:
                fig = create_visualization_grid(image, pred_mask, true_mask)
                viz_path = os.path.join(viz_dir, f'sample_{i}_comparison.png')
                save_visualization(fig, viz_path)
        
        # 输出统计信息
        print(f"检测完成！")
        print(f"结果保存在: {output_dir}")
        print(f"平均IoU: {avg_metrics['iou']:.4f}")
        print(f"平均精确率: {avg_metrics['precision']:.4f}")
        print(f"平均召回率: {avg_metrics['recall']:.4f}")
        print(f"平均F1分数: {avg_metrics['f1_score']:.4f}")
        
        return {
            'output_dir': output_dir,
            'metrics': avg_metrics,
            'total_images': len(all_metrics['iou']),
            'total_time': total_time
        }
    
    def detect_single_image(self, image_path, custom_prompt=None, save_visualization=True):
        """对单张图像进行瑕疵检测"""
        from PIL import Image
        
        # 设置默认提示词
        if custom_prompt is None:
            custom_prompt = "defect, scratch, damage, imperfection, anomaly"
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 记录推理时间
        start_time = time.time()
        
        # 进行预测
        prediction = self.model.predict_defect(image, custom_prompt)
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        # 创建结果保存目录
        if save_visualization:
            sample_dir = os.path.join(self.results_dir, "single_images")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 生成保存路径
            image_name = os.path.basename(image_path)
            name, ext = os.path.splitext(image_name)
            
            # 创建多种可视化
            # 1. 基本叠加图
            basic_viz = visualize_prediction(image, prediction["mask"])
            basic_viz_path = os.path.join(sample_dir, f'{name}_basic_viz{ext}')
            cv2.imwrite(basic_viz_path, cv2.cvtColor(basic_viz, cv2.COLOR_RGB2BGR))
            
            # 2. 高级可视化网格
            fig = create_visualization_grid(image, prediction["mask"])
            grid_viz_path = os.path.join(sample_dir, f'{name}_grid_viz.png')
            save_visualization(fig, grid_viz_path)
            
            print(f"可视化结果已保存到: {basic_viz_path}, {grid_viz_path}")
            
            visualization_paths = {
                'basic': basic_viz_path,
                'grid': grid_viz_path
            }
        
        # 判断是否有异常
        has_anomaly = (prediction["scores"] > model_config['confidence_threshold']).any() if isinstance(prediction["scores"], np.ndarray) else np.any(prediction["mask"])
        
        print(f"检测完成，耗时: {inference_time:.4f}秒")
        print(f"检测到异常: {has_anomaly}")
        
        result = {
            'has_anomaly': has_anomaly,
            'mask': prediction["mask"],
            'scores': prediction["scores"],
            'inference_time': inference_time,
            'prompt_used': custom_prompt
        }
        
        if save_visualization:
            result['visualization_paths'] = visualization_paths
        
        return result

if __name__ == "__main__":
    # 示例用法
    detector = DefectDetector()
    
    # 可以根据需要取消注释以下行并修改参数
    # detector.detect("mvtec", "bottle")
    # detector.detect_single_image("path/to/your/image.jpg")
