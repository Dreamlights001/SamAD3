# 单张图像瑕疵检测示例脚本
import argparse
import os
from defect_detector import DefectDetector

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="对单张图像进行零样本工业瑕疵检测")
    parser.add_argument("--image_path", type=str, required=True,
                       help="输入图像的路径")
    parser.add_argument("--prompt", type=str, default=None,
                       help="自定义提示词，用于引导模型检测特定类型的瑕疵")
    parser.add_argument("--no_visualization", action="store_true",
                       help="不保存可视化结果")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 检查图像路径是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        return
    
    print(f"对图像 {args.image_path} 进行瑕疵检测...")
    
    # 创建瑕疵检测器
    detector = DefectDetector()
    
    # 运行检测
    result = detector.detect_single_image(
        image_path=args.image_path,
        custom_prompt=args.prompt,
        save_visualization=not args.no_visualization
    )
    
    print("检测结果:")
    print(f"是否检测到异常: {result['has_anomaly']}")
    print(f"推理时间: {result['inference_time']:.4f}秒")
    if result['scores'].size > 0:
        print(f"最高置信度分数: {result['scores'].max():.4f}")

if __name__ == "__main__":
    main()
