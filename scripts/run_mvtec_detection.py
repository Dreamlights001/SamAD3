# MVTec数据集瑕疵检测示例脚本
import argparse
from defect_detector import DefectDetector

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在MVTec数据集上运行零样本工业瑕疵检测")
    parser.add_argument("--category", type=str, default="bottle",
                       help="MVTec数据集的类别，如bottle、cable、capsule等")
    parser.add_argument("--prompt", type=str, default=None,
                       help="自定义提示词，用于引导模型检测特定类型的瑕疵")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print(f"在MVTec数据集的{args.category}类别上运行瑕疵检测...")
    
    # 创建瑕疵检测器
    detector = DefectDetector()
    
    # 运行检测
    metrics, accuracy = detector.detect(
        dataset_name="mvtec",
        category=args.category,
        custom_prompt=args.prompt
    )
    
    print("检测完成!")
    if metrics:
        print(f"平均IoU: {metrics['iou']:.4f}")
        print(f"平均精确率: {metrics['precision']:.4f}")
        print(f"平均召回率: {metrics['recall']:.4f}")
        print(f"平均F1分数: {metrics['f1_score']:.4f}")
        print(f"检测准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()
