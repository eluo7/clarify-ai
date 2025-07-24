"""
单实例解释示例
演示 force_plot 和 waterfall_plot 的使用
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clarify_ai import ModelExplainer

def create_sample_data():
    """创建示例数据"""
    X, y = make_classification(
        n_samples=200, 
        n_features=8, 
        n_informative=5, 
        n_redundant=1,
        random_state=42
    )
    
    feature_names = [
        '收入水平', '年龄', '工作经验', 
        '信用评分', '负债率', '教育程度',
        '婚姻状态', '居住稳定性'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 标准化数据使其更有意义
    X_df['收入水平'] = (X_df['收入水平'] - X_df['收入水平'].min()) * 10000 + 30000
    X_df['年龄'] = (X_df['年龄'] - X_df['年龄'].min()) * 30 + 25
    X_df['工作经验'] = np.abs(X_df['工作经验']) * 10
    X_df['信用评分'] = (X_df['信用评分'] - X_df['信用评分'].min()) * 200 + 500
    X_df['负债率'] = np.abs(X_df['负债率']) * 0.5
    X_df['教育程度'] = np.round(np.abs(X_df['教育程度']) * 3 + 1)
    X_df['婚姻状态'] = np.round(np.abs(X_df['婚姻状态']))
    X_df['居住稳定性'] = np.abs(X_df['居住稳定性']) * 5
    
    return X_df, y

def main():
    print("=== 单实例解释示例 ===")
    
    # 创建数据
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"模型准确率: {model.score(X_test, y_test):.3f}")
    
    # 初始化解释器
    explainer = ModelExplainer(
        model=model,
        X_train=X_train,
        output_dir="reports/single_instance"
    )
    
    # 选择要解释的实例
    instance_indices = [0, 5, 10]
    
    print(f"\n=== 分析 {len(instance_indices)} 个实例 ===")
    
    for idx in instance_indices:
        print(f"\n--- 实例 #{idx} ---")
        
        # 显示实例信息
        instance = X_test.iloc[idx]
        prediction = model.predict_proba([instance])[0][1]
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        
        print(f"预测概率: {prediction:.3f}")
        print(f"实际标签: {actual}")
        print(f"预测结果: {'通过' if prediction > 0.5 else '拒绝'}")
        
        print("\n特征值:")
        for feature, value in instance.items():
            print(f"  {feature}: {value:.2f}")
        
        # 获取局部解释
        explanation = explainer.explain_prediction(idx, X_test)
        
        print(f"\n特征贡献度 (SHAP值):")
        contributions = explanation['feature_contributions']
        
        # 按绝对值排序显示
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, contrib in sorted_contrib:
            direction = "正向" if contrib > 0 else "负向"
            print(f"  {feature}: {contrib:.4f} ({direction})")
        
        # 生成 Waterfall Plot
        print(f"\n生成 Waterfall Plot...")
        waterfall_path = f"reports/single_instance/waterfall_instance_{idx}.png"
        explainer.plot_prediction_explanation(idx, X_test, waterfall_path)
        print(f"Waterfall图已保存: {waterfall_path}")
        
        # 生成自然语言解释
        nl_explanation = explainer.get_natural_language_explanation(
            local_explanation=explanation
        )
        
        if 'local' in nl_explanation:
            print(f"\n自然语言解释:")
            print(nl_explanation['local'])
        
        print("-" * 50)
    
    # 生成完整报告
    print(f"\n=== 生成完整报告 ===")
    report_path = explainer.generate_report(
        X_test=X_test,
        instances_to_explain=instance_indices,
        model_type="随机森林分类器",
        report_name="single_instance_analysis"
    )
    
    print(f"完整报告已生成: {report_path}")
    
    # 演示 Force Plot（如果支持的话）
    print(f"\n=== Force Plot 演示 ===")
    try:
        # Force plot 通常在 Jupyter notebook 中效果更好
        force_plot = explainer.shap_explainer.create_force_plot(0, X_test)
        print("Force plot 已创建（在 Jupyter notebook 中查看效果更佳）")
    except Exception as e:
        print(f"Force plot 创建失败: {e}")

if __name__ == "__main__":
    main()