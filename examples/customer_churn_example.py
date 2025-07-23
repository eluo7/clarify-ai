"""
客户流失预测模型解释示例
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clarify_ai import ModelExplainer

# 设置字体支持
try:
    plt.rcParams['font.sans-serif'] = ['Arial']  # 使用通用字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass

# 创建示例数据集
def create_churn_dataset(n_samples=1000):
    """创建模拟的客户流失数据集"""
    np.random.seed(42)
    
    # 创建特征
    tenure = np.random.randint(1, 73, n_samples)  # 在网时长
    monthly_charges = np.random.uniform(20, 120, n_samples)  # 月费
    total_charges = monthly_charges * tenure * (0.9 + 0.2 * np.random.random(n_samples))  # 总费用
    
    contract = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])  # 合约类型
    online_security = np.random.choice([0, 1], n_samples)  # 是否有在线安全服务
    tech_support = np.random.choice([0, 1], n_samples)  # 是否有技术支持
    streaming_tv = np.random.choice([0, 1], n_samples)  # 是否有流媒体电视
    streaming_movies = np.random.choice([0, 1], n_samples)  # 是否有流媒体电影
    
    payment_method = np.random.choice([0, 1, 2, 3], n_samples)  # 支付方式
    paperless_billing = np.random.choice([0, 1], n_samples)  # 是否无纸化账单
    
    # 创建DataFrame
    data = pd.DataFrame({
        '在网时长': tenure,
        '月费': monthly_charges,
        '总费用': total_charges,
        '合约类型': contract,
        '在线安全': online_security,
        '技术支持': tech_support,
        '流媒体电视': streaming_tv,
        '流媒体电影': streaming_movies,
        '支付方式': payment_method,
        '无纸化账单': paperless_billing
    })
    
    # 生成目标变量（流失概率与在网时长负相关，与月费正相关）
    churn_prob = 1 / (1 + np.exp(0.05 * tenure - 0.03 * monthly_charges + 
                                 0.5 * online_security + 0.5 * tech_support - 
                                 0.2 * contract + np.random.normal(0, 0.5, n_samples)))
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return data, y

# 特征描述字典
feature_descriptions = {
    '在网时长': '客户使用服务的月数',
    '月费': '客户每月支付的费用(元)',
    '总费用': '客户迄今为止支付的总费用(元)',
    '合约类型': '客户的合约类型(0-月付,1-一年,2-两年)',
    '在线安全': '客户是否有在线安全服务(0-无,1-有)',
    '技术支持': '客户是否有技术支持服务(0-无,1-有)',
    '流媒体电视': '客户是否有流媒体电视服务(0-无,1-有)',
    '流媒体电影': '客户是否有流媒体电影服务(0-无,1-有)',
    '支付方式': '客户的支付方式(0-电子支票,1-邮寄支票,2-银行转账,3-信用卡)',
    '无纸化账单': '客户是否使用无纸化账单(0-否,1-是)'
}

def main():
    # 创建数据集
    X, y = create_churn_dataset()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化数值特征
    scaler = StandardScaler()
    num_features = ['在网时长', '月费', '总费用']
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    
    # 训练模型
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 初始化解释器
    explainer = ModelExplainer(
        model=model,
        X_train=X_train,
        feature_descriptions=feature_descriptions,
        output_dir="reports/customer_churn"
    )
    
    # 生成报告
    report_path = explainer.generate_report(
        X_test=X_test,
        instances_to_explain=[0, 1, 2],  # 解释前3个测试样本
        model_type="梯度提升树 (客户流失预测)",
        report_name="customer_churn_explanation"
    )
    
    print(f"报告已生成: {report_path}")
    print(f"准确率: {model.score(X_test, y_test):.2f}")
    
    # 单独获取全局特征重要性
    importance = explainer.get_global_importance(X_test)
    print("\n全局特征重要性:")
    for feature, score in importance.items():
        print(f"{feature}: {score:.4f}")
    
    # 解释单个预测
    instance_id = 0
    explanation = explainer.explain_prediction(instance_id, X_test)
    
    # 获取自然语言解释
    nl_explanation = explainer.get_natural_language_explanation(
        global_importance=importance,
        local_explanation=explanation
    )
    
    print("\n局部预测解释:")
    print(nl_explanation.get('local', ''))

if __name__ == "__main__":
    main()