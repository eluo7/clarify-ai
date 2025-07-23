"""
信贷评分模型解释示例
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
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
def create_credit_dataset(n_samples=1000):
    """创建模拟的信贷数据集"""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2,
        random_state=42
    )
    
    # 创建有意义的特征名称
    feature_names = [
        '月收入', '年龄', '工作年限', 
        '负债比例', '信用记录长度', 
        '历史逾期次数', '贷款金额', 
        '房产价值', '教育水平', '婚姻状态'
    ]
    
    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 调整数据范围使其更符合实际
    X_df['月收入'] = X_df['月收入'] * 5000 + 8000  # 3000-13000
    X_df['年龄'] = X_df['年龄'] * 20 + 25  # 20-45岁
    X_df['工作年限'] = X_df['工作年限'] * 8 + 1  # 1-9年
    X_df['负债比例'] = X_df['负债比例'] * 0.4 + 0.2  # 20%-60%
    X_df['信用记录长度'] = X_df['信用记录长度'] * 5 + 1  # 1-6年
    X_df['历史逾期次数'] = np.round(X_df['历史逾期次数'] * 3)  # 0-3次
    X_df['贷款金额'] = X_df['贷款金额'] * 200000 + 100000  # 10-30万
    X_df['房产价值'] = X_df['房产价值'] * 1000000 + 500000  # 50-150万
    X_df['教育水平'] = np.round(X_df['教育水平'] * 3 + 1)  # 1-4 (高中、大专、本科、研究生)
    X_df['婚姻状态'] = np.round(X_df['婚姻状态'])  # 0-1 (未婚、已婚)
    
    return X_df, y

# 特征描述字典
feature_descriptions = {
    '月收入': '申请人月收入(元)',
    '年龄': '申请人年龄(岁)',
    '工作年限': '当前工作持续时间(年)',
    '负债比例': '月还款总额/月收入',
    '信用记录长度': '信用历史长度(年)',
    '历史逾期次数': '过去12个月内的逾期次数',
    '贷款金额': '申请贷款金额(元)',
    '房产价值': '申请人房产估值(元)',
    '教育水平': '教育程度(1-高中,2-大专,3-本科,4-研究生)',
    '婚姻状态': '婚姻状况(0-未婚,1-已婚)'
}

def main():
    # 创建数据集
    X, y = create_credit_dataset()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 初始化解释器
    explainer = ModelExplainer(
        model=model,
        X_train=X_train,
        feature_descriptions=feature_descriptions,
        output_dir="reports/credit_scoring"
    )
    
    # 生成报告
    report_path = explainer.generate_report(
        X_test=X_test,
        instances_to_explain=[0, 5, 10],  # 解释前3个测试样本
        model_type="随机森林分类器 (信贷评分)",
        report_name="credit_scoring_explanation"
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