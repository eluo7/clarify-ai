# Clarify AI

一个基于SHAP的机器学习模型可解释性Agent，专注于将复杂的模型解释转化为业务友好的报告。

## 项目特点

- 🔍 **全局解释**：分析整体模型行为和特征重要性
- 🎯 **局部解释**：解释单个预测结果的决策过程
- 📊 **可视化报告**：生成直观的HTML报告，包含图表和解释
- 💬 **自然语言解释**：将技术结果转化为业务友好的语言
- 🚀 **易于集成**：简单API，可与各种ML模型配合使用

## 安装

```bash
git clone https://github.com/yourusername/clarify-ai.git
cd clarify-ai
pip install -r requirements.txt
```

## 快速开始

```python
from clarify_ai import ModelExplainer

# 初始化解释器
explainer = ModelExplainer(model, X_train)

# 生成解释报告
report_path = explainer.generate_report(
    X_test=X_test,
    report_name="my_model_explanation"
)

print(f"报告已生成: {report_path}")
```

## 核心功能

### 1. 全局模型解释

分析整个模型的行为和特征重要性，帮助理解模型的整体决策逻辑。

```python
# 获取全局特征重要性
importance = explainer.get_global_importance(X_test)

# 生成全局解释图
explainer.plot_global_importance(X_test)
```

### 2. 局部预测解释

解释单个预测结果，分析各特征对该预测的贡献。

```python
# 解释单个预测
explanation = explainer.explain_prediction(instance_id=42, X_test=X_test)

# 可视化单个预测
explainer.plot_prediction_explanation(instance_id=42, X_test=X_test)
```

### 3. 自然语言解释

将技术解释转化为业务友好的自然语言描述。

```python
# 获取自然语言解释
nl_explanation = explainer.get_natural_language_explanation(
    global_importance=importance,
    local_explanation=explanation
)
```

### 4. 解释性报告生成

生成包含全局和局部解释的综合HTML报告。

```python
# 生成完整报告
report_path = explainer.generate_report(
    X_test=X_test,
    instances_to_explain=[42, 56, 78],  # 选择要解释的样本
    model_type="随机森林分类器"
)
```

## 项目结构

```
clarify-ai/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── shap_explainer.py      # SHAP解释器核心
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── natural_language_explainer.py  # 自然语言解释
│   ├── report/
│   │   ├── __init__.py
│   │   └── report_generator.py    # 报告生成器
│   └── __init__.py
├── examples/
│   ├── credit_scoring_example.py  # 信贷评分示例
│   └── customer_churn_example.py  # 客户流失示例
├── tests/
│   └── test_explainer.py
├── requirements.txt
└── README.md
```

## 使用场景

- **风控模型解释**：解释信贷审批决策，提高合规性
- **客户流失分析**：理解客户流失预测的关键因素
- **医疗诊断支持**：解释医疗AI模型的诊断依据
- **营销模型优化**：分析营销效果预测的关键驱动因素

## 贡献

欢迎提交问题和拉取请求！

## 许可

MIT