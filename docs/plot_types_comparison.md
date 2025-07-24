# SHAP 可视化类型对比

## 概述

SHAP 提供了多种可视化方法来解释机器学习模型的预测结果。主要分为**全局解释**和**局部解释**两类。

## 全局解释（针对整个模型）

### 1. Summary Plot
- **用途**：显示所有特征的整体重要性
- **输入**：整个测试集
- **特点**：
  - 显示每个特征对所有样本的影响分布
  - 颜色表示特征值的高低
  - 横轴表示SHAP值（影响程度）

```python
# 使用方法
explainer.plot_global_importance(X_test)
# 或
explainer.shap_explainer.create_summary_plot(X_test)
```

### 2. Feature Importance Bar Plot
- **用途**：显示特征重要性排序
- **输入**：整个测试集
- **特点**：简单的条形图显示平均绝对SHAP值

## 局部解释（针对单个实例）

### 1. Waterfall Plot ⭐
- **用途**：显示单个预测中各特征的**累积贡献**
- **输入**：单个实例索引
- **特点**：
  - 从基准值开始
  - 逐步显示每个特征如何影响最终预测
  - 瀑布式累积效应
  - 最终到达预测值

```python
# 使用方法
explainer.plot_prediction_explanation(instance_idx=0, X_test=X_test)
# 或
explainer.shap_explainer.create_waterfall_plot(0, X_test)
```

**Waterfall Plot 示例解读**：
```
基准值: 0.3
+ 收入水平: +0.15  → 0.45
+ 信用评分: +0.10  → 0.55
- 负债率: -0.05     → 0.50
+ 年龄: +0.08       → 0.58
= 最终预测: 0.58
```

### 2. Force Plot ⭐
- **用途**：显示单个预测中特征的**推拉效应**
- **输入**：单个实例索引
- **特点**：
  - 交互式可视化（在Jupyter中效果最佳）
  - 红色箭头：推向正类的特征
  - 蓝色箭头：推向负类的特征
  - 箭头长度表示影响强度

```python
# 使用方法
force_plot = explainer.shap_explainer.create_force_plot(0, X_test)
```

### 3. Decision Plot
- **用途**：显示决策路径
- **输入**：单个或多个实例
- **特点**：显示从平均预测到最终预测的路径

## 使用场景对比

| 图表类型 | 适用场景 | 优势 | 局限性 |
|---------|---------|------|--------|
| **Summary Plot** | 模型整体分析 | 全局视角，特征重要性排序 | 不显示具体实例 |
| **Waterfall Plot** | 单个预测解释 | 清晰的累积效应，易于理解 | 只能看单个实例 |
| **Force Plot** | 单个预测解释 | 交互式，直观的推拉效应 | 需要Jupyter环境 |

## 在 Clarify AI 中的实现

### 自动选择
```python
# 系统会根据输入自动选择合适的图表类型
explainer = ModelExplainer(model, X_train)

# 全局解释 - 自动使用 Summary Plot
explainer.plot_global_importance(X_test)

# 局部解释 - 自动使用 Waterfall Plot
explainer.plot_prediction_explanation(instance_idx=0, X_test=X_test)
```

### 手动选择
```python
# 直接调用特定图表类型
shap_explainer = explainer.shap_explainer

# Waterfall Plot
shap_explainer.create_waterfall_plot(0, X_test, "waterfall.png")

# Force Plot
force_plot = shap_explainer.create_force_plot(0, X_test)

# Summary Plot
shap_explainer.create_summary_plot(X_test, "summary.png")
```

## 最佳实践

1. **全局分析**：先用 Summary Plot 了解模型整体行为
2. **局部解释**：用 Waterfall Plot 解释具体预测
3. **交互探索**：在 Jupyter 中用 Force Plot 进行交互式分析
4. **报告生成**：Clarify AI 自动选择最适合的图表类型

## 注意事项

- **Force Plot** 在静态环境中可能显示效果不佳
- **Waterfall Plot** 更适合生成静态报告
- 所有局部解释都需要指定具体的实例索引
- 确保实例索引在有效范围内（0 到 len(X_test)-1）