"""
报告生成器模块
生成包含全局和局部解释的可解释性报告
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from jinja2 import Template
import base64
from io import BytesIO
import matplotlib.pyplot as plt


class ExplanationReportGenerator:
    """可解释性报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML报告模板
        self.html_template = """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>模型可解释性报告</title>
                <style>
                    body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }
                    .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
                    .section { margin-bottom: 30px; }
                    .section h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; }
                    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
                    .feature-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
                    .importance-bar { height: 20px; background: linear-gradient(90deg, #3498db, #2980b9); border-radius: 10px; }
                    .chart-container { text-align: center; margin: 20px 0; }
                    .explanation-text { background: #e8f4fd; padding: 15px; border-radius: 8px; margin: 10px 0; }
                    .risk-high { color: #e74c3c; font-weight: bold; }
                    .risk-medium { color: #f39c12; font-weight: bold; }
                    .risk-low { color: #27ae60; font-weight: bold; }
                    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🔍 机器学习模型可解释性报告</h1>
                    <p>生成时间: {{ timestamp }}</p>
                    <p>模型类型: {{ model_type }}</p>
                </div>

                <div class="section">
                    <h2>📊 模型整体表现</h2>
                    <div class="metric-card">
                        <p><strong>样本总数:</strong> {{ total_samples }}</p>
                        <p><strong>特征数量:</strong> {{ feature_count }}</p>
                        {% if model_performance %}
                        <p><strong>模型准确率:</strong> {{ model_performance.accuracy }}%</p>
                        {% endif %}
                    </div>
                </div>

                <div class="section">
                    <h2>🌍 全局特征重要性分析</h2>
                    <div class="explanation-text">
                        <p>{{ global_explanation }}</p>
                    </div>
                    
                    {% for feature, importance in global_importance.items() %}
                    <div class="feature-item">
                        <span>{{ feature }}</span>
                        <div style="width: 200px;">
                            {% set importance_value = importance if importance is number else 0 %}
                            <div class="importance-bar" style="width: {{ (importance_value * 100) }}%;"></div>
                            <small>{{ "%.3f"|format(importance_value) }}</small>
                        </div>
                    </div>
                    {% endfor %}
                    
                    {% if global_chart %}
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ global_chart }}" alt="全局特征重要性图" style="max-width: 100%;">
                    </div>
                    {% endif %}
                </div>

                {% if local_explanations %}
                <div class="section">
                    <h2>🎯 局部解释分析</h2>
                    {% for explanation in local_explanations %}
                    <div class="metric-card">
                        <h3>样本 #{{ explanation.instance_id }}</h3>
                        <p><strong>预测结果:</strong> 
                            <span class="{% if explanation.prediction_risk == 'high' %}risk-high{% elif explanation.prediction_risk == 'medium' %}risk-medium{% else %}risk-low{% endif %}">
                                {{ explanation.prediction_label }}
                            </span>
                        </p>
                        
                        <div class="explanation-text">
                            <p>{{ explanation.natural_language_explanation }}</p>
                        </div>
                        
                        <h4>特征贡献度:</h4>
                        <table>
                            <tr><th>特征</th><th>特征值</th><th>贡献度</th><th>影响</th></tr>
                            {% for feature, contrib in explanation.feature_contributions.items() %}
                            <tr>
                                <td>{{ feature }}</td>
                                <td>{{ explanation.feature_values[feature] }}</td>
                                {% set contrib_value = contrib if contrib is number else 0 %}
                                <td>{{ "%.3f"|format(contrib_value) }}</td>
                                <td>{{ "正向" if contrib_value > 0 else "负向" }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                        
                        {% if explanation.chart %}
                        <div class="chart-container">
                            <img src="data:image/png;base64,{{ explanation.chart }}" alt="局部解释图" style="max-width: 100%;">
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                <div class="section">
                    <h2>💡 业务建议</h2>
                    <div class="explanation-text">
                        {{ business_recommendations }}
                    </div>
                </div>

                <div class="section">
                    <h2>⚠️ 注意事项</h2>
                    <ul>
                        <li>本报告基于SHAP (SHapley Additive exPlanations) 方法生成</li>
                        <li>特征重要性反映了特征对模型预测的平均影响程度</li>
                        <li>局部解释仅针对特定样本，不代表全局模式</li>
                        <li>模型解释结果应结合业务知识进行判断</li>
                    </ul>
                </div>
            </body>
            </html>
        """
    
    def generate_report(self, 
                       shap_explainer,
                       X_test: pd.DataFrame,
                       global_importance: Dict[str, float],
                       local_explanations: Optional[List[Dict]] = None,
                       natural_language_explanations: Optional[Dict] = None,
                       model_type: str = "Unknown",
                       report_name: str = None) -> str:
        """
        生成完整的可解释性报告
        
        Args:
            shap_explainer: SHAP解释器实例
            X_test: 测试数据
            global_importance: 全局特征重要性
            local_explanations: 局部解释列表
            natural_language_explanations: 自然语言解释
            model_type: 模型类型
            report_name: 报告名称
            
        Returns:
            str: 生成的报告文件路径
        """
        if report_name is None:
            report_name = f"explanation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 生成全局图表
        global_chart_path = os.path.join(self.output_dir, f"{report_name}_global.png")
        shap_explainer.create_summary_plot(X_test, global_chart_path)
        global_chart_b64 = self._image_to_base64(global_chart_path)
        
        # 处理局部解释
        processed_local_explanations = []
        if local_explanations:
            for i, explanation in enumerate(local_explanations):
                try:
                    # 生成局部图表
                    local_chart_path = os.path.join(self.output_dir, f"{report_name}_local_{i}.png")
                    instance_idx = explanation.get('instance_idx', i)
                    
                    # 确保实例索引有效
                    if instance_idx >= len(X_test):
                        print(f"警告: 实例索引 {instance_idx} 超出范围，使用索引 {i}")
                        instance_idx = min(i, len(X_test) - 1)
                    
                    result = shap_explainer.create_waterfall_plot(instance_idx, X_test, local_chart_path)
                    
                    if result == "save_failed":
                        print(f"警告: 无法保存局部图表 {i}，跳过图表生成")
                        local_chart_b64 = ""
                    else:
                        local_chart_b64 = self._image_to_base64(local_chart_path)
                        
                except Exception as e:
                    print(f"生成局部图表 {i} 时出错: {e}")
                    local_chart_b64 = ""
                
                processed_explanation = {
                    'instance_id': explanation.get('instance_idx', i),
                    'prediction_label': explanation.get('prediction_label', 'Unknown'),
                    'prediction_risk': explanation.get('prediction_risk', 'medium'),
                    'feature_contributions': explanation.get('feature_contributions', {}),
                    'feature_values': explanation.get('feature_values', {}),
                    'natural_language_explanation': explanation.get('natural_language_explanation', ''),
                    'chart': local_chart_b64
                }
                processed_local_explanations.append(processed_explanation)
        
        # 准备模板数据
        template_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'total_samples': len(X_test),
            'feature_count': len(X_test.columns),
            'global_importance': global_importance,
            'global_explanation': natural_language_explanations.get('global', '') if natural_language_explanations else '',
            'global_chart': global_chart_b64,
            'local_explanations': processed_local_explanations,
            'business_recommendations': natural_language_explanations.get('recommendations', '') if natural_language_explanations else ''
        }
        
        # 渲染HTML报告
        template = Template(self.html_template)
        html_content = template.render(**template_data)
        
        # 保存HTML报告
        html_path = os.path.join(self.output_dir, f"{report_name}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 清理临时图片文件
        self._cleanup_temp_files([global_chart_path] + 
                                [os.path.join(self.output_dir, f"{report_name}_local_{i}.png") 
                                 for i in range(len(processed_local_explanations))])
        
        return html_path
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except:
            return ""
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """清理临时文件"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass