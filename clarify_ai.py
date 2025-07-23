"""
Clarify AI - 机器学习模型可解释性Agent
主入口文件
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt

from src.core.shap_explainer import SHAPExplainer
from src.nlp.natural_language_explainer import NaturalLanguageExplainer
from src.report.report_generator import ExplanationReportGenerator


class ModelExplainer:
    """模型可解释性Agent主类"""
    
    def __init__(self, 
                model, 
                X_train: pd.DataFrame,
                feature_names: Optional[List[str]] = None,
                feature_descriptions: Optional[Dict[str, str]] = None,
                use_llm: bool = False,
                output_dir: str = "reports"):
        """
        初始化模型解释器
        
        Args:
            model: 训练好的机器学习模型
            X_train: 训练数据
            feature_names: 特征名称列表
            feature_descriptions: 特征描述字典
            use_llm: 是否使用大语言模型生成解释
            output_dir: 报告输出目录
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        self.feature_descriptions = feature_descriptions or {}
        
        # 初始化组件
        self.shap_explainer = SHAPExplainer(model, X_train, self.feature_names)
        self.nl_explainer = NaturalLanguageExplainer(use_llm=use_llm)
        self.report_generator = ExplanationReportGenerator(output_dir=output_dir)
    
    def get_global_importance(self, X_test: pd.DataFrame) -> Dict[str, float]:
        """
        获取全局特征重要性
        
        Args:
            X_test: 测试数据
            
        Returns:
            Dict: 特征重要性字典
        """
        return self.shap_explainer.global_feature_importance(X_test)
    
    def explain_prediction(self, 
                         instance_id: Union[int, pd.Series],
                         X_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        解释单个预测
        
        Args:
            instance_id: 实例ID或实例数据
            X_test: 测试数据集
            
        Returns:
            Dict: 局部解释结果
        """
        if isinstance(instance_id, pd.Series):
            # 如果传入的是实例数据，将其添加到测试集
            if X_test is None:
                X_test = pd.DataFrame([instance_id])
            else:
                X_test = pd.concat([X_test, pd.DataFrame([instance_id])])
            instance_idx = len(X_test) - 1
        else:
            instance_idx = instance_id
        
        return self.shap_explainer.local_explanation(instance_idx, X_test)
    
    def get_natural_language_explanation(self,
                                       global_importance: Dict[str, float] = None,
                                       local_explanation: Dict[str, Any] = None,
                                       X_test: pd.DataFrame = None) -> Dict[str, str]:
        """
        获取自然语言解释
        
        Args:
            global_importance: 全局特征重要性
            local_explanation: 局部解释
            X_test: 测试数据
            
        Returns:
            Dict: 自然语言解释字典
        """
        explanations = {}
        
        if global_importance:
            explanations['global'] = self.nl_explainer.explain_global_importance(
                global_importance, self.feature_descriptions
            )
        
        if local_explanation:
            explanations['local'] = self.nl_explainer.explain_local_prediction(
                local_explanation, self.feature_descriptions
            )
        
        if global_importance and X_test is not None:
            # 生成业务建议
            local_explanations = []
            if local_explanation:
                local_explanations.append(local_explanation)
            
            explanations['recommendations'] = self.nl_explainer.generate_business_recommendations(
                global_importance, local_explanations
            )
        
        return explanations
    
    def plot_global_importance(self, X_test: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        绘制全局特征重要性图
        
        Args:
            X_test: 测试数据
            save_path: 保存路径
            
        Returns:
            str: 图片保存路径或显示状态
        """
        return self.shap_explainer.create_summary_plot(X_test, save_path)
    
    def plot_prediction_explanation(self, 
                                  instance_id: int, 
                                  X_test: pd.DataFrame,
                                  save_path: Optional[str] = None) -> str:
        """
        绘制单个预测的解释图
        
        Args:
            instance_id: 实例ID
            X_test: 测试数据
            save_path: 保存路径
            
        Returns:
            str: 图片保存路径或显示状态
        """
        return self.shap_explainer.create_waterfall_plot(instance_id, X_test, save_path)
    
    def generate_report(self,
                      X_test: pd.DataFrame,
                      instances_to_explain: Optional[List[int]] = None,
                      model_type: str = "Unknown",
                      report_name: Optional[str] = None) -> str:
        """
        生成完整的解释报告
        
        Args:
            X_test: 测试数据
            instances_to_explain: 要解释的实例ID列表
            model_type: 模型类型
            report_name: 报告名称
            
        Returns:
            str: 报告文件路径
        """
        # 计算全局特征重要性
        global_importance = self.get_global_importance(X_test)
        
        # 生成自然语言解释
        nl_explanations = {
            'global': self.nl_explainer.explain_global_importance(
                global_importance, self.feature_descriptions
            )
        }
        
        # 处理局部解释
        local_explanations = []
        if instances_to_explain:
            for idx in instances_to_explain:
                local_exp = self.shap_explainer.local_explanation(idx, X_test)
                
                # 添加预测标签和风险级别
                prediction = local_exp.get('prediction', 0)
                if hasattr(self.model, 'predict_proba'):
                    try:
                        proba = self.model.predict_proba(X_test.iloc[[idx]])[0][1]
                        local_exp['prediction'] = proba
                    except:
                        pass
                
                # 确定风险级别
                if prediction >= 0.7:
                    risk = 'high'
                    label = '高风险'
                elif prediction >= 0.3:
                    risk = 'medium'
                    label = '中风险'
                else:
                    risk = 'low'
                    label = '低风险'
                
                # 生成自然语言解释
                nl_explanation = self.nl_explainer.explain_local_prediction(
                    local_exp, self.feature_descriptions
                )
                
                local_explanations.append({
                    'instance_idx': idx,
                    'prediction_label': label,
                    'prediction_risk': risk,
                    'feature_contributions': local_exp['feature_contributions'],
                    'feature_values': local_exp['feature_values'],
                    'natural_language_explanation': nl_explanation
                })
        
        # 生成业务建议
        nl_explanations['recommendations'] = self.nl_explainer.generate_business_recommendations(
            global_importance, local_explanations
        )
        
        # 生成报告
        return self.report_generator.generate_report(
            shap_explainer=self.shap_explainer,
            X_test=X_test,
            global_importance=global_importance,
            local_explanations=local_explanations,
            natural_language_explanations=nl_explanations,
            model_type=model_type,
            report_name=report_name
        )


if __name__ == "__main__":
    print("Clarify AI - 机器学习模型可解释性Agent")
    print("请导入此模块并使用ModelExplainer类")