"""
自然语言解释器模块
将SHAP结果转换为业务友好的自然语言描述
"""
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class NaturalLanguageExplainer:
    """自然语言解释器"""
    
    def __init__(self, use_llm: bool = False):
        """
        初始化自然语言解释器
        
        Args:
            use_llm: 是否使用大语言模型生成解释
        """
        self.use_llm = use_llm
        if use_llm:
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except ImportError:
                print("Warning: OpenAI not installed, falling back to template-based explanations")
                self.use_llm = False
    
    def explain_global_importance(self, 
                                feature_importance: Dict[str, float],
                                feature_descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        生成全局特征重要性的自然语言解释
        
        Args:
            feature_importance: 特征重要性字典
            feature_descriptions: 特征描述字典
            
        Returns:
            str: 自然语言解释
        """
        if not feature_importance:
            return "无法生成全局解释：特征重要性数据为空"
        
        # 获取前5个最重要的特征
        top_features = list(feature_importance.items())[:5]
        
        if self.use_llm:
            return self._generate_llm_global_explanation(top_features, feature_descriptions)
        else:
            return self._generate_template_global_explanation(top_features, feature_descriptions)
    
    def explain_local_prediction(self,
                               local_explanation: Dict[str, Any],
                               feature_descriptions: Optional[Dict[str, str]] = None,
                               prediction_threshold: float = 0.5) -> str:
        """
        生成局部预测的自然语言解释
        
        Args:
            local_explanation: 局部解释字典
            feature_descriptions: 特征描述字典
            prediction_threshold: 预测阈值
            
        Returns:
            str: 自然语言解释
        """
        feature_contributions = local_explanation.get('feature_contributions', {})
        feature_values = local_explanation.get('feature_values', {})
        prediction = local_explanation.get('prediction', 0)
        
        if self.use_llm:
            return self._generate_llm_local_explanation(
                feature_contributions, feature_values, prediction, 
                feature_descriptions, prediction_threshold
            )
        else:
            return self._generate_template_local_explanation(
                feature_contributions, feature_values, prediction,
                feature_descriptions, prediction_threshold
            )
    
    def generate_business_recommendations(self,
                                        global_importance: Dict[str, float],
                                        local_explanations: List[Dict[str, Any]]) -> str:
        """
        生成业务建议
        
        Args:
            global_importance: 全局特征重要性
            local_explanations: 局部解释列表
            
        Returns:
            str: 业务建议
        """
        if self.use_llm:
            return self._generate_llm_recommendations(global_importance, local_explanations)
        else:
            return self._generate_template_recommendations(global_importance, local_explanations)
    
    def _get_importance_level(self, importance: float, max_importance: float) -> str:
        """根据重要性得分确定重要性级别"""
        ratio = importance / max_importance if max_importance > 0 else 0
        
        if ratio >= 0.7:
            return "决定性"
        elif ratio >= 0.3:
            return "重要"
        else:
            return "次要"
    
    def _generate_template_global_explanation(self, 
                                            top_features: List[tuple],
                                            feature_descriptions: Optional[Dict[str, str]]) -> str:
        """基于模板生成全局解释"""
        explanation = "根据模型分析，影响预测结果的关键因素按重要性排序如下：\n\n"
        
        if not top_features:
            return explanation + "未能识别到重要特征。"
            
        max_importance = top_features[0][1] if top_features else 0
        
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_name = feature_descriptions.get(feature, feature) if feature_descriptions else feature
            importance_level = self._get_importance_level(importance, max_importance)
            
            explanation += f"{i}. **{feature_name}** (重要性: {importance:.3f})\n"
            explanation += f"   - 影响程度: {importance_level}\n"
            
            if importance_level == "决定性":
                explanation += f"   - 该因素对模型决策起决定性作用，是最关键的判断依据\n"
            elif importance_level == "重要":
                explanation += f"   - 该因素对模型决策有重要影响，需要重点关注\n"
            else:
                explanation += f"   - 该因素对模型决策有一定影响，可作为辅助判断依据\n"
            explanation += "\n"
        
        return explanation
    
    def _generate_template_local_explanation(self,
                                           feature_contributions: Dict[str, float],
                                           feature_values: Dict[str, Any],
                                           prediction: float,
                                           feature_descriptions: Optional[Dict[str, str]],
                                           threshold: float) -> str:
        """基于模板生成局部解释"""
        # 确保预测值是标量
        try:
            prediction_value = float(prediction)
        except (ValueError, TypeError):
            prediction_value = 0.5
            
        # 判断预测结果
        if prediction_value >= threshold:
            result_text = "通过/高风险"
            result_reason = "正向贡献因素占主导"
        else:
            result_text = "拒绝/低风险"
            result_reason = "负向贡献因素占主导"
        
        explanation = f"**预测结果: {result_text}** (预测值: {prediction_value:.3f})\n\n"
        explanation += f"**决策原因: {result_reason}**\n\n"
        
        # 确保贡献度是标量值
        processed_contributions = {}
        for feature, value in feature_contributions.items():
            # 如果是数组，取第一个值
            if hasattr(value, '__iter__') and not isinstance(value, str):
                try:
                    processed_contributions[feature] = float(value[0])
                except (IndexError, TypeError):
                    processed_contributions[feature] = 0.0
            else:
                try:
                    processed_contributions[feature] = float(value)
                except (ValueError, TypeError):
                    processed_contributions[feature] = 0.0
        
        # 按贡献度排序
        sorted_contributions = sorted(processed_contributions.items(), 
                                    key=lambda x: abs(x[1]), reverse=True)
        
        positive_factors = [(f, c) for f, c in sorted_contributions if c > 0]
        negative_factors = [(f, c) for f, c in sorted_contributions if c < 0]
        
        if positive_factors:
            explanation += "**正向影响因素 (支持通过):**\n"
            for feature, contribution in positive_factors[:3]:
                feature_name = feature_descriptions.get(feature, feature) if feature_descriptions else feature
                feature_value = feature_values.get(feature, "未知")
                explanation += f"• {feature_name}: {feature_value} (贡献度: +{contribution:.3f})\n"
            explanation += "\n"
        
        if negative_factors:
            explanation += "**负向影响因素 (支持拒绝):**\n"
            for feature, contribution in negative_factors[:3]:
                feature_name = feature_descriptions.get(feature, feature) if feature_descriptions else feature
                feature_value = feature_values.get(feature, "未知")
                explanation += f"• {feature_name}: {feature_value} (贡献度: {contribution:.3f})\n"
            explanation += "\n"
        
        return explanation
    
    def _generate_template_recommendations(self,
                                         global_importance: Dict[str, float],
                                         local_explanations: List[Dict[str, Any]]) -> str:
        """基于模板生成业务建议"""
        recommendations = "## 业务优化建议\n\n"
        
        # 基于全局重要性的建议
        top_feature = list(global_importance.keys())[0]
        recommendations += f"### 1. 重点关注 '{top_feature}'\n"
        recommendations += f"该特征是模型决策的最重要因素，建议:\n"
        recommendations += f"- 确保该特征数据的准确性和完整性\n"
        recommendations += f"- 建立该特征的监控机制\n"
        recommendations += f"- 针对该特征制定相应的业务策略\n\n"
        
        # 基于局部解释的建议
        if local_explanations:
            recommendations += "### 2. 个案处理建议\n"
            high_risk_count = sum(1 for exp in local_explanations 
                                if exp.get('prediction', 0) >= 0.5)
            
            recommendations += f"- 在分析的 {len(local_explanations)} 个样本中，"
            recommendations += f"有 {high_risk_count} 个被预测为高风险\n"
            recommendations += f"- 建议对高风险样本进行人工复核\n"
            recommendations += f"- 关注负向贡献因素，制定针对性改进措施\n\n"
        
        recommendations += "### 3. 模型监控建议\n"
        recommendations += "- 定期评估模型性能，确保预测准确性\n"
        recommendations += "- 监控特征分布变化，及时发现数据漂移\n"
        recommendations += "- 建立模型解释结果的业务验证机制\n"
        
        return recommendations