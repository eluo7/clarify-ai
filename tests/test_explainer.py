"""
模型解释器测试
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clarify_ai import ModelExplainer


class TestModelExplainer(unittest.TestCase):
    """测试ModelExplainer类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        X, y = make_classification(
            n_samples=100, 
            n_features=5, 
            n_informative=3, 
            random_state=42
        )
        
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        self.X_train = pd.DataFrame(X[:80], columns=feature_names)
        self.X_test = pd.DataFrame(X[80:], columns=feature_names)
        self.y_train = y[:80]
        
        # 训练模型
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # 初始化解释器
        self.explainer = ModelExplainer(
            model=self.model,
            X_train=self.X_train,
            output_dir="test_reports"
        )
    
    def test_global_importance(self):
        """测试全局特征重要性"""
        importance = self.explainer.get_global_importance(self.X_test)
        
        # 检查结果
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 5)  # 5个特征
        
        # 检查值是否合理
        for feature, score in importance.items():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_local_explanation(self):
        """测试局部解释"""
        explanation = self.explainer.explain_prediction(0, self.X_test)
        
        # 检查结果
        self.assertIsInstance(explanation, dict)
        self.assertIn('feature_contributions', explanation)
        self.assertIn('feature_values', explanation)
        
        # 检查特征贡献
        contributions = explanation['feature_contributions']
        self.assertEqual(len(contributions), 5)  # 5个特征
    
    def test_natural_language_explanation(self):
        """测试自然语言解释"""
        importance = self.explainer.get_global_importance(self.X_test)
        explanation = self.explainer.explain_prediction(0, self.X_test)
        
        nl_explanation = self.explainer.get_natural_language_explanation(
            global_importance=importance,
            local_explanation=explanation
        )
        
        # 检查结果
        self.assertIsInstance(nl_explanation, dict)
        self.assertIn('global', nl_explanation)
        self.assertIn('local', nl_explanation)
        
        # 检查解释内容
        self.assertIsInstance(nl_explanation['global'], str)
        self.assertGreater(len(nl_explanation['global']), 0)
    
    def test_report_generation(self):
        """测试报告生成"""
        report_path = self.explainer.generate_report(
            X_test=self.X_test,
            instances_to_explain=[0, 1],
            model_type="测试模型",
            report_name="test_report"
        )
        
        # 检查报告文件是否生成
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith('.html'))
    
    def tearDown(self):
        """测试后清理"""
        # 清理测试生成的报告
        import shutil
        if os.path.exists("test_reports"):
            shutil.rmtree("test_reports")


if __name__ == '__main__':
    unittest.main()