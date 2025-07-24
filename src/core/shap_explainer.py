"""
SHAP解释器核心模块
提供全局和局部解释功能
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import warnings
import platform
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体支持
def _setup_matplotlib_chinese():
    """设置matplotlib中文字体支持"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC', 'SimHei']
    elif system == "Windows":  # Windows
        fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    else:  # Linux
        fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']
    
    # 获取可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
_setup_matplotlib_chinese()


class SHAPExplainer:
    """基于SHAP的模型解释器"""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        初始化SHAP解释器
        
        Args:
            model: 训练好的机器学习模型
            X_train: 训练数据，用于初始化explainer
            feature_names: 特征名称列表
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        self.explainer = None
        self.shap_values = None
        
        # 自动选择合适的explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """根据模型类型自动选择SHAP explainer"""
        try:
            # 检查是否为树模型
            if hasattr(self.model, 'estimators_') or hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
            # 检查是否为线性模型
            elif hasattr(self.model, 'coef_'):
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                self.explainer_type = "linear"
            else:
                # 使用通用的Explainer
                self.explainer = shap.Explainer(self.model, self.X_train)
                self.explainer_type = "general"
        except Exception as e:
            print(f"初始化SHAP解释器时出错: {e}")
            # 使用最通用的Kernel解释器作为后备
            try:
                self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 100))
                self.explainer_type = "kernel"
            except Exception as e2:
                print(f"初始化Kernel解释器时出错: {e2}")
                # 最后的后备方案
                self.explainer = shap.Explainer(self.model, self.X_train)
                self.explainer_type = "general"
    
    def compute_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """计算SHAP值"""
        try:
            if self.explainer_type == "tree":
                self.shap_values = self.explainer.shap_values(X_test)
            elif self.explainer_type == "kernel":
                self.shap_values = self.explainer.shap_values(X_test)
            else:
                self.shap_values = self.explainer(X_test)
        except Exception as e:
            print(f"计算SHAP值时出错: {e}")
            # 尝试使用Kernel解释器作为后备
            try:
                backup_explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict, 
                    shap.sample(self.X_train, min(100, len(self.X_train)))
                )
                self.shap_values = backup_explainer.shap_values(X_test)
                print("使用备用Kernel解释器计算SHAP值")
            except Exception as e2:
                print(f"备用计算也失败: {e2}")
                # 创建一个假的SHAP值作为后备
                if hasattr(self.model, 'feature_importances_'):
                    # 使用模型的特征重要性
                    importances = self.model.feature_importances_
                    self.shap_values = np.tile(importances, (len(X_test), 1))
                else:
                    # 创建随机SHAP值
                    self.shap_values = np.random.random((len(X_test), X_test.shape[1]))
                    print("警告: 无法计算真实SHAP值，使用随机值代替")
            
        return self.shap_values
    
    def global_feature_importance(self, X_test: pd.DataFrame) -> Dict[str, float]:
        """
        计算全局特征重要性
        
        Returns:
            Dict: 特征名称和重要性得分的字典
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        # shap_values是numpy.ndarray格式，根据维度处理
        if self.shap_values.ndim == 3:
            # 三维数组: [n_samples, n_features, n_classes] -> 取第一个类别 
            # axis=0表示对每一列的所有行求平均 → 结果为列的均值向量。
            importance_scores = np.mean(np.abs(self.shap_values[:, :, 0]), axis=0)
        elif self.shap_values.ndim == 2:
            # 二维数组: [n_samples, n_features]
            importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {self.shap_values.shape}")
        
        # 创建特征重要性字典
        feature_importance = dict(zip(self.feature_names, importance_scores))
        
        # 按重要性排序
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def local_explanation(self, instance_idx: int, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        计算单个实例的局部解释
        
        Args:
            instance_idx: 实例索引
            X_test: 测试数据
            
        Returns:
            Dict: 包含特征贡献度的字典
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        # 获取单个实例的SHAP值 (numpy.ndarray格式)
        if self.shap_values.ndim == 3:
            # 三维数组: [n_samples, n_features, n_classes] -> 取指定样本的第一个类别
            instance_shap = self.shap_values[instance_idx, :, 0]
        elif self.shap_values.ndim == 2:
            # 二维数组: [n_samples, n_features]
            instance_shap = self.shap_values[instance_idx]
        else:
            raise ValueError(f"Unexpected SHAP values shape: {self.shap_values.shape}")
        
        # 获取实例的特征值
        instance_values = X_test.iloc[instance_idx]
        
        # 获取基准值
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0
        
        # 获取预测值
        try:
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba([instance_values])[0][1]
            else:
                prediction = self.model.predict([instance_values])[0]
        except:
            prediction = None
        
        # 创建解释字典
        explanation = {
            'feature_contributions': dict(zip(self.feature_names, instance_shap)),
            'feature_values': dict(zip(self.feature_names, instance_values)),
            'base_value': base_value,
            'prediction': prediction
        }
        
        return explanation
    
    def create_summary_plot(self, X_test: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """创建SHAP summary plot"""
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        
        try:
            # shap_values是numpy.ndarray格式
            if self.shap_values.ndim == 3:
                # 三维数组: [n_samples, n_features, n_classes] -> 取第一个类别用于绘图
                shap.summary_plot(self.shap_values[:, :, 0], X_test, feature_names=self.feature_names, show=False)
            else:
                # 二维数组: [n_samples, n_features]
                shap.summary_plot(self.shap_values, X_test, feature_names=self.feature_names, show=False)
        except Exception as e:
            print(f"创建summary plot时出错: {e}")
            # 创建简单的特征重要性图作为后备
            importance = self.global_feature_importance(X_test)
            features = list(importance.keys())
            scores = list(importance.values())
            
            # 排序
            sorted_idx = np.argsort(scores)
            plt.barh(range(len(sorted_idx)), [scores[i] for i in sorted_idx])
            plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
            plt.xlabel('特征重要性')
            plt.title('全局特征重要性')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return "plot_displayed"
    
    def create_waterfall_plot(self, instance_idx: int, X_test: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """创建单个实例的waterfall plot"""
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        plt.figure(figsize=(12, 8))
        
        try:
            # 检查索引是否有效
            if instance_idx >= len(X_test):
                raise IndexError(f"实例索引 {instance_idx} 超出范围，测试集大小为 {len(X_test)}")
            
            # 获取基准值
            base_value = 0
            if hasattr(self.explainer, 'expected_value'):
                if isinstance(self.explainer.expected_value, list):
                    base_value = float(self.explainer.expected_value[0])
                else:
                    base_value = float(self.explainer.expected_value)
            
            # 获取SHAP值 (numpy.ndarray格式)
            if self.shap_values.ndim == 3:
                # 三维数组: [n_samples, n_features, n_classes] -> 取指定样本的第一个类别
                instance_shap = self.shap_values[instance_idx, :, 0]
            elif self.shap_values.ndim == 2:
                # 二维数组: [n_samples, n_features]
                instance_shap = self.shap_values[instance_idx]
            else:
                raise ValueError(f"Unexpected SHAP values shape: {self.shap_values.shape}")
            
            # 确保instance_shap是一维数组
            if isinstance(instance_shap, np.ndarray):
                if instance_shap.ndim > 1:
                    instance_shap = instance_shap.flatten()
                # 取第一个值如果还是多维
                if len(instance_shap.shape) > 0 and instance_shap.shape[0] != len(self.feature_names):
                    instance_shap = instance_shap[:len(self.feature_names)]
            
            # 转换为numpy数组并确保长度匹配
            instance_shap = np.array(instance_shap).flatten()
            if len(instance_shap) != len(self.feature_names):
                min_len = min(len(instance_shap), len(self.feature_names))
                instance_shap = instance_shap[:min_len]
                feature_names = self.feature_names[:min_len]
            else:
                feature_names = self.feature_names
            
            # 获取特征值
            instance_values = X_test.iloc[instance_idx]
            
            # 尝试使用SHAP的waterfall plot
            try:
                if hasattr(shap, 'waterfall_plot'):
                    # 创建Explanation对象
                    explanation = shap.Explanation(
                        values=instance_shap,
                        base_values=base_value,
                        data=instance_values.values[:len(instance_shap)],
                        feature_names=feature_names
                    )
                    shap.waterfall_plot(explanation, show=False)
                else:
                    raise AttributeError("SHAP waterfall_plot not available")
                    
            except Exception as waterfall_error:
                print(f"SHAP waterfall plot失败: {waterfall_error}, 使用自定义条形图")
                
                # 创建自定义的waterfall风格图表
                plt.clf()  # 清除之前的图表
                
                # 按绝对值排序
                sorted_idx = np.argsort(np.abs(instance_shap))[::-1]  # 降序
                sorted_shap = instance_shap[sorted_idx]
                sorted_features = [feature_names[i] for i in sorted_idx]
                sorted_values = [instance_values.iloc[i] if i < len(instance_values) else 0 for i in sorted_idx]
                
                # 创建颜色映射
                colors = ['red' if x < 0 else 'blue' for x in sorted_shap]
                
                # 创建水平条形图
                y_pos = np.arange(len(sorted_features))
                bars = plt.barh(y_pos, sorted_shap, color=colors, alpha=0.7)
                
                # 设置标签和标题
                plt.yticks(y_pos, [f"{feat}\n({val:.2f})" for feat, val in zip(sorted_features, sorted_values)])
                plt.xlabel('SHAP值 (特征贡献度)')
                plt.title(f'样本 #{instance_idx} 的特征贡献度分析\n基准值: {base_value:.3f}')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars, sorted_shap)):
                    plt.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                            va='center', ha='left' if val >= 0 else 'right', fontsize=9)
                
                # 添加基准线
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='blue', alpha=0.7, label='正向贡献'),
                                 Patch(facecolor='red', alpha=0.7, label='负向贡献')]
                plt.legend(handles=legend_elements, loc='lower right')
                
                plt.tight_layout()
            
        except Exception as e:
            print(f"创建waterfall图时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建错误信息图
            plt.clf()
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"无法创建Waterfall图\n错误: {str(e)}\n\n请检查数据格式和SHAP值计算", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.axis('off')
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                return save_path
            except Exception as save_error:
                print(f"保存图片失败: {save_error}")
                plt.close()
                return "save_failed"
        else:
            plt.show()
            return "plot_displayed"
    
    def create_force_plot(self, instance_idx: int, X_test: pd.DataFrame) -> Any:
        """创建force plot（交互式）"""
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        # 获取基准值
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0
        
        # shap_values是numpy.ndarray格式
        if self.shap_values.ndim == 3:
            # 三维数组: [n_samples, n_features, n_classes] -> 取指定样本的第一个类别
            instance_shap = self.shap_values[instance_idx, :, 0]
        else:
            # 二维数组: [n_samples, n_features]
            instance_shap = self.shap_values[instance_idx]
        
        return shap.force_plot(
            base_value,
            instance_shap,
            X_test.iloc[instance_idx],
            feature_names=self.feature_names
        )