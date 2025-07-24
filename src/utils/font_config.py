"""
字体配置工具模块
解决matplotlib中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


def get_system_chinese_fonts():
    """获取系统可用的中文字体列表"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return [
            'Arial Unicode MS',
            'Hiragino Sans GB', 
            'PingFang SC',
            'Heiti SC',
            'STHeiti',
            'SimHei'
        ]
    elif system == "Windows":  # Windows
        return [
            'Microsoft YaHei',
            'SimHei',
            'KaiTi',
            'FangSong',
            'SimSun'
        ]
    else:  # Linux
        return [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Source Han Sans SC',
            'DejaVu Sans',
            'SimHei'
        ]


def find_available_chinese_font():
    """查找系统中可用的中文字体"""
    chinese_fonts = get_system_chinese_fonts()
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            return font
    
    return None


def setup_matplotlib_chinese_font():
    """设置matplotlib中文字体支持"""
    # 查找可用的中文字体
    chinese_font = find_available_chinese_font()
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
        print(f"已设置中文字体: {chinese_font}")
    else:
        # 使用默认字体，但可能无法显示中文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到合适的中文字体，中文可能显示为方块")
    
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_font is not None


def list_all_fonts():
    """列出系统所有可用字体"""
    fonts = [f.name for f in fm.fontManager.ttflist]
    fonts = sorted(list(set(fonts)))  # 去重并排序
    
    print("系统可用字体:")
    for i, font in enumerate(fonts, 1):
        print(f"{i:3d}. {font}")
    
    return fonts


def test_chinese_display():
    """测试中文显示效果"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='正弦曲线')
    ax.set_xlabel('横轴标签')
    ax.set_ylabel('纵轴标签')
    ax.set_title('中文字体测试图表')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("如果图表中的中文正常显示，说明字体配置成功")


if __name__ == "__main__":
    print("=== matplotlib中文字体配置工具 ===")
    
    # 设置中文字体
    success = setup_matplotlib_chinese_font()
    
    if success:
        print("字体配置成功！")
        # 测试中文显示
        test_chinese_display()
    else:
        print("字体配置失败，列出所有可用字体:")
        list_all_fonts()