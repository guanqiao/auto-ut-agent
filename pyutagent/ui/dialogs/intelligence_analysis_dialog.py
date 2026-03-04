"""智能分析结果展示对话框

用于展示 SemanticAnalyzer 和 RootCauseAnalyzer 的分析结果，
提供可视化的代码理解、测试场景、错误根因等信息。
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QLabel,
    QScrollArea, QWidget, QFrame, QSplitter, QPushButton,
    QProgressBar, QGroupBox, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QTextCursor

from ..core.semantic_analyzer import SemanticAnalyzer, TestScenario, BoundaryCondition
from ..core.root_cause_analyzer import RootCauseAnalyzer, RootCauseAnalysis

logger = logging.getLogger(__name__)


class IntelligenceAnalysisDialog(QDialog):
    """智能分析结果对话框
    
    功能:
    - 展示语义分析结果
    - 展示错误根因分析结果
    - 可视化测试场景
    - 可视化边界条件
    - 提供修复建议
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("智能分析结果")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        self.semantic_analyzer = SemanticAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建标签页
        self.tabs = QTabWidget()
        
        # 标签页 1: 语义分析
        self.semantic_tab = self._create_semantic_analysis_tab()
        self.tabs.addTab(self.semantic_tab, "📊 语义分析")
        
        # 标签页 2: 测试场景
        self.scenarios_tab = self._create_test_scenarios_tab()
        self.tabs.addTab(self.scenarios_tab, "🎯 测试场景")
        
        # 标签页 3: 边界条件
        self.boundaries_tab = self._create_boundary_conditions_tab()
        self.tabs.addTab(self.boundaries_tab, "⚠️ 边界条件")
        
        # 标签页 4: 错误根因
        self.rca_tab = self._create_root_cause_tab()
        self.tabs.addTab(self.rca_tab, "🔍 错误根因")
        
        # 标签页 5: 修复建议
        self.fixes_tab = self._create_fix_suggestions_tab()
        self.tabs.addTab(self.fixes_tab, "💡 修复建议")
        
        layout.addWidget(self.tabs)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 刷新分析")
        self.refresh_btn.clicked.connect(self._on_refresh)
        button_layout.addWidget(self.refresh_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_semantic_analysis_tab(self) -> QWidget:
        """创建语义分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：代码结构树
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_label = QLabel("📁 代码结构")
        left_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(left_label)
        
        self.code_tree = QTreeWidget()
        self.code_tree.setHeaderLabels(["名称", "类型", "复杂度"])
        self.code_tree.setColumnWidth(0, 200)
        self.code_tree.setColumnWidth(1, 150)
        self.code_tree.setColumnWidth(2, 80)
        left_layout.addWidget(self.code_tree)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # 右侧：业务逻辑信息
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_label = QLabel("💼 业务逻辑")
        right_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_layout.addWidget(right_label)
        
        self.business_logic_text = QTextEdit()
        self.business_logic_text.setReadOnly(True)
        right_layout.addWidget(self.business_logic_text)
        
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        widget.setLayout(layout)
        return widget
    
    def _create_test_scenarios_tab(self) -> QWidget:
        """创建测试场景标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 统计信息
        stats_layout = QGridLayout()
        
        self.total_scenarios_label = QLabel("总场景数：0")
        stats_layout.addWidget(self.total_scenarios_label, 0, 0)
        
        self.normal_scenarios_label = QLabel("正常场景：0")
        stats_layout.addWidget(self.normal_scenarios_label, 0, 1)
        
        self.edge_scenarios_label = QLabel("边界场景：0")
        stats_layout.addWidget(self.edge_scenarios_label, 0, 2)
        
        self.exception_scenarios_label = QLabel("异常场景：0")
        stats_layout.addWidget(self.exception_scenarios_label, 0, 3)
        
        layout.addLayout(stats_layout)
        
        # 场景列表
        self.scenarios_tree = QTreeWidget()
        self.scenarios_tree.setHeaderLabels(["场景", "目标方法", "类型", "优先级", "描述"])
        self.scenarios_tree.setColumnWidth(0, 300)
        self.scenarios_tree.setColumnWidth(1, 150)
        self.scenarios_tree.setColumnWidth(2, 100)
        self.scenarios_tree.setColumnWidth(3, 80)
        self.scenarios_tree.setColumnWidth(4, 300)
        self.scenarios_tree.itemClicked.connect(self._on_scenario_clicked)
        layout.addWidget(self.scenarios_tree)
        
        # 场景详情
        details_group = QGroupBox("场景详情")
        details_layout = QVBoxLayout()
        
        self.scenario_details = QTextEdit()
        self.scenario_details.setReadOnly(True)
        details_layout.addWidget(self.scenario_details)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        widget.setLayout(layout)
        return widget
    
    def _create_boundary_conditions_tab(self) -> QWidget:
        """创建边界条件标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 边界条件列表
        self.boundaries_tree = QTreeWidget()
        self.boundaries_tree.setHeaderLabels(["参数", "边界类型", "测试值", "预期行为"])
        self.boundaries_tree.setColumnWidth(0, 150)
        self.boundaries_tree.setColumnWidth(1, 150)
        self.boundaries_tree.setColumnWidth(2, 150)
        self.boundaries_tree.setColumnWidth(3, 400)
        layout.addWidget(self.boundaries_tree)
        
        # 统计信息
        stats_layout = QHBoxLayout()
        
        self.null_checks_label = QLabel("🔴 空值检查：0")
        stats_layout.addWidget(self.null_checks_label)
        
        self.empty_checks_label = QLabel("🟡 空集合检查：0")
        stats_layout.addWidget(self.empty_checks_label)
        
        self.range_checks_label = QLabel("🟢 范围检查：0")
        stats_layout.addWidget(self.range_checks_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        widget.setLayout(layout)
        return widget
    
    def _create_root_cause_tab(self) -> QWidget:
        """创建错误根因标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 置信度指示器
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("分析置信度:"))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p%")
        confidence_layout.addWidget(self.confidence_bar)
        
        self.confidence_label = QLabel("0%")
        confidence_layout.addWidget(self.confidence_label)
        
        layout.addLayout(confidence_layout)
        
        # 根因列表
        self.root_causes_tree = QTreeWidget()
        self.root_causes_tree.setHeaderLabels(["根因", "类别", "置信度", "位置", "贡献因素"])
        self.root_causes_tree.setColumnWidth(0, 300)
        self.root_causes_tree.setColumnWidth(1, 150)
        self.root_causes_tree.setColumnWidth(2, 100)
        self.root_causes_tree.setColumnWidth(3, 200)
        self.root_causes_tree.setColumnWidth(4, 200)
        layout.addWidget(self.root_causes_tree)
        
        # 证据展示
        evidence_group = QGroupBox("证据")
        evidence_layout = QVBoxLayout()
        
        self.evidence_text = QTextEdit()
        self.evidence_text.setReadOnly(True)
        evidence_layout.addWidget(self.evidence_text)
        
        evidence_group.setLayout(evidence_layout)
        layout.addWidget(evidence_group)
        
        widget.setLayout(layout)
        return widget
    
    def _create_fix_suggestions_tab(self) -> QWidget:
        """创建修复建议标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 修复策略列表
        self.fixes_tree = QTreeWidget()
        self.fixes_tree.setHeaderLabels(["策略", "类型", "优先级", "工作量", "成功率", "描述"])
        self.fixes_tree.setColumnWidth(0, 150)
        self.fixes_tree.setColumnWidth(1, 150)
        self.fixes_tree.setColumnWidth(2, 80)
        self.fixes_tree.setColumnWidth(3, 100)
        self.fixes_tree.setColumnWidth(4, 100)
        self.fixes_tree.setColumnWidth(5, 300)
        self.fixes_tree.itemClicked.connect(self._on_fix_clicked)
        layout.addWidget(self.fixes_tree)
        
        # 修复详情
        details_group = QGroupBox("修复详情")
        details_layout = QVBoxLayout()
        
        self.fix_details = QTextEdit()
        self.fix_details.setReadOnly(True)
        details_layout.addWidget(self.fix_details)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        widget.setLayout(layout)
        return widget
    
    def analyze_code(self, file_path: str, java_class: Any):
        """分析代码
        
        Args:
            file_path: Java 文件路径
            java_class: 解析后的 JavaClass 对象
        """
        logger.info(f"[IntelligenceAnalysisDialog] Analyzing code: {file_path}")
        
        try:
            # 1. 语义分析
            semantic_result = self.semantic_analyzer.analyze_file(file_path, java_class)
            
            # 2. 更新 UI
            self._update_semantic_tab(semantic_result)
            self._update_scenarios_tab(semantic_result)
            self._update_boundaries_tab(semantic_result)
            
            # 3. 切换到第一个标签页
            self.tabs.setCurrentIndex(0)
            
            logger.info(f"[IntelligenceAnalysisDialog] Analysis complete")
            
        except Exception as e:
            logger.error(f"[IntelligenceAnalysisDialog] Analysis failed: {e}")
            QMessageBox.critical(
                self,
                "分析失败",
                f"代码分析失败:\n{str(e)}"
            )
    
    def analyze_errors(self, error_output: str, test_code: Optional[str] = None, 
                      source_code: Optional[str] = None):
        """分析错误
        
        Args:
            error_output: 错误输出 (编译错误或测试失败)
            test_code: 可选的测试代码
            source_code: 可选的源代码
        """
        logger.info("[IntelligenceAnalysisDialog] Analyzing errors")
        
        try:
            # 1. 判断错误类型并分析
            if "Tests run" in error_output or "Failure" in error_output:
                # 测试失败
                analysis = self.root_cause_analyzer.analyze_test_failures(
                    error_output, test_code, source_code
                )
            else:
                # 编译错误
                analysis = self.root_cause_analyzer.analyze_compilation_errors(
                    error_output, source_code
                )
            
            # 2. 更新 UI
            self._update_rca_tab(analysis)
            self._update_fixes_tab(analysis)
            
            # 3. 切换到根因分析标签页
            self.tabs.setCurrentIndex(3)
            
            logger.info(f"[IntelligenceAnalysisDialog] Error analysis complete")
            
        except Exception as e:
            logger.error(f"[IntelligenceAnalysisDialog] Error analysis failed: {e}")
            QMessageBox.critical(
                self,
                "分析失败",
                f"错误分析失败:\n{str(e)}"
            )
    
    def _update_semantic_tab(self, semantic_result: Dict[str, Any]):
        """更新语义分析标签页"""
        self.code_tree.clear()
        
        # 添加类节点
        class_name = semantic_result.get("class_name", "Unknown")
        class_item = QTreeWidgetItem(self.code_tree)
        class_item.setText(0, class_name)
        class_item.setText(1, "Class")
        class_item.setText(2, "-")
        
        # 添加方法节点
        call_graph = semantic_result.get("call_graph", {})
        nodes = call_graph.get("nodes", [])
        
        for node in nodes:
            method_item = QTreeWidgetItem(class_item)
            method_item.setText(0, node.get("method", "unknown"))
            method_item.setText(1, "Method")
            method_item.setText(2, str(node.get("complexity", 1)))
            
            # 根据复杂度设置颜色
            complexity = node.get("complexity", 1)
            if complexity > 5:
                method_item.setForeground(2, QColor(255, 0, 0))  # 红色 - 高复杂度
            elif complexity > 3:
                method_item.setForeground(2, QColor(255, 165, 0))  # 橙色 - 中等复杂度
            else:
                method_item.setForeground(2, QColor(0, 128, 0))  # 绿色 - 低复杂度
        
        class_item.setExpanded(True)
        
        # 更新业务逻辑信息
        business_logic = semantic_result.get("business_logic", [])
        if business_logic:
            logic_text = "<h3>业务逻辑分析</h3>"
            for bl in business_logic:
                logic_text += f"""
                <div style="margin: 10px; padding: 10px; border-left: 3px solid #4CAF50;">
                    <strong>方法:</strong> {bl.get('method', 'N/A')}<br>
                    <strong>类型:</strong> {bl.get('type', 'N/A')}<br>
                    <strong>描述:</strong> {bl.get('description', 'N/A')}<br>
                    <strong>前置条件:</strong> {', '.join(bl.get('preconditions', [])) or '无'}<br>
                    <strong>后置条件:</strong> {', '.join(bl.get('postconditions', [])) or '无'}<br>
                    <strong>副作用:</strong> {', '.join(bl.get('side_effects', [])) or '无'}
                </div>
                """
            self.business_logic_text.setHtml(logic_text)
        else:
            self.business_logic_text.setHtml("<p>未识别到业务逻辑</p>")
    
    def _update_scenarios_tab(self, semantic_result: Dict[str, Any]):
        """更新测试场景标签页"""
        self.scenarios_tree.clear()
        
        scenarios = semantic_result.get("test_scenarios", [])
        
        # 统计
        total = len(scenarios)
        normal = sum(1 for s in scenarios if s.get("type") == "normal")
        edge = sum(1 for s in scenarios if s.get("type") == "edge")
        exception = sum(1 for s in scenarios if s.get("type") == "exception")
        
        self.total_scenarios_label.setText(f"总场景数：{total}")
        self.normal_scenarios_label.setText(f"正常场景：{normal}")
        self.edge_scenarios_label.setText(f"边界场景：{edge}")
        self.exception_scenarios_label.setText(f"异常场景：{exception}")
        
        # 添加场景到树
        for scenario in scenarios:
            item = QTreeWidgetItem(self.scenarios_tree)
            item.setText(0, scenario.get("description", "Unknown"))
            item.setText(1, scenario.get("target", "N/A"))
            item.setText(2, scenario.get("type", "normal"))
            item.setText(3, str(scenario.get("priority", 3)))
            item.setText(4, scenario.get("expected", ""))
            
            # 根据类型设置颜色
            scenario_type = scenario.get("type", "normal")
            if scenario_type == "normal":
                item.setForeground(0, QColor(0, 128, 0))  # 绿色
            elif scenario_type == "edge":
                item.setForeground(0, QColor(255, 165, 0))  # 橙色
            elif scenario_type == "exception":
                item.setForeground(0, QColor(255, 0, 0))  # 红色
        
        self.scenarios_tree.resizeColumnToContents(0)
    
    def _update_boundaries_tab(self, semantic_result: Dict[str, Any]):
        """更新边界条件标签页"""
        self.boundaries_tree.clear()
        
        boundaries = semantic_result.get("boundary_conditions", [])
        
        # 统计
        null_checks = sum(1 for b in boundaries if b.get("type") == "NULL_CHECK")
        empty_checks = sum(1 for b in boundaries if b.get("type") == "EMPTY_CHECK")
        range_checks = sum(1 for b in boundaries if b.get("type") == "RANGE_CHECK")
        
        self.null_checks_label.setText(f"🔴 空值检查：{null_checks}")
        self.empty_checks_label.setText(f"🟡 空集合检查：{empty_checks}")
        self.range_checks_label.setText(f"🟢 范围检查：{range_checks}")
        
        # 添加边界条件到树
        for boundary in boundaries:
            item = QTreeWidgetItem(self.boundaries_tree)
            item.setText(0, boundary.get("parameter", "N/A"))
            item.setText(1, boundary.get("type", "UNKNOWN"))
            item.setText(2, str(boundary.get("test_value", "N/A")))
            item.setText(3, boundary.get("expected_behavior", "N/A"))
            
            # 根据类型设置图标颜色
            boundary_type = boundary.get("type", "")
            if "NULL" in boundary_type:
                item.setForeground(0, QColor(255, 0, 0))
            elif "EMPTY" in boundary_type:
                item.setForeground(0, QColor(255, 165, 0))
            elif "RANGE" in boundary_type:
                item.setForeground(0, QColor(0, 128, 0))
        
        self.boundaries_tree.resizeColumnToContents(0)
    
    def _update_rca_tab(self, analysis: RootCauseAnalysis):
        """更新根因分析标签页"""
        self.root_causes_tree.clear()
        
        # 更新置信度
        confidence = int(analysis.confidence_score * 100)
        self.confidence_bar.setValue(confidence)
        self.confidence_label.setText(f"{confidence}%")
        
        # 设置颜色
        if confidence >= 80:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        elif confidence >= 60:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }")
        else:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
        
        # 添加根因
        for cause in analysis.root_causes:
            item = QTreeWidgetItem(self.root_causes_tree)
            item.setText(0, cause.description)
            item.setText(1, cause.category.name)
            item.setText(2, f"{cause.confidence:.0%}")
            item.setText(3, cause.location or "N/A")
            item.setText(4, ", ".join(cause.contributing_factors))
            
            # 根据置信度设置颜色
            conf = int(cause.confidence * 100)
            if conf >= 80:
                item.setForeground(0, QColor(0, 128, 0))
            elif conf >= 60:
                item.setForeground(0, QColor(255, 165, 0))
            else:
                item.setForeground(0, QColor(255, 0, 0))
        
        # 显示证据
        if analysis.root_causes:
            evidence = analysis.root_causes[0].evidence
            evidence_text = "\n".join([f"• {e}" for e in evidence])
            self.evidence_text.setPlainText(evidence_text)
        else:
            self.evidence_text.setPlainText("无证据")
    
    def _update_fixes_tab(self, analysis: RootCauseAnalysis):
        """更新修复建议标签页"""
        self.fixes_tree.clear()
        
        for fix in analysis.suggested_fixes:
            item = QTreeWidgetItem(self.fixes_tree)
            item.setText(0, fix.strategy_id)
            item.setText(1, fix.strategy_type.name)
            item.setText(2, str(fix.priority))
            item.setText(3, fix.estimated_effort)
            item.setText(4, f"{fix.success_probability:.0%}")
            item.setText(5, fix.description)
            
            # 根据优先级设置颜色
            priority = fix.priority
            if priority == 1:
                item.setForeground(0, QColor(255, 0, 0))  # 红色 - 高优先级
            elif priority == 2:
                item.setForeground(0, QColor(255, 165, 0))  # 橙色 - 中优先级
            else:
                item.setForeground(0, QColor(0, 128, 0))  # 绿色 - 低优先级
        
        self.fixes_tree.resizeColumnToContents(0)
    
    def _on_scenario_clicked(self, item: QTreeWidgetItem, column: int):
        """场景点击事件"""
        if item.columnCount() > 0:
            description = item.text(0)
            target = item.text(1)
            scenario_type = item.text(2)
            
            details = f"""
<h3>场景详情</h3>
<p><strong>描述:</strong> {description}</p>
<p><strong>目标方法:</strong> {target}</p>
<p><strong>类型:</strong> {scenario_type}</p>
<p><strong>测试步骤:</strong></p>
<ul>
    <li>准备测试数据</li>
    <li>调用目标方法</li>
    <li>验证结果</li>
</ul>
            """
            self.scenario_details.setHtml(details)
    
    def _on_fix_clicked(self, item: QTreeWidgetItem, column: int):
        """修复建议点击事件"""
        strategy_id = item.text(0)
        strategy_type = item.text(1)
        priority = item.text(2)
        effort = item.text(3)
        success = item.text(4)
        description = item.text(5)
        
        details = f"""
<h3>修复策略详情</h3>
<p><strong>策略 ID:</strong> {strategy_id}</p>
<p><strong>类型:</strong> {strategy_type}</p>
<p><strong>优先级:</strong> {priority}</p>
<p><strong>预估工作量:</strong> {effort}</p>
<p><strong>成功概率:</strong> {success}</p>
<p><strong>描述:</strong> {description}</p>
<p><strong>建议操作:</strong></p>
<ol>
    <li>审查推荐的修复策略</li>
    <li>定位需要修改的代码位置</li>
    <li>应用修复</li>
    <li>运行测试验证修复效果</li>
</ol>
        """
        self.fix_details.setHtml(details)
    
    def _on_refresh(self):
        """刷新按钮点击事件"""
        # 这里可以重新执行分析
        QMessageBox.information(
            self,
            "刷新",
            "分析结果已刷新"
        )
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            "semantic_analysis": {
                "test_scenarios_count": self.scenarios_tree.topLevelItemCount(),
                "boundary_conditions_count": self.boundaries_tree.topLevelItemCount()
            },
            "error_analysis": {
                "root_causes_count": self.root_causes_tree.topLevelItemCount(),
                "fix_suggestions_count": self.fixes_tree.topLevelItemCount(),
                "confidence": self.confidence_bar.value()
            }
        }


def show_intelligence_analysis(parent=None):
    """显示智能分析对话框"""
    dialog = IntelligenceAnalysisDialog(parent)
    return dialog
