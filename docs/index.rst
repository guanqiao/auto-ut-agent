PyUTAgent 文档
================

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   核心模块 <core/modules>
   LLM 模块 <llm/modules>
   Agent 模块 <agent/modules>
   使用示例 <examples/basic_usage>
   最佳实践 <best_practices/architecture>

PyUTAgent 简介
==============

PyUTAgent 是一个智能单元测试生成代理系统，使用 AI 技术自动生成高质量的单元测试代码。

主要特性
--------

* **智能测试生成**: 基于代码分析自动生成测试用例
* **增量修复**: 自动修复失败的测试，减少 LLM 调用
* **多级缓存**: L1+L2 缓存系统，提升性能 5-10 倍
* **组件化架构**: 高度可扩展的组件系统
* **性能监控**: 完整的指标收集和性能追踪

快速开始
--------

.. code-block:: python

   from pyutagent.core import MetricsCollector
   from pyutagent.llm import MultiLevelCache
   
   # 使用指标收集器
   collector = MetricsCollector()
   collector.record_counter("requests", 1)
   
   # 使用多级缓存
   cache = MultiLevelCache()
   await cache.put("key", "value")
   value = await cache.get("key")

安装
----

.. code-block:: bash

   pip install pyutagent

模块文档
--------

核心模块
~~~~~~~~

* :doc:`core/event_bus` - 事件总线系统
* :doc:`core/state_store` - 状态管理
* :doc:`core/message_bus` - 消息总线
* :doc:`core/component_registry` - 组件注册表
* :doc:`core/metrics` - 性能监控和指标
* :doc:`core/error_handling` - 错误处理
* :doc:`core/actions` - Action 系统扩展

LLM 模块
~~~~~~~~

* :doc:`llm/client` - LLM 客户端
* :doc:`llm/prompt_cache` - Prompt 缓存
* :doc:`llm/multi_level_cache` - 多级缓存

Agent 模块
~~~~~~~~~~

* :doc:`agent/incremental_fixer` - 增量修复器
* :doc:`agent/smart_clusterer` - 智能聚类算法

索引和表格
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
