基本使用示例
============

本章节展示 PyUTAgent 各个模块的基本使用方法。

1. 事件总线使用示例
------------------

.. code-block:: python

   import asyncio
   from pyutagent.core.event_bus import EventBus
   
   # 创建事件总线
   bus = EventBus()
   
   # 订阅事件
   def on_test_started(data):
       print(f"Test started: {data['test_name']}")
   
   bus.subscribe('test_started', on_test_started)
   
   # 发布事件
   bus.publish('test_started', {
       'test_name': 'test_example',
       'file': 'test_sample.py'
   })
   
   # 异步事件
   async def async_handler(data):
       await asyncio.sleep(0.1)
       print(f"Async event: {data}")
   
   bus.subscribe_async('async_event', async_handler)
   await bus.publish_async('async_event', {'message': 'Hello'})

2. 状态管理示例
-------------

.. code-block:: python

   from pyutagent.core.state_store import StateStore, AgentState, LifecycleState
   from pyutagent.core.state_store import UpdateIterationAction, UpdateCoverageAction
   
   # 创建状态存储
   store = StateStore()
   
   # 初始化状态
   initial_state = AgentState(
       lifecycle_state=LifecycleState.IDLE,
       current_iteration=0,
       target_coverage=0.8
   )
   store.set_state(initial_state)
   
   # 使用 Action 更新状态
   action = UpdateIterationAction(5)
   new_state = store.dispatch(action)
   
   print(f"Current iteration: {new_state.current_iteration}")
   
   # 批量 Action
   from pyutagent.core.actions import BatchAction
   
   batch = BatchAction()
   batch.add_action(UpdateIterationAction(10))
   batch.add_action(UpdateCoverageAction(0.75))
   
   final_state = store.dispatch(batch)

3. 消息总线示例
-------------

.. code-block:: python

   import asyncio
   from pyutagent.core.message_bus import MessageBus, Message, MessagePriority
   
   async def main():
       bus = MessageBus()
       
       # 订阅队列
       async def handler(message):
           print(f"Received: {message.body}")
       
       await bus.subscribe('test_queue', handler)
       
       # 发布消息
       msg = Message(
           queue_name='test_queue',
           body={'action': 'run_test'},
           priority=MessagePriority.HIGH
       )
       await bus.publish('test_queue', msg)
       
       # 消费消息
       message = await bus.consume('test_queue')
       if message:
           print(f"Consumed: {message.body}")
   
   asyncio.run(main())

4. 多级缓存示例
-------------

.. code-block:: python

   import asyncio
   from pyutagent.llm.multi_level_cache import MultiLevelCache, CacheConfig
   
   async def main():
       # 配置多级缓存
       config = CacheConfig(
           l1_capacity=1000,
           l2_storage_path='/tmp/cache',
           enable_compression=True
       )
       cache = MultiLevelCache(config)
       
       # 写入缓存
       await cache.put('prompt_1', {'result': 'test code'})
       
       # 读取缓存
       value = await cache.get('prompt_1')
       print(f"Cache hit: {value}")
       
       # 缓存统计
       stats = cache.get_stats()
       print(f"Hit rate: {stats['hit_rate']:.2%}")
       
       # 缓存预热
       warmup_data = {
           'key1': 'value1',
           'key2': 'value2'
       }
       await cache.warmup(warmup_data)
       
       await cache.cleanup()
   
   asyncio.run(main())

5. 性能监控示例
-------------

.. code-block:: python

   from pyutagent.core.metrics import (
       MetricsCollector,
       PerformanceTracker,
       record_counter,
       record_gauge
   )
   
   # 使用全局函数
   record_counter('requests_total', 1)
   record_gauge('active_connections', 10)
   
   # 使用收集器
   collector = MetricsCollector()
   collector.record_counter('errors', 1)
   collector.record_histogram('response_time', 0.05)
   
   # 导出指标
   metrics = collector.export_metrics()
   print(metrics)
   
   # 性能追踪
   tracker = PerformanceTracker()
   
   tracker.start_timer('operation')
   # ... 执行操作 ...
   elapsed = tracker.stop_timer('operation')
   
   # 装饰器方式
   @tracker.record_execution_time('function_name')
   def my_function():
       pass
   
   # 获取统计信息
   stats = tracker.get_metrics()
   print(f"Average time: {stats['function_name']['avg']}")

6. 组件注册表示例
---------------

.. code-block:: python

   from pyutagent.core.component_registry import (
       ComponentRegistry,
       component,
       SimpleComponent,
       discover_components
   )
   
   # 使用装饰器注册
   @component('my_component', version='1.0')
   class MyComponent(SimpleComponent):
       def initialize(self):
           print("Initializing...")
           return super().initialize()
   
   # 手动注册
   registry = ComponentRegistry()
   registry.register('my_component', MyComponent)
   
   # 获取组件
   comp = registry.get_component('my_component')
   
   # 初始化所有组件
   registry.initialize_all()
   
   # 解析依赖
   deps = registry.resolve_dependencies('my_component')
   
   # 发现模块中的组件
   import my_module
   discover_components(my_module, registry)

7. 智能聚类示例
-------------

.. code-block:: python

   from pyutagent.agent.smart_clusterer import SmartClusterer, ClusteringConfig
   from pyutagent.agent.incremental_fixer import TestFailure
   
   # 创建聚类器
   config = ClusteringConfig(similarity_threshold=0.7)
   clusterer = SmartClusterer(config)
   
   # 创建测试失败列表
   failures = [
       TestFailure(
           test_name='test1',
           error_type='AssertionError',
           message='Expected 5 but got 3'
       ),
       TestFailure(
           test_name='test2',
           error_type='AssertionError',
           message='Expected 10 but got 7'
       ),
       TestFailure(
           test_name='test3',
           error_type='TimeoutError',
           message='Operation timed out'
       )
   ]
   
   # 聚类
   clusters = clusterer.cluster_failures(failures)
   
   for cluster in clusters:
       print(f"Cluster root cause: {cluster.root_cause}")
       print(f"Failures count: {len(cluster.failures)}")

8. Action 系统示例
---------------

.. code-block:: python

   from pyutagent.core.actions import (
       BatchAction,
       TransactionalAction,
       ConditionalAction,
       ActionSequence
   )
   from pyutagent.core.state_store import UpdateIterationAction
   
   # 批量 Action
   batch = BatchAction()
   batch.add_action(UpdateIterationAction(5))
   batch.add_action(UpdateIterationAction(10))
   
   # 事务性 Action
   transactional = TransactionalAction()
   transactional.add_action(UpdateIterationAction(5))
   # 如果失败会自动回滚
   
   # 条件 Action
   condition = lambda s: s.current_iteration < 10
   conditional = ConditionalAction(condition, UpdateIterationAction(15))
   
   # Action 序列
   sequence = ActionSequence()
   sequence.add(UpdateIterationAction(5))
   sequence.add_conditional(
       lambda s: s.current_iteration >= 5,
       UpdateIterationAction(10)
   )
   
   # 执行序列
   from pyutagent.core.state_store import AgentState
   state = AgentState(current_iteration=0)
   new_state = sequence.execute(state)

9. 错误处理示例
-------------

.. code-block:: python

   from pyutagent.core.error_handling import (
       ErrorTracker,
       ErrorPropagationChain,
       RecoveryStrategy,
       RecoveryAction
   )
   from pyutagent.core.error_types import ErrorCategory, ErrorSeverity
   
   # 创建错误追踪器
   tracker = ErrorTracker()
   
   # 记录错误
   tracker.track_error(
       category=ErrorCategory.TEST_FAILURE,
       severity=ErrorSeverity.MEDIUM,
       message="Test failed"
   )
   
   # 获取错误频率
   freq = tracker.get_error_frequency(ErrorCategory.TEST_FAILURE)
   
   # 错误传播链
   chain = ErrorPropagationChain()
   
   # 添加恢复策略
   strategy = RecoveryStrategy(
       category=ErrorCategory.TEST_FAILURE,
       action=RecoveryAction.RETRY,
       max_retries=3
   )
   chain.add_recovery_strategy(strategy)
   
   # 处理错误
   success = chain.handle_error(error)

下一步
----

查看 :doc:`../best_practices/architecture` 了解更多最佳实践。
