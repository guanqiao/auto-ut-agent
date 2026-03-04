/**
 * API 客户端单元测试
 */

import * as assert from 'assert';
import { PyUTAgentAPI, GenerationResult, TaskResult } from '../src/backend/apiClient';

suite('PyUTAgentAPI Unit Tests', () => {
    let api: PyUTAgentAPI;

    setup(() => {
        api = new PyUTAgentAPI('http://localhost:8000');
    });

    /**
     * 测试：API 客户端构造函数
     */
    test('Constructor should create instance with default URL', () => {
        const defaultApi = new PyUTAgentAPI();
        assert.ok(defaultApi);
    });

    test('Constructor should create instance with custom URL', () => {
        const customApi = new PyUTAgentAPI('http://custom:9000');
        assert.ok(customApi);
    });

    /**
     * 测试：API 响应类型
     */
    test('GenerationResult should have correct structure', () => {
        const result: GenerationResult = {
            success: true,
            testCode: 'public class Test {}',
            message: 'Success'
        };
        
        assert.strictEqual(result.success, true);
        assert.strictEqual(result.testCode, 'public class Test {}');
        assert.strictEqual(result.message, 'Success');
    });

    test('TaskResult should have correct structure', () => {
        const result: TaskResult = {
            success: true,
            output: 'Test passed',
            message: 'Completed'
        };
        
        assert.strictEqual(result.success, true);
        assert.strictEqual(result.output, 'Test passed');
    });

    /**
     * 测试：错误处理
     */
    test('Should handle connection errors gracefully', async () => {
        // 注意：这个测试会失败，因为后端未运行，但应该优雅地处理错误
        const result = await api.generateTest('/path/to/Test.java');
        
        assert.strictEqual(result.success, false);
        assert.ok(result.error);
    });

    /**
     * 测试：单例模式
     */
    test('getApiInstance should return same instance', () => {
        const { getApiInstance } = require('../src/backend/apiClient');
        const instance1 = getApiInstance();
        const instance2 = getApiInstance();
        
        assert.strictEqual(instance1, instance2);
    });
});
