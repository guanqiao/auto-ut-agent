/**
 * VS Code Extension 集成测试
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { PyUTAgentAPI } from '../src/backend/apiClient';
import { DiffViewProvider } from '../src/diff/diffProvider';
import { TerminalManager } from '../src/terminal/terminalManager';

suite('PyUT Agent Integration Tests', () => {
    let api: PyUTAgentAPI;
    let terminalManager: TerminalManager;

    suiteSetup(() => {
        api = new PyUTAgentAPI('http://localhost:8000');
        terminalManager = new TerminalManager();
    });

    suiteTeardown(() => {
        terminalManager.dispose();
    });

    /**
     * 测试 1: API 健康检查
     */
    test('API should respond to health check', async () => {
        // 注意：这个测试需要后端服务运行
        const isHealthy = await api.healthCheck();
        // 如果后端未运行，测试会跳过
        if (!isHealthy) {
            console.log('Backend not running, skipping health check test');
            return;
        }
        assert.strictEqual(isHealthy, true);
    });

    /**
     * 测试 2: API 客户端初始化
     */
    test('API client should initialize correctly', () => {
        const testApi = new PyUTAgentAPI('http://localhost:8000');
        assert.ok(testApi);
    });

    /**
     * 测试 3: 终端管理器创建
     */
    test('Terminal manager should create terminal', () => {
        const manager = new TerminalManager();
        assert.ok(manager);
        manager.dispose();
    });

    /**
     * 测试 4: 终端执行命令
     */
    test('Terminal should execute commands', async () => {
        const manager = new TerminalManager();
        
        try {
            const result = await manager.executeCommand('echo "test"', {
                showOutput: false,
                timeout: 5000
            });
            
            assert.strictEqual(result.success, true);
            assert.ok(result.stdout.includes('test'));
        } finally {
            manager.dispose();
        }
    });

    /**
     * 测试 5: 终端超时处理
     */
    test('Terminal should handle timeout', async () => {
        const manager = new TerminalManager();
        
        try {
            const result = await manager.executeCommand('sleep 10', {
                showOutput: false,
                timeout: 1000 // 1 秒超时
            });
            
            assert.strictEqual(result.success, false);
            assert.ok(result.stderr.includes('timed out') || result.exitCode === null);
        } finally {
            manager.dispose();
        }
    });

    /**
     * 测试 6: Diff 预览创建
     */
    test('Diff view should create correctly', () => {
        const extensionUri = vscode.Uri.file(__dirname);
        const diffProvider = new DiffViewProvider(extensionUri);
        assert.ok(diffProvider);
    });

    /**
     * 测试 7: 配置读取
     */
    test('Configuration should be readable', () => {
        const config = vscode.workspace.getConfiguration('pyutagent');
        assert.ok(config);
        
        const apiUrl = config.get<string>('apiUrl');
        assert.ok(apiUrl);
    });
});
