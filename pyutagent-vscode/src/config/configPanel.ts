import * as vscode from 'vscode';

/**
 * 配置管理面板
 */
export class ConfigPanel {
    public static currentPanel: ConfigPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    public static readonly viewType = 'pyutagent.config';

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // 如果面板已存在，显示它
        if (ConfigPanel.currentPanel) {
            ConfigPanel.currentPanel._panel.reveal(column);
            return;
        }

        // 创建新面板
        const panel = vscode.window.createWebviewPanel(
            ConfigPanel.viewType,
            'PyUT Agent Configuration',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );

        ConfigPanel.currentPanel = new ConfigPanel(panel, extensionUri);
    }

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        // 设置 HTML
        this._update();

        // 监听面板关闭
        this._panel.onDidDispose(
            () => this.dispose(),
            null,
            this._disposables
        );

        // 监听配置变化
        this._panel.onDidChangeViewState(
            () => {
                if (this._panel.visible) {
                    this._update();
                }
            },
            null,
            this._disposables
        );

        // 监听 Webview 消息
        this._panel.webview.onDidReceiveMessage(
            (message) => {
                switch (message.type) {
                    case 'saveConfig':
                        this._saveConfiguration(message.config);
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    /**
     * 更新面板内容
     */
    private _update() {
        const config = vscode.workspace.getConfiguration('pyutagent');
        
        this._panel.webview.html = this._getHtmlForWebview({
            apiUrl: config.get<string>('apiUrl', 'http://localhost:8000'),
            autoApprove: config.get<boolean>('autoApprove', false),
            mode: config.get<string>('mode', 'interactive'),
            timeout: config.get<number>('timeout', 30000),
            maxRetries: config.get<number>('maxRetries', 3)
        });
    }

    /**
     * 生成配置面板 HTML
     */
    private _getHtmlForWebview(config: any): string {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src vscode-resource: 'unsafe-inline'; script-src vscode-resource: 'unsafe-inline';">
                <title>PyUT Agent Configuration</title>
                <style>
                    :root {
                        --vscode-font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
                        --vscode-editor-background: var(--vscode-editor-background, #1e1e1e);
                        --vscode-foreground: var(--vscode-foreground, #cccccc);
                        --vscode-input-background: var(--vscode-input-background, #2b2b2b);
                        --vscode-input-foreground: var(--vscode-input-foreground, #cccccc);
                        --vscode-button-background: var(--vscode-button-background, #0e639c);
                        --vscode-widget-border: var(--vscode-widget-border, #333);
                    }
                    
                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }
                    
                    body {
                        font-family: var(--vscode-font-family);
                        background: var(--vscode-editor-background);
                        color: var(--vscode-foreground);
                        padding: 20px;
                        line-height: 1.6;
                    }
                    
                    h1 {
                        font-size: 24px;
                        font-weight: 600;
                        margin-bottom: 10px;
                        padding-bottom: 10px;
                        border-bottom: 1px solid var(--vscode-widget-border);
                    }
                    
                    .description {
                        color: #888;
                        margin-bottom: 20px;
                        font-size: 13px;
                    }
                    
                    .form-group {
                        margin-bottom: 20px;
                    }
                    
                    label {
                        display: block;
                        font-size: 13px;
                        font-weight: 500;
                        margin-bottom: 6px;
                    }
                    
                    .description-text {
                        font-size: 12px;
                        color: #888;
                        margin-top: 4px;
                    }
                    
                    input[type="text"],
                    input[type="number"],
                    select {
                        width: 100%;
                        padding: 8px 12px;
                        border: 1px solid var(--vscode-widget-border);
                        border-radius: 4px;
                        background: var(--vscode-input-background);
                        color: var(--vscode-input-foreground);
                        font-family: var(--vscode-font-family);
                        font-size: 13px;
                    }
                    
                    input:focus,
                    select:focus {
                        outline: 2px solid var(--vscode-button-background);
                        border-color: transparent;
                    }
                    
                    .checkbox-group {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    input[type="checkbox"] {
                        width: 16px;
                        height: 16px;
                        cursor: pointer;
                    }
                    
                    .button-group {
                        margin-top: 30px;
                        display: flex;
                        gap: 10px;
                    }
                    
                    button {
                        padding: 10px 24px;
                        border: none;
                        border-radius: 4px;
                        font-size: 13px;
                        font-weight: 500;
                        cursor: pointer;
                        transition: all 0.2s;
                    }
                    
                    button.primary {
                        background: var(--vscode-button-background);
                        color: white;
                    }
                    
                    button.secondary {
                        background: transparent;
                        color: var(--vscode-foreground);
                        border: 1px solid var(--vscode-widget-border);
                    }
                    
                    button:hover {
                        opacity: 0.9;
                        transform: translateY(-1px);
                    }
                    
                    .status-message {
                        margin-top: 15px;
                        padding: 10px;
                        border-radius: 4px;
                        font-size: 13px;
                        display: none;
                    }
                    
                    .status-message.success {
                        background: rgba(76, 175, 80, 0.15);
                        color: #4CAF50;
                        display: block;
                    }
                    
                    .status-message.error {
                        background: rgba(244, 67, 54, 0.15);
                        color: #f44336;
                        display: block;
                    }
                </style>
            </head>
            <body>
                <h1>⚙️ PyUT Agent Configuration</h1>
                <p class="description">Configure your PyUT Agent settings</p>
                
                <form id="configForm">
                    <div class="form-group">
                        <label for="apiUrl">API URL</label>
                        <input 
                            type="text" 
                            id="apiUrl" 
                            name="apiUrl"
                            value="${config.apiUrl}"
                            placeholder="http://localhost:8000"
                        />
                        <div class="description-text">
                            Backend API endpoint URL
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="mode">Operating Mode</label>
                        <select id="mode" name="mode">
                            <option value="autonomous" ${config.mode === 'autonomous' ? 'selected' : ''}>
                                Autonomous (Auto-execute)
                            </option>
                            <option value="interactive" ${config.mode === 'interactive' ? 'selected' : ''}>
                                Interactive (Ask for approval)
                            </option>
                            <option value="supervised" ${config.mode === 'supervised' ? 'selected' : ''}>
                                Supervised (Step-by-step approval)
                            </option>
                        </select>
                        <div class="description-text">
                            Control how the agent executes tasks
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="timeout">Timeout (ms)</label>
                        <input 
                            type="number" 
                            id="timeout" 
                            name="timeout"
                            value="${config.timeout}"
                            min="5000"
                            max="300000"
                            step="1000"
                        />
                        <div class="description-text">
                            Maximum execution time for tasks (5000-300000ms)
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="maxRetries">Max Retries</label>
                        <input 
                            type="number" 
                            id="maxRetries" 
                            name="maxRetries"
                            value="${config.maxRetries}"
                            min="0"
                            max="10"
                        />
                        <div class="description-text">
                            Number of retry attempts on failure (0-10)
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input 
                                type="checkbox" 
                                id="autoApprove" 
                                name="autoApprove"
                                ${config.autoApprove ? 'checked' : ''}
                            />
                            <label for="autoApprove">Auto-approve changes (No confirmation)</label>
                        </div>
                        <div class="description-text">
                            ⚠️ Warning: Enable only if you trust the agent completely
                        </div>
                    </div>
                    
                    <div class="button-group">
                        <button type="submit" class="primary">Save Configuration</button>
                        <button type="button" class="secondary" id="resetBtn">Reset to Defaults</button>
                    </div>
                    
                    <div id="statusMessage" class="status-message"></div>
                </form>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    const form = document.getElementById('configForm');
                    const statusMessage = document.getElementById('statusMessage');
                    const resetBtn = document.getElementById('resetBtn');
                    
                    // 保存配置
                    form.addEventListener('submit', (e) => {
                        e.preventDefault();
                        
                        const config = {
                            apiUrl: document.getElementById('apiUrl').value,
                            mode: document.getElementById('mode').value,
                            timeout: parseInt(document.getElementById('timeout').value),
                            maxRetries: parseInt(document.getElementById('maxRetries').value),
                            autoApprove: document.getElementById('autoApprove').checked
                        };
                        
                        // 验证
                        if (!config.apiUrl) {
                            showStatus('API URL is required', 'error');
                            return;
                        }
                        
                        if (config.timeout < 5000 || config.timeout > 300000) {
                            showStatus('Timeout must be between 5000 and 300000ms', 'error');
                            return;
                        }
                        
                        if (config.maxRetries < 0 || config.maxRetries > 10) {
                            showStatus('Max retries must be between 0 and 10', 'error');
                            return;
                        }
                        
                        // 发送到扩展
                        vscode.postMessage({
                            type: 'saveConfig',
                            config: config
                        });
                        
                        showStatus('Configuration saved successfully!', 'success');
                    });
                    
                    // 重置配置
                    resetBtn.addEventListener('click', () => {
                        vscode.postMessage({
                            type: 'resetConfig'
                        });
                        
                        // 设置默认值
                        document.getElementById('apiUrl').value = 'http://localhost:8000';
                        document.getElementById('mode').value = 'interactive';
                        document.getElementById('timeout').value = '30000';
                        document.getElementById('maxRetries').value = '3';
                        document.getElementById('autoApprove').checked = false;
                        
                        showStatus('Configuration reset to defaults', 'success');
                    });
                    
                    function showStatus(message, type) {
                        statusMessage.textContent = message;
                        statusMessage.className = 'status-message ' + type;
                        
                        setTimeout(() => {
                            statusMessage.className = 'status-message';
                        }, 3000);
                    }
                </script>
            </body>
            </html>
        `;
    }

    /**
     * 保存配置
     */
    private _saveConfiguration(config: any) {
        const configuration = vscode.workspace.getConfiguration('pyutagent');
        
        configuration.update('apiUrl', config.apiUrl, vscode.ConfigurationTarget.Global);
        configuration.update('mode', config.mode, vscode.ConfigurationTarget.Global);
        configuration.update('timeout', config.timeout, vscode.ConfigurationTarget.Global);
        configuration.update('maxRetries', config.maxRetries, vscode.ConfigurationTarget.Global);
        configuration.update('autoApprove', config.autoApprove, vscode.ConfigurationTarget.Global);
        
        vscode.window.showInformationMessage('PyUT Agent configuration saved successfully!');
    }

    /**
     * 释放资源
     */
    public dispose() {
        ConfigPanel.currentPanel = undefined;
        this._panel.dispose();
        
        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }
}
