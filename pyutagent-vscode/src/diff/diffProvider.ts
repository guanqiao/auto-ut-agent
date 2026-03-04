import * as vscode from 'vscode';
import { join } from 'path';

/**
 * Diff 预览结果
 */
export interface DiffResult {
    action: 'accept' | 'reject' | 'cancel';
}

/**
 * Diff 预览提供者
 */
export class DiffViewProvider {
    private panel: vscode.WebviewPanel | undefined;
    private extensionUri: vscode.Uri;

    constructor(extensionUri: vscode.Uri) {
        this.extensionUri = extensionUri;
    }

    /**
     * 显示代码差异预览
     */
    public async showDiff(
        originalCode: string,
        modifiedCode: string,
        language: string = 'java',
        title: string = 'Preview Changes'
    ): Promise<DiffResult | undefined> {
        // 创建面板
        this.panel = vscode.window.createWebviewPanel(
            'pyutagent.diff',
            title,
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [this.extensionUri]
            }
        );

        // 设置 HTML
        this.panel.webview.html = this._getHtmlContent();

        // 发送代码数据
        this.panel.webview.postMessage({
            type: 'setCode',
            originalCode,
            modifiedCode,
            language
        });

        // 监听用户操作
        return new Promise((resolve) => {
            // 监听消息
            const disposable = this.panel!.webview.onDidReceiveMessage(
                (message) => {
                    switch (message.type) {
                        case 'accept':
                            resolve({ action: 'accept' });
                            disposable.dispose();
                            break;
                        case 'reject':
                            resolve({ action: 'reject' });
                            disposable.dispose();
                            break;
                        case 'cancel':
                            resolve({ action: 'cancel' });
                            disposable.dispose();
                            break;
                    }
                }
            );

            // 面板关闭
            this.panel!.onDidDispose(() => {
                resolve({ action: 'cancel' });
            });
        });
    }

    /**
     * 获取 HTML 内容
     */
    private _getHtmlContent(): string {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src vscode-resource: 'unsafe-inline'; script-src vscode-resource: 'unsafe-inline';">
                <title>Diff Preview</title>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        padding: 10px;
                        background: var(--vscode-editor-background, #1e1e1e);
                        color: var(--vscode-foreground, #cccccc);
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                    }
                    
                    #header {
                        padding: 10px 0;
                        border-bottom: 1px solid var(--vscode-widget-border, #333);
                        margin-bottom: 10px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    
                    #header h2 {
                        font-size: 16px;
                        font-weight: normal;
                    }
                    
                    #stats {
                        font-size: 12px;
                        color: var(--vscode-descriptionForeground, #cccccc80);
                    }
                    
                    #diff-container {
                        flex: 1;
                        border: 1px solid var(--vscode-widget-border, #333);
                        min-height: 400px;
                    }
                    
                    #buttons {
                        margin-top: 15px;
                        display: flex;
                        justify-content: center;
                        gap: 15px;
                    }
                    
                    button {
                        padding: 10px 25px;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        transition: opacity 0.2s;
                    }
                    
                    button:hover {
                        opacity: 0.9;
                    }
                    
                    #accept {
                        background: var(--vscode-button-background, #0e639c);
                        color: white;
                    }
                    
                    #reject {
                        background: var(--vscode-button-secondaryBackground, #333);
                        color: white;
                    }
                    
                    #cancel {
                        background: transparent;
                        color: var(--vscode-foreground, #cccccc);
                        border: 1px solid var(--vscode-widget-border, #333);
                    }
                    
                    .loading {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100%;
                        font-size: 14px;
                        color: var(--vscode-descriptionForeground, #cccccc80);
                    }
                    
                    .loading::after {
                        content: '';
                        width: 20px;
                        height: 20px;
                        margin-left: 10px;
                        border: 2px solid var(--vscode-progressBarBackground, #0e639c);
                        border-top-color: transparent;
                        border-radius: 50%;
                        animation: spin 0.8s linear infinite;
                    }
                    
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div id="header">
                    <h2 id="title">Preview Changes</h2>
                    <div id="stats"></div>
                </div>
                
                <div id="diff-container">
                    <div class="loading">Loading Monaco Editor...</div>
                </div>
                
                <div id="buttons">
                    <button id="cancel">Cancel</button>
                    <button id="reject">Reject Changes</button>
                    <button id="accept">✓ Accept Changes</button>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    let monacoLoaded = false;
                    
                    // 加载 Monaco Editor
                    function loadMonaco() {
                        const monacoPath = vscode.getResourcePath('node_modules/monaco-editor/min/vs');
                        require.config({ paths: { vs: monacoPath } });
                        require(['vs/editor/editor.main'], function() {
                            monacoLoaded = true;
                            initDiffEditor();
                        });
                    }
                    
                    // 初始化 Diff 编辑器
                    function initDiffEditor() {
                        if (!window.monaco) {
                            console.error('Monaco not loaded');
                            return;
                        }
                        
                        const container = document.getElementById('diff-container');
                        container.innerHTML = '';
                        
                        const diffEditor = monaco.editor.createDiffEditor(container, {
                            originalEditable: false,
                            renderSideBySide: true,
                            theme: 'vs-dark',
                            automaticLayout: true,
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: 'on',
                            wordWrap: 'off',
                            readOnly: true,
                            scrollBeyondLastLine: false
                        });
                        
                        // 监听按钮点击
                        document.getElementById('accept').onclick = () => {
                            vscode.postMessage({ type: 'accept' });
                        };
                        
                        document.getElementById('reject').onclick = () => {
                            vscode.postMessage({ type: 'reject' });
                        };
                        
                        document.getElementById('cancel').onclick = () => {
                            vscode.postMessage({ type: 'cancel' });
                        };
                    }
                    
                    // 监听代码设置消息
                    window.addEventListener('message', (event) => {
                        const data = event.data;
                        if (data.type === 'setCode') {
                            if (monacoLoaded) {
                                const originalModel = monaco.editor.createModel(
                                    data.originalCode,
                                    data.language
                                );
                                const modifiedModel = monaco.editor.createModel(
                                    data.modifiedCode,
                                    data.language
                                );
                                
                                const editor = document.querySelector('.monaco-diff-editor');
                                if (editor) {
                                    const diffEditor = monaco.editor.getDiffEditors().find(e => 
                                        e.getContainerDomNode() === editor
                                    );
                                    if (diffEditor) {
                                        diffEditor.setModel({
                                            original: originalModel,
                                            modified: modifiedModel
                                        });
                                        
                                        // 更新统计信息
                                        const changes = diffEditor.getLineChanges();
                                        const additions = changes.reduce((sum, c) => sum + (c.modifiedEnd - c.modifiedStart + 1), 0);
                                        const deletions = changes.reduce((sum, c) => sum + (c.originalEnd - c.originalStart + 1), 0);
                                        
                                        document.getElementById('stats').textContent = 
                                            '+' + additions + ' -' + deletions + ' lines';
                                    }
                                }
                            }
                        }
                    });
                    
                    // 延迟加载 Monaco
                    setTimeout(loadMonaco, 100);
                </script>
            </body>
            </html>
        `;
    }

    /**
     * 关闭面板
     */
    public dispose() {
        this.panel?.dispose();
        this.panel = undefined;
    }
}
