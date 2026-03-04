import * as vscode from 'vscode';

/**
 * Diff 文件项
 */
export interface DiffFile {
    filePath: string;
    originalContent: string;
    modifiedContent: string;
    language: string;
}

/**
 * Diff 结果
 */
export interface MultiDiffResult {
    acceptedFiles: string[];
    rejectedFiles: string[];
    cancelled: boolean;
}

/**
 * 增强版 Diff 查看器
 * 支持多文件预览、批量操作、语法高亮
 */
export class EnhancedDiffViewer {
    private panel: vscode.WebviewPanel | undefined;
    private extensionUri: vscode.Uri;
    private files: DiffFile[] = [];
    private currentIndex: number = 0;

    constructor(extensionUri: vscode.Uri) {
        this.extensionUri = extensionUri;
    }

    /**
     * 显示多文件差异预览
     */
    public async showMultiFileDiff(
        files: DiffFile[],
        title: string = 'Preview Changes'
    ): Promise<MultiDiffResult> {
        this.files = files;
        this.currentIndex = 0;

        const result: MultiDiffResult = {
            acceptedFiles: [],
            rejectedFiles: [],
            cancelled: false
        };

        if (files.length === 0) {
            return result;
        }

        // 创建面板
        this.panel = vscode.window.createWebviewPanel(
            'pyutagent.multidiff',
            title,
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [this.extensionUri]
            }
        );

        this.panel.webview.html = this._getHtmlContent(files);

        return new Promise((resolve) => {
            const disposable = this.panel!.webview.onDidReceiveMessage(
                (message) => {
                    switch (message.type) {
                        case 'accept':
                            result.acceptedFiles.push(message.filePath);
                            break;
                        case 'reject':
                            result.rejectedFiles.push(message.filePath);
                            break;
                        case 'acceptAll':
                            result.acceptedFiles = files.map(f => f.filePath);
                            disposable.dispose();
                            resolve(result);
                            break;
                        case 'rejectAll':
                            result.rejectedFiles = files.map(f => f.filePath);
                            disposable.dispose();
                            resolve(result);
                            break;
                        case 'cancel':
                            result.cancelled = true;
                            disposable.dispose();
                            resolve(result);
                            break;
                        case 'done':
                            disposable.dispose();
                            resolve(result);
                            break;
                    }
                }
            );

            this.panel!.onDidDispose(() => {
                result.cancelled = true;
                resolve(result);
            });
        });
    }

    /**
     * 显示单文件差异预览
     */
    public async showSingleDiff(
        file: DiffFile,
        title?: string
    ): Promise<'accept' | 'reject' | 'cancel'> {
        const result = await this.showMultiFileDiff([file], title || `Preview: ${file.filePath}`);
        
        if (result.cancelled) {
            return 'cancel';
        }
        if (result.acceptedFiles.length > 0) {
            return 'accept';
        }
        return 'reject';
    }

    /**
     * 获取 HTML 内容
     */
    private _getHtmlContent(files: DiffFile[]): string {
        const fileListHtml = files.map((f, i) => `
            <div class="file-item ${i === 0 ? 'active' : ''}" data-index="${i}">
                <span class="file-icon">📄</span>
                <span class="file-name">${this._getFileName(f.filePath)}</span>
                <span class="file-path">${f.filePath}</span>
            </div>
        `).join('');

        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${this.panel!.webview.cspSource} 'unsafe-inline'; script-src ${this.panel!.webview.cspSource} 'unsafe-inline';">
    <title>Diff Preview</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif);
            background: var(--vscode-editor-background, #1e1e1e);
            color: var(--vscode-foreground, #cccccc);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        #header {
            padding: 12px 16px;
            background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
            border-bottom: 1px solid var(--vscode-widget-border, #333);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        #header h2 {
            font-size: 14px;
            font-weight: 500;
        }
        
        #stats {
            font-size: 12px;
            color: var(--vscode-descriptionForeground, #858585);
        }
        
        /* Main container */
        #main {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        /* File list sidebar */
        #file-list {
            width: 250px;
            background: var(--vscode-sideBar-background, #252526);
            border-right: 1px solid var(--vscode-widget-border, #333);
            overflow-y: auto;
        }
        
        .file-item {
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid var(--vscode-widget-border, #333);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .file-item:hover {
            background: var(--vscode-list-hoverBackground, #2a2d2e);
        }
        
        .file-item.active {
            background: var(--vscode-list-activeSelectionBackground, #094771);
        }
        
        .file-icon {
            font-size: 14px;
        }
        
        .file-name {
            font-weight: 500;
            font-size: 13px;
        }
        
        .file-path {
            font-size: 11px;
            color: var(--vscode-descriptionForeground, #858585);
            margin-left: auto;
        }
        
        /* Diff container */
        #diff-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        #diff-header {
            padding: 8px 16px;
            background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
            border-bottom: 1px solid var(--vscode-widget-border, #333);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        #current-file {
            font-size: 13px;
            font-weight: 500;
        }
        
        #diff-stats {
            font-size: 12px;
        }
        
        .added { color: #4ec9b0; }
        .removed { color: #f14c4c; }
        
        #editor-container {
            flex: 1;
            min-height: 300px;
        }
        
        /* Buttons */
        #buttons {
            padding: 12px 16px;
            background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
            border-top: 1px solid var(--vscode-widget-border, #333);
            display: flex;
            justify-content: center;
            gap: 12px;
        }
        
        button {
            padding: 8px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        #accept {
            background: var(--vscode-button-background, #0e639c);
            color: white;
        }
        
        #accept:hover {
            background: var(--vscode-button-hoverBackground, #1177bb);
        }
        
        #reject {
            background: var(--vscode-button-secondaryBackground, #3c3c3c);
            color: white;
        }
        
        #acceptAll {
            background: #2ea043;
            color: white;
        }
        
        #rejectAll {
            background: #da3633;
            color: white;
        }
        
        #cancel {
            background: transparent;
            color: var(--vscode-foreground, #cccccc);
            border: 1px solid var(--vscode-widget-border, #3c3c3c);
        }
        
        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            font-size: 14px;
            color: var(--vscode-descriptionForeground, #858585);
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
        
        /* Empty state */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--vscode-descriptionForeground, #858585);
        }
        
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        /* Single file mode */
        .single-file #file-list {
            display: none;
        }
        
        .single-file #acceptAll,
        .single-file #rejectAll {
            display: none;
        }
    </style>
</head>
<body class="${files.length === 1 ? 'single-file' : ''}">
    <div id="header">
        <h2>📝 Preview Changes (${files.length} file${files.length > 1 ? 's' : ''})</h2>
        <div id="stats"></div>
    </div>
    
    <div id="main">
        <div id="file-list">
            ${fileListHtml}
        </div>
        
        <div id="diff-container">
            <div id="diff-header">
                <span id="current-file">${files[0]?.filePath || 'No file'}</span>
                <span id="diff-stats"></span>
            </div>
            <div id="editor-container">
                <div class="loading">Loading Monaco Editor...</div>
            </div>
        </div>
    </div>
    
    <div id="buttons">
        <button id="cancel">Cancel</button>
        <button id="rejectAll">Reject All</button>
        <button id="reject">Reject</button>
        <button id="accept">✓ Accept</button>
        <button id="acceptAll">Accept All</button>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        const files = ${JSON.stringify(files)};
        let currentIndex = 0;
        let diffEditor = null;
        
        // Initialize Monaco
        function initMonaco() {
            const container = document.getElementById('editor-container');
            container.innerHTML = '';
            
            // Create a simple diff view without Monaco dependency
            const diffView = document.createElement('div');
            diffView.style.cssText = 'display: flex; height: 100%; overflow: hidden;';
            
            // Original panel
            const originalPanel = document.createElement('div');
            originalPanel.style.cssText = 'flex: 1; overflow: auto; border-right: 1px solid var(--vscode-widget-border);';
            originalPanel.innerHTML = '<div style="padding: 8px; font-size: 12px; background: var(--vscode-editorGroupHeader-tabsBackground);">Original</div><pre style="padding: 12px; font-family: monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap;"></pre>';
            
            // Modified panel
            const modifiedPanel = document.createElement('div');
            modifiedPanel.style.cssText = 'flex: 1; overflow: auto;';
            modifiedPanel.innerHTML = '<div style="padding: 8px; font-size: 12px; background: var(--vscode-editorGroupHeader-tabsBackground);">Modified</div><pre style="padding: 12px; font-family: monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap;"></pre>';
            
            diffView.appendChild(originalPanel);
            diffView.appendChild(modifiedPanel);
            container.appendChild(diffView);
            
            // Show first file
            showFile(0);
        }
        
        // Show file at index
        function showFile(index) {
            if (index < 0 || index >= files.length) return;
            
            currentIndex = index;
            const file = files[index];
            
            // Update file list
            document.querySelectorAll('.file-item').forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });
            
            // Update header
            document.getElementById('current-file').textContent = file.filePath;
            
            // Update content
            const preElements = document.querySelectorAll('#editor-container pre');
            if (preElements.length >= 2) {
                preElements[0].textContent = file.originalContent;
                preElements[1].textContent = file.modifiedContent;
            }
            
            // Calculate stats
            const addedLines = file.modifiedContent.split('\\n').length - file.originalContent.split('\\n').length;
            const stats = document.getElementById('diff-stats');
            if (addedLines > 0) {
                stats.innerHTML = '<span class="added">+' + addedLines + '</span> lines';
            } else if (addedLines < 0) {
                stats.innerHTML = '<span class="removed">' + addedLines + '</span> lines';
            } else {
                stats.textContent = 'No line change';
            }
        }
        
        // File list click handler
        document.querySelectorAll('.file-item').forEach((item, index) => {
            item.addEventListener('click', () => showFile(index));
        });
        
        // Button handlers
        document.getElementById('accept').onclick = () => {
            vscode.postMessage({ type: 'accept', filePath: files[currentIndex].filePath });
            if (currentIndex < files.length - 1) {
                showFile(currentIndex + 1);
            } else {
                vscode.postMessage({ type: 'done' });
            }
        };
        
        document.getElementById('reject').onclick = () => {
            vscode.postMessage({ type: 'reject', filePath: files[currentIndex].filePath });
            if (currentIndex < files.length - 1) {
                showFile(currentIndex + 1);
            } else {
                vscode.postMessage({ type: 'done' });
            }
        };
        
        document.getElementById('acceptAll').onclick = () => {
            vscode.postMessage({ type: 'acceptAll' });
        };
        
        document.getElementById('rejectAll').onclick = () => {
            vscode.postMessage({ type: 'rejectAll' });
        };
        
        document.getElementById('cancel').onclick = () => {
            vscode.postMessage({ type: 'cancel' });
        };
        
        // Initialize
        initMonaco();
    </script>
</body>
</html>
        `;
    }

    private _getFileName(filePath: string): string {
        const parts = filePath.split('/');
        return parts[parts.length - 1];
    }

    public dispose() {
        this.panel?.dispose();
        this.panel = undefined;
    }
}
