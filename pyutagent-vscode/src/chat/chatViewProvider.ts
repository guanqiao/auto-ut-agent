import * as vscode from 'vscode';
import { PyUTAgentAPI } from '../backend/apiClient';

/**
 * Chat View Provider - 侧边栏聊天面板
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'pyutagent.chatView';
    
    private _view?: vscode.WebviewView;
    
    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _api: PyUTAgentAPI
    ) {}
    
    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;
        
        // 配置 Webview
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        
        // 设置 HTML 内容
        webviewView.webview.html = this._getHtmlForWebview(webviewView);
        
        // 监听消息
        webviewView.webview.onDidReceiveMessage((message) => {
            this._handleMessage(message);
        });
    }
    
    /**
     * 生成 Webview HTML
     */
    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PyUT Agent Chat</title>
                <style>
                    :root {
                        --vscode-font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
                        --vscode-editor-background: var(--vscode-editor-background, #1e1e1e);
                        --vscode-button-background: var(--vscode-button-background, #0e639c);
                        --vscode-foreground: var(--vscode-foreground, #cccccc);
                    }
                    
                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }
                    
                    body {
                        padding: 10px;
                        font-family: var(--vscode-font-family);
                        background: var(--vscode-editor-background);
                        color: var(--vscode-foreground);
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                    }
                    
                    #header {
                        padding: 10px;
                        border-bottom: 1px solid #333;
                        margin-bottom: 10px;
                    }
                    
                    #header h2 {
                        font-size: 16px;
                        font-weight: normal;
                    }
                    
                    #messages {
                        flex: 1;
                        overflow-y: auto;
                        padding: 10px;
                    }
                    
                    .message {
                        margin: 10px 0;
                        padding: 8px 12px;
                        border-radius: 4px;
                        max-width: 90%;
                        word-wrap: break-word;
                    }
                    
                    .message.user {
                        background: var(--vscode-button-background);
                        color: white;
                        margin-left: auto;
                    }
                    
                    .message.agent {
                        background: rgba(255, 255, 255, 0.1);
                    }
                    
                    .message.system {
                        background: rgba(255, 165, 0, 0.2);
                        font-style: italic;
                    }
                    
                    .input-area {
                        display: flex;
                        gap: 5px;
                        padding: 10px;
                        border-top: 1px solid #333;
                    }
                    
                    #input {
                        flex: 1;
                        padding: 8px;
                        border: 1px solid #333;
                        border-radius: 3px;
                        background: var(--vscode-editor-background);
                        color: var(--vscode-foreground);
                        font-family: var(--vscode-font-family);
                    }
                    
                    #input:focus {
                        outline: 1px solid var(--vscode-button-background);
                    }
                    
                    #send {
                        padding: 8px 20px;
                        background: var(--vscode-button-background);
                        color: white;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                    }
                    
                    #send:hover {
                        opacity: 0.9;
                    }
                    
                    #send:disabled {
                        opacity: 0.5;
                        cursor: not-allowed;
                    }
                    
                    .typing-indicator {
                        display: inline-block;
                        width: 20px;
                        height: 10px;
                        animation: typing 1s infinite;
                    }
                    
                    @keyframes typing {
                        0%, 100% { opacity: 0.3; }
                        50% { opacity: 1; }
                    }
                    
                    pre {
                        background: rgba(0, 0, 0, 0.3);
                        padding: 8px;
                        border-radius: 3px;
                        overflow-x: auto;
                        margin: 5px 0;
                    }
                    
                    code {
                        font-family: 'Consolas', 'Courier New', monospace;
                    }
                </style>
            </head>
            <body>
                <div id="header">
                    <h2>PyUT Agent Chat</h2>
                </div>
                
                <div id="messages">
                    <div class="message agent">
                        👋 Hello! I'm your PyUT Agent assistant. How can I help you today?
                    </div>
                </div>
                
                <div class="input-area">
                    <input 
                        id="input" 
                        placeholder="Type a message..." 
                        autocomplete="off"
                    />
                    <button id="send">Send</button>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    const messagesDiv = document.getElementById('messages');
                    const input = document.getElementById('input');
                    const sendButton = document.getElementById('send');
                    
                    // 发送消息
                    function sendMessage() {
                        const message = input.value.trim();
                        if (message) {
                            // 添加用户消息到界面
                            addMessage(message, 'user');
                            
                            // 发送到扩展
                            vscode.postMessage({
                                type: 'chat_message',
                                content: message
                            });
                            
                            input.value = '';
                            sendButton.disabled = true;
                        }
                    }
                    
                    // 添加消息到界面
                    function addMessage(content, role) {
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message ' + role;
                        msgDiv.textContent = content;
                        messagesDiv.appendChild(msgDiv);
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }
                    
                    // 监听返回消息
                    window.addEventListener('message', event => {
                        const message = event.data;
                        if (message.type === 'response') {
                            addMessage(message.content, 'agent');
                            sendButton.disabled = false;
                        } else if (message.type === 'system') {
                            addMessage(message.content, 'system');
                        }
                    });
                    
                    // 事件监听
                    sendButton.onclick = sendMessage;
                    input.onkeypress = (e) => {
                        if (e.key === 'Enter') {
                            sendMessage();
                        }
                    };
                </script>
            </body>
            </html>
        `;
    }
    
    /**
     * 处理接收到的消息
     */
    private _handleMessage(message: any) {
        switch (message.type) {
            case 'chat_message':
                // 处理聊天消息
                this._processChatMessage(message.content);
                break;
        }
    }
    
    /**
     * 处理聊天消息
     */
    private async _processChatMessage(content: string) {
        // TODO: 调用后端 API 处理消息
        // 这里是示例响应
        this.sendMessage({
            type: 'response',
            content: 'Received your message: ' + content
        });
    }
    
    /**
     * 发送消息到 Webview
     */
    public sendMessage(message: any) {
        this._view?.webview.postMessage(message);
    }
    
    /**
     * 显示系统消息
     */
    public showSystemMessage(content: string) {
        this.sendMessage({
            type: 'system',
            content: content
        });
    }
}
