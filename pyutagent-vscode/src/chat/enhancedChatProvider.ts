import * as vscode from 'vscode';
import { PyUTAgentAPI, StreamChunk } from '../backend/apiClient';

/**
 * 增强的 Chat View Provider - 支持流式输出和 Markdown 渲染
 */
export class EnhancedChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'pyutagent.chatView';
    
    private _view?: vscode.WebviewView;
    private _api: PyUTAgentAPI;
    private _messageQueue: any[] = [];
    
    constructor(
        private readonly _extensionUri: vscode.Uri,
        api: PyUTAgentAPI
    ) {
        this._api = api;
    }
    
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
        
        // 设置 HTML 内容（支持 Markdown 和流式输出）
        webviewView.webview.html = this._getEnhancedHtmlForWebview();
        
        // 监听消息
        webviewView.webview.onDidReceiveMessage((message) => {
            this._handleMessage(message);
        });
    }
    
    /**
     * 生成增强的 HTML 内容（支持 Markdown 和流式输出）
     */
    private _getEnhancedHtmlForWebview(): string {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PyUT Agent Chat</title>
                <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                <style>
                    :root {
                        --vscode-font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
                        --vscode-editor-background: var(--vscode-editor-background, #1e1e1e);
                        --vscode-button-background: var(--vscode-button-background, #0e639c);
                        --vscode-foreground: var(--vscode-foreground, #cccccc);
                        --vscode-code-background: var(--vscode-textCodeBlock-background, #1e1e1e);
                    }
                    
                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }
                    
                    body {
                        padding: 0;
                        font-family: var(--vscode-font-family);
                        background: var(--vscode-editor-background);
                        color: var(--vscode-foreground);
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                        overflow: hidden;
                    }
                    
                    #header {
                        padding: 12px;
                        border-bottom: 1px solid #333;
                        background: rgba(0, 0, 0, 0.2);
                    }
                    
                    #header h2 {
                        font-size: 14px;
                        font-weight: 600;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    #status {
                        font-size: 11px;
                        color: #888;
                        margin-left: auto;
                    }
                    
                    #status.typing {
                        color: #4CAF50;
                    }
                    
                    #messages {
                        flex: 1;
                        overflow-y: auto;
                        padding: 12px;
                        scroll-behavior: smooth;
                    }
                    
                    .message {
                        margin: 12px 0;
                        padding: 10px 14px;
                        border-radius: 6px;
                        max-width: 90%;
                        word-wrap: break-word;
                        animation: fadeIn 0.3s ease;
                    }
                    
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(5px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    
                    .message.user {
                        background: var(--vscode-button-background);
                        color: white;
                        margin-left: auto;
                        border-bottom-right-radius: 2px;
                    }
                    
                    .message.agent {
                        background: rgba(255, 255, 255, 0.08);
                        border-bottom-left-radius: 2px;
                    }
                    
                    .message.system {
                        background: rgba(255, 165, 0, 0.15);
                        font-style: italic;
                        font-size: 12px;
                        text-align: center;
                        margin: 8px auto;
                        max-width: 80%;
                    }
                    
                    .message-content {
                        line-height: 1.5;
                    }
                    
                    .message-content p {
                        margin: 6px 0;
                    }
                    
                    .message-content pre {
                        background: var(--vscode-code-background);
                        padding: 10px;
                        border-radius: 4px;
                        overflow-x: auto;
                        margin: 8px 0;
                        font-family: 'Consolas', 'Courier New', monospace;
                        font-size: 12px;
                        border: 1px solid #333;
                    }
                    
                    .message-content code {
                        font-family: 'Consolas', 'Courier New', monospace;
                        background: rgba(0, 0, 0, 0.2);
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 0.9em;
                    }
                    
                    .message-content pre code {
                        background: transparent;
                        padding: 0;
                    }
                    
                    .message-content ul, .message-content ol {
                        margin: 6px 0;
                        padding-left: 20px;
                    }
                    
                    .message-content blockquote {
                        border-left: 3px solid var(--vscode-button-background);
                        padding-left: 12px;
                        margin: 8px 0;
                        color: #888;
                    }
                    
                    .input-area {
                        display: flex;
                        gap: 8px;
                        padding: 12px;
                        border-top: 1px solid #333;
                        background: rgba(0, 0, 0, 0.2);
                    }
                    
                    #input {
                        flex: 1;
                        padding: 10px 14px;
                        border: 1px solid #333;
                        border-radius: 6px;
                        background: var(--vscode-editor-background);
                        color: var(--vscode-foreground);
                        font-family: var(--vscode-font-family);
                        font-size: 13px;
                        resize: none;
                        min-height: 40px;
                        max-height: 120px;
                    }
                    
                    #input:focus {
                        outline: 2px solid var(--vscode-button-background);
                        border-color: transparent;
                    }
                    
                    #send {
                        padding: 10px 24px;
                        background: var(--vscode-button-background);
                        color: white;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 13px;
                        font-weight: 500;
                        transition: all 0.2s;
                    }
                    
                    #send:hover {
                        opacity: 0.9;
                        transform: translateY(-1px);
                    }
                    
                    #send:disabled {
                        opacity: 0.5;
                        cursor: not-allowed;
                        transform: none;
                    }
                    
                    .typing-indicator {
                        display: flex;
                        gap: 4px;
                        align-items: center;
                        padding: 10px 14px;
                    }
                    
                    .typing-indicator span {
                        width: 8px;
                        height: 8px;
                        background: #888;
                        border-radius: 50%;
                        animation: typing 1.4s infinite;
                    }
                    
                    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
                    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
                    
                    @keyframes typing {
                        0%, 100% { opacity: 0.3; transform: translateY(0); }
                        50% { opacity: 1; transform: translateY(-4px); }
                    }
                    
                    .progress-bar {
                        height: 3px;
                        background: linear-gradient(90deg, var(--vscode-button-background), #4CAF50);
                        border-radius: 2px;
                        margin-top: 8px;
                        transition: width 0.3s ease;
                    }
                    
                    /* Scrollbar styling */
                    ::-webkit-scrollbar {
                        width: 8px;
                    }
                    
                    ::-webkit-scrollbar-track {
                        background: rgba(0, 0, 0, 0.2);
                    }
                    
                    ::-webkit-scrollbar-thumb {
                        background: rgba(255, 255, 255, 0.2);
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb:hover {
                        background: rgba(255, 255, 255, 0.3);
                    }
                </style>
            </head>
            <body>
                <div id="header">
                    <h2>
                        <span>🤖</span>
                        PyUT Agent Chat
                        <span id="status">Ready</span>
                    </h2>
                </div>
                
                <div id="messages">
                    <div class="message agent">
                        <div class="message-content">
                            👋 Hello! I'm your PyUT Agent assistant. I can help you with:
                            <ul>
                                <li>Generate unit tests for Java code</li>
                                <li>Refactor and improve code quality</li>
                                <li>Debug and fix issues</li>
                                <li>Explain code functionality</li>
                            </ul>
                            How can I help you today?
                        </div>
                    </div>
                </div>
                
                <div class="input-area">
                    <textarea 
                        id="input" 
                        placeholder="Type your message... (Shift+Enter for new line)"
                        rows="1"
                    ></textarea>
                    <button id="send">Send</button>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    const messagesDiv = document.getElementById('messages');
                    const input = document.getElementById('input');
                    const sendButton = document.getElementById('send');
                    const statusSpan = document.getElementById('status');
                    
                    let isStreaming = false;
                    let currentMessageDiv = null;
                    
                    // 配置 marked (Markdown 解析器)
                    marked.setOptions({
                        breaks: true,
                        gfm: true,
                        highlight: function(code, lang) {
                            return code;
                        }
                    });
                    
                    // 自动调整输入框高度
                    input.addEventListener('input', function() {
                        this.style.height = 'auto';
                        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                    });
                    
                    // 发送消息
                    function sendMessage() {
                        const message = input.value.trim();
                        if (message && !isStreaming) {
                            // 添加用户消息
                            addMessage(message, 'user');
                            
                            // 发送到扩展
                            vscode.postMessage({
                                type: 'chat_message',
                                content: message
                            });
                            
                            input.value = '';
                            input.style.height = 'auto';
                            sendButton.disabled = true;
                            setStatus('Thinking...', true);
                        }
                    }
                    
                    // 添加消息到界面
                    function addMessage(content, role) {
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message ' + role;
                        
                        if (role === 'agent' || role === 'system') {
                            // 解析 Markdown
                            const htmlContent = marked.parse(content);
                            msgDiv.innerHTML = '<div class="message-content">' + htmlContent + '</div>';
                        } else {
                            msgDiv.innerHTML = '<div class="message-content">' + escapeHtml(content) + '</div>';
                        }
                        
                        messagesDiv.appendChild(msgDiv);
                        scrollToBottom();
                        return msgDiv;
                    }
                    
                    // 创建流式消息
                    function createStreamingMessage() {
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message agent';
                        msgDiv.innerHTML = '<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
                        messagesDiv.appendChild(msgDiv);
                        scrollToBottom();
                        return msgDiv;
                    }
                    
                    // 更新流式消息内容
                    function updateStreamingMessage(content) {
                        if (!currentMessageDiv) {
                            currentMessageDiv = createStreamingMessage();
                        }
                        
                        const contentDiv = currentMessageDiv.querySelector('.message-content');
                        const htmlContent = marked.parse(content);
                        contentDiv.innerHTML = htmlContent;
                        
                        // 添加进度条
                        if (isStreaming) {
                            const progress = document.createElement('div');
                            progress.className = 'progress-bar';
                            progress.style.width = '60%';
                            contentDiv.appendChild(progress);
                        }
                        
                        scrollToBottom();
                    }
                    
                    // 完成流式输出
                    function finishStreaming() {
                        isStreaming = false;
                        currentMessageDiv = null;
                        sendButton.disabled = false;
                        setStatus('Ready', false);
                    }
                    
                    // 设置状态
                    function setStatus(text, typing = false) {
                        statusSpan.textContent = text;
                        statusSpan.className = typing ? 'typing' : '';
                    }
                    
                    // 滚动到底部
                    function scrollToBottom() {
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }
                    
                    // HTML 转义
                    function escapeHtml(text) {
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }
                    
                    // 监听返回消息
                    window.addEventListener('message', event => {
                        const message = event.data;
                        
                        if (message.type === 'stream_start') {
                            isStreaming = true;
                            currentMessageDiv = createStreamingMessage();
                        } else if (message.type === 'stream_chunk') {
                            updateStreamingMessage(message.content);
                        } else if (message.type === 'stream_end') {
                            finishStreaming();
                        } else if (message.type === 'response') {
                            addMessage(message.content, 'agent');
                            sendButton.disabled = false;
                            setStatus('Ready', false);
                        } else if (message.type === 'system') {
                            addMessage(message.content, 'system');
                        } else if (message.type === 'error') {
                            addMessage('❌ Error: ' + message.content, 'system');
                            finishStreaming();
                        }
                    });
                    
                    // 事件监听
                    sendButton.onclick = sendMessage;
                    
                    input.onkeypress = (e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
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
    private async _handleMessage(message: any) {
        switch (message.type) {
            case 'chat_message':
                await this._processChatMessage(message.content);
                break;
        }
    }
    
    /**
     * 处理聊天消息（支持流式输出）
     */
    private async _processChatMessage(content: string) {
        try {
            // 发送流式开始消息
            this._view?.webview.postMessage({ type: 'stream_start' });
            
            let accumulatedContent = '';
            
            // 流式执行
            for await (const chunk of this._api.streamExecute(content)) {
                if (chunk.type === 'output') {
                    accumulatedContent += chunk.content;
                    this._view?.webview.postMessage({
                        type: 'stream_chunk',
                        content: accumulatedContent
                    });
                } else if (chunk.type === 'progress') {
                    this._view?.webview.postMessage({
                        type: 'system',
                        content: chunk.content
                    });
                } else if (chunk.type === 'complete') {
                    this._view?.webview.postMessage({
                        type: 'stream_end'
                    });
                } else if (chunk.type === 'error') {
                    this._view?.webview.postMessage({
                        type: 'error',
                        content: chunk.content
                    });
                }
            }
            
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            this._view?.webview.postMessage({
                type: 'error',
                content: errorMessage
            });
        }
    }
    
    /**
     * 发送普通消息
     */
    public sendMessage(message: any) {
        this._view?.webview.postMessage(message);
    }
}
