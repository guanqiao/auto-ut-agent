import * as vscode from 'vscode';
import { ChatViewProvider } from './chat/chatViewProvider';
import { generateTest } from './commands/generateTest';
import { showChatPanel } from './commands/showChatPanel';
import { PyUTAgentAPI, getApiInstance } from './backend/apiClient';
import { getTerminalInstance } from './terminal/terminalManager';

let api: PyUTAgentAPI;

export function activate(context: vscode.ExtensionContext) {
    console.log('PyUT Agent is now active!');
    
    // 初始化 API 客户端
    const config = vscode.workspace.getConfiguration('pyutagent');
    const apiUrl = config.get<string>('apiUrl', 'http://localhost:8000');
    api = getApiInstance(apiUrl);
    
    // 注册 Chat View Provider
    const chatProvider = new ChatViewProvider(context.extensionUri, api);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'pyutagent.chatView',
            chatProvider
        )
    );
    
    // 注册命令
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'pyutagent.generateTest',
            () => generateTest(api)
        )
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'pyutagent.showChatPanel',
            showChatPanel
        )
    );
    
    // 注册终端命令
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'pyutagent.runCommand',
            async () => {
                const command = await vscode.window.showInputBox({
                    prompt: 'Enter command to execute',
                    placeHolder: 'mvn clean test'
                });
                
                if (command) {
                    const terminal = getTerminalInstance();
                    await terminal.executeCommand(command);
                }
            }
        )
    );
    
    // 显示激活通知
    vscode.window.showInformationMessage(
        'PyUT Agent is ready! Generate tests with right-click on Java files.'
    );
}

export function deactivate() {
    console.log('PyUT Agent is now deactivated.');
    getTerminalInstance().dispose();
}
