import * as vscode from 'vscode';

/**
 * 显示 Chat 面板命令
 */
export async function showChatPanel() {
    // 聚焦到 Chat 视图
    await vscode.commands.executeCommand('pyutagent.chatView.focus');
}
