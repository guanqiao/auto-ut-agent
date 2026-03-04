import * as vscode from 'vscode';

/**
 * 生成单元测试命令
 */
export async function generateTest() {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor) {
        vscode.window.showErrorMessage('No active editor. Please open a Java file first.');
        return;
    }
    
    const document = editor.document;
    
    // 检查是否是 Java 文件
    if (document.languageId !== 'java') {
        vscode.window.showErrorMessage('Please select a Java file to generate tests.');
        return;
    }
    
    const filePath = document.fileName;
    const code = document.getText();
    
    vscode.window.showInformationMessage(
        `Generating test for: ${document.fileName}`
    );
    
    try {
        // TODO: 调用后端 API 生成测试
        // const result = await apiClient.generateTest(code, filePath);
        
        // 临时实现：显示成功消息
        vscode.window.showInformationMessage(
            'Test generation started! Check the Chat panel for progress.'
        );
        
        // 通知 Chat 面板
        vscode.commands.executeCommand('pyutagent.showChatPanel');
        
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        vscode.window.showErrorMessage(`Failed to generate test: ${errorMessage}`);
    }
}
