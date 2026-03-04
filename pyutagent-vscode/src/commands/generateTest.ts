import * as vscode from 'vscode';
import { PyUTAgentAPI } from '../backend/apiClient';
import { DiffViewProvider } from '../diff/diffProvider';
import { getTerminalInstance } from '../terminal/terminalManager';

/**
 * 生成单元测试命令
 */
export async function generateTest(api: PyUTAgentAPI) {
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
        // 调用后端 API 生成测试
        const result = await api.generateTest(filePath);
        
        if (result.success && result.testCode) {
            // 显示 Diff 预览
            const diffProvider = new DiffViewProvider(
                vscode.extensions.getExtension('pyutagent.pyutagent-vscode')!.extensionUri
            );
            
            const diffResult = await diffProvider.showDiff(
                code,
                result.testCode,
                'java',
                'Preview Generated Test'
            );
            
            if (diffResult?.action === 'accept') {
                // 创建测试文件
                const testUri = await createTestFile(document, result.testCode);
                
                // 在终端中运行测试
                const terminal = getTerminalInstance();
                const testFileName = testUri.fsPath;
                await terminal.executeCommand(`mvn test -Dtest=${testFileName}`);
                
                vscode.window.showInformationMessage('Test generated and executed successfully!');
            } else {
                vscode.window.showInformationMessage('Test generation cancelled.');
            }
        } else {
            vscode.window.showErrorMessage(
                result.error || 'Failed to generate test'
            );
        }
        
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        vscode.window.showErrorMessage(`Failed to generate test: ${errorMessage}`);
    }
}

/**
 * 创建测试文件
 */
async function createTestFile(
    document: vscode.TextDocument,
    testCode: string
): Promise<vscode.Uri> {
    const originalPath = document.uri.fsPath;
    const testPath = originalPath.replace(
        /\/src\/main\/java\//,
        '/src/test/java/'
    ).replace('.java', 'Test.java');
    
    const testUri = vscode.Uri.file(testPath);
    
    // 确保目录存在
    await vscode.workspace.fs.createDirectory(
        vscode.Uri.parse(testPath.substring(0, testPath.lastIndexOf('/')))
    );
    
    // 写入文件
    await vscode.workspace.fs.writeFile(
        testUri,
        Buffer.from(testCode, 'utf-8')
    );
    
    // 打开文件
    await vscode.window.showTextDocument(testUri);
    
    return testUri;
}
