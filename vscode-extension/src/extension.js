const vscode = require('vscode');
const { spawn } = require('child_process');

let pyutagentProcess = null;
let apiClient = null;

async function activate(context) {
    console.log('PyUT Agent extension activating...');

    const config = vscode.workspace.getConfiguration('pyutagent');
    const endpoint = config.get('endpoint', 'http://localhost:8080');
    const apiKey = config.get('apiKey', '');

    apiClient = createApiClient(endpoint, apiKey);

    vscode.commands.registerCommand('pyutagent.generateTests', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (!document.fileName.endsWith('.java')) {
            vscode.window.showWarningMessage('Please select a Java file');
            return;
        }

        await generateTests(document);
    });

    vscode.commands.registerCommand('pyutagent.analyzeCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const document = editor.document;
        await analyzeCode(document);
    });

    vscode.commands.registerCommand('pyutagent.fixErrors', async () => {
        await fixErrors();
    });

    vscode.commands.registerCommand('pyutagent.openPanel', async () => {
        const panel = vscode.window.createWebviewPanel(
            'pyutagent-panel',
            'PyUT Agent',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = getWebviewHtml();
    });

    console.log('PyUT Agent extension activated');
}

async function generateTests(document) {
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBar.text = '$(sync~spin) Generating tests...';
    statusBar.show();

    try {
        const result = await apiClient.generateTests({
            filePath: document.fileName,
            content: document.getText()
        });

        if (result.success) {
            const testFilePath = result.testFilePath;
            const doc = await vscode.workspace.openTextDocument(testFilePath);
            await vscode.window.showTextDocument(doc);

            vscode.window.showInformationMessage(`Tests generated: ${testFilePath}`);
        } else {
            vscode.window.showErrorMessage(`Failed: ${result.error}`);
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Error: ${error.message}`);
    } finally {
        statusBar.dispose();
    }
}

async function analyzeCode(document) {
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBar.text = '$(sync~spin) Analyzing code...';
    statusBar.show();

    try {
        const result = await apiClient.analyzeCode({
            filePath: document.fileName,
            content: document.getText()
        });

        const channel = vscode.window.createOutputChannel('PyUT Agent Analysis');
        channel.show();

        if (result.issues && result.issues.length > 0) {
            channel.appendLine('=== Code Analysis Results ===');
            result.issues.forEach(issue => {
                const severity = issue.severity === 'error' ? '$(error)' : '$(warning)';
                channel.appendLine(`${severity} Line ${issue.line}: ${issue.message}`);
            });
        } else {
            channel.appendLine('No issues found');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Error: ${error.message}`);
    } finally {
        statusBar.dispose();
    }
}

async function fixErrors() {
    const diagnostics = vscode.languages.getDiagnostics();
    const issues = [];

    for (const [uri, diags] of diagnostics) {
        if (uri.fsPath.endsWith('.java')) {
            issues.push(...diags.map(d => ({
                file: uri.fsPath,
                range: d.range,
                message: d.message
            })));
        }
    }

    if (issues.length === 0) {
        vscode.window.showInformationMessage('No errors to fix');
        return;
    }

    const result = await apiClient.fixErrors({ issues });

    if (result.success && result.changes) {
        for (const change of result.changes) {
            const doc = await vscode.workspace.openTextDocument(change.filePath);
            const edit = new vscode.WorkspaceEdit();
            edit.replace(doc.uri, change.range, change.newText);
            await vscode.workspace.applyEdit(edit);
        }
        vscode.window.showInformationMessage(`Fixed ${result.changes.length} errors`);
    }
}

function createApiClient(endpoint, apiKey) {
    return {
        async generateTests(params) {
            return {
                success: true,
                testFilePath: params.filePath.replace('/main/', '/test/').replace('.java', 'Test.java')
            };
        },

        async analyzeCode(params) {
            return {
                issues: []
            };
        },

        async fixErrors(params) {
            return {
                success: true,
                changes: []
            };
        }
    };
}

function getWebviewHtml() {
    return `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #333; }
        .btn { 
            background: #007acc; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            cursor: pointer; 
            margin: 5px;
        }
        .btn:hover { background: #005a9e; }
        #output { 
            background: #f5f5f5; 
            padding: 10px; 
            margin-top: 20px; 
            white-space: pre-wrap; 
        }
    </style>
</head>
<body>
    <h1>PyUT Agent</h1>
    <button class="btn" onclick="generateTests()">Generate Tests</button>
    <button class="btn" onclick="analyzeCode()">Analyze Code</button>
    <button class="btn" onclick="fixErrors()">Fix Errors</button>
    <div id="output"></div>
    <script>
        function log(msg) {
            document.getElementById('output').textContent += msg + '\\n';
        }
        function generateTests() { log('Generating tests...'); }
        function analyzeCode() { log('Analyzing code...'); }
        function fixErrors() { log('Fixing errors...'); }
    </script>
</body>
</html>`;
}

function deactivate() {
    console.log('PyUT Agent extension deactivated');
}

module.exports = { activate, deactivate };
