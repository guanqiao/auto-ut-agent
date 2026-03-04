import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';

/**
 * 命令执行结果
 */
export interface CommandResult {
    success: boolean;
    stdout: string;
    stderr: string;
    exitCode: number | null;
}

/**
 * 终端管理器
 */
export class TerminalManager {
    private terminal: vscode.Terminal | undefined;
    private readonly terminalName: string = 'PyUT Agent';
    private disposables: vscode.Disposable[] = [];

    /**
     * 在集成终端中执行命令
     */
    public async executeCommand(
        command: string,
        options: {
            cwd?: string;
            showOutput?: boolean;
            timeout?: number;
        } = {}
    ): Promise<CommandResult> {
        const { cwd, showOutput = true, timeout = 30000 } = options;

        return new Promise((resolve) => {
            const shell = process.platform === 'win32' ? 'powershell.exe' : 'bash';
            const args = process.platform === 'win32' 
                ? ['-ExecutionPolicy', 'Bypass', '-Command', command]
                : ['-c', command];

            const proc: ChildProcess = spawn(shell, args, {
                cwd: cwd || vscode.workspace.rootPath,
                env: process.env,
                shell: true
            });

            let stdout = '';
            let stderr = '';
            let outputShown = false;

            // 创建或显示终端
            if (showOutput) {
                this.showTerminal();
            }

            // 处理标准输出
            proc.stdout?.on('data', (data: Buffer) => {
                const output = data.toString();
                stdout += output;
                if (showOutput) {
                    this.writeToTerminal(output, 'stdout');
                    outputShown = true;
                }
            });

            // 处理错误输出
            proc.stderr?.on('data', (data: Buffer) => {
                const output = data.toString();
                stderr += output;
                if (showOutput) {
                    this.writeToTerminal(output, 'stderr');
                    outputShown = true;
                }
            });

            // 处理进程结束
            proc.on('close', (code: number | null) => {
                resolve({
                    success: code === 0,
                    stdout,
                    stderr,
                    exitCode: code
                });
            });

            // 处理错误
            proc.on('error', (error: Error) => {
                resolve({
                    success: false,
                    stdout,
                    stderr: error.message,
                    exitCode: null
                });
            });

            // 超时处理
            if (timeout > 0) {
                setTimeout(() => {
                    proc.kill();
                    resolve({
                        success: false,
                        stdout,
                        stderr: 'Command timed out',
                        exitCode: null
                    });
                }, timeout);
            }
        });
    }

    /**
     * 显示终端
     */
    public showTerminal() {
        if (!this.terminal) {
            this.terminal = vscode.window.createTerminal(this.terminalName);
            this.disposables.push(
                this.terminal,
                vscode.window.onDidCloseTerminal((t) => {
                    if (t === this.terminal) {
                        this.terminal = undefined;
                    }
                })
            );
        }
        this.terminal.show();
    }

    /**
     * 写入内容到终端
     */
    private writeToTerminal(text: string, type: 'stdout' | 'stderr') {
        if (!this.terminal) {
            this.terminal = vscode.window.createTerminal(this.terminalName);
            this.terminal.show();
        }

        // 错误输出用红色显示
        if (type === 'stderr') {
            this.terminal.sendText(`\x1b[31m${text}\x1b[0m`, false);
        } else {
            this.terminal.sendText(text, false);
        }
    }

    /**
     * 发送文本到终端
     */
    public sendText(text: string, addNewLine: boolean = true) {
        if (!this.terminal) {
            this.terminal = vscode.window.createTerminal(this.terminalName);
            this.terminal.show();
        }
        this.terminal.sendText(text, addNewLine);
    }

    /**
     * 清理终端
     */
    public dispose() {
        this.terminal?.dispose();
        this.disposables.forEach(d => d.dispose());
        this.disposables = [];
    }
}

// 单例模式
let terminalInstance: TerminalManager | null = null;

export function getTerminalInstance(): TerminalManager {
    if (!terminalInstance) {
        terminalInstance = new TerminalManager();
    }
    return terminalInstance;
}
