import axios, { AxiosInstance, AxiosResponse } from 'axios';

/**
 * API 响应类型定义
 */
export interface GenerationResult {
    success: boolean;
    testCode?: string;
    message?: string;
    error?: string;
}

export interface TaskResult {
    success: boolean;
    output?: string;
    changes?: FileChange[];
    message?: string;
}

export interface FileChange {
    filePath: string;
    originalContent: string;
    modifiedContent: string;
    status: 'created' | 'modified' | 'deleted';
}

export interface StreamChunk {
    type: 'progress' | 'output' | 'complete' | 'error';
    content: string;
    step?: number;
    total?: number;
}

/**
 * PyUT Agent API 客户端
 */
export class PyUTAgentAPI {
    private client: AxiosInstance;
    private baseUrl: string;

    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    /**
     * 生成单元测试
     */
    async generateTest(filePath: string): Promise<GenerationResult> {
        try {
            const response: AxiosResponse<GenerationResult> = await this.client.post(
                '/api/generate',
                {
                    action: 'generate_test',
                    file_path: filePath
                }
            );
            return response.data;
        } catch (error) {
            if (axios.isAxiosError(error)) {
                return {
                    success: false,
                    error: error.response?.data?.message || error.message
                };
            }
            return {
                success: false,
                error: 'Unknown error occurred'
            };
        }
    }

    /**
     * 执行任务
     */
    async executeTask(
        request: string,
        context: any = {}
    ): Promise<TaskResult> {
        try {
            const response: AxiosResponse<TaskResult> = await this.client.post(
                '/api/execute',
                {
                    request: request,
                    context: context
                }
            );
            return response.data;
        } catch (error) {
            if (axios.isAxiosError(error)) {
                return {
                    success: false,
                    error: error.response?.data?.message || error.message
                };
            }
            return {
                success: false,
                error: 'Unknown error occurred'
            };
        }
    }

    /**
     * 流式执行任务
     */
    async *streamExecute(
        request: string,
        context: any = {}
    ): AsyncGenerator<StreamChunk> {
        try {
            const response = await fetch(`${this.baseUrl}/api/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    request: request,
                    context: context
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('No response body');
            }

            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    
                    try {
                        const data = JSON.parse(line);
                        yield data as StreamChunk;
                    } catch (e) {
                        console.error('Failed to parse chunk:', e);
                    }
                }
            }
        } catch (error) {
            yield {
                type: 'error',
                content: error instanceof Error ? error.message : 'Unknown error'
            };
        }
    }

    /**
     * 健康检查
     */
    async healthCheck(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }
}

// 单例模式
let apiInstance: PyUTAgentAPI | null = null;

export function getApiInstance(baseUrl?: string): PyUTAgentAPI {
    if (!apiInstance) {
        apiInstance = new PyUTAgentAPI(baseUrl);
    }
    return apiInstance;
}
