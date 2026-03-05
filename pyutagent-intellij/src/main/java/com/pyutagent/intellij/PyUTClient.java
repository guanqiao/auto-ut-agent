package com.pyutagent.intellij;

import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.components.Service;
import com.intellij.openapi.diagnostic.Logger;
import com.intellij.openapi.project.Project;
import okhttp3.*;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

@Service
public final class PyUTClient {
    private static final Logger LOG = Logger.getInstance(PyUTClient.class);
    private static final Gson GSON = new Gson();
    
    private final OkHttpClient httpClient;
    private String serverUrl = "http://localhost:8080";
    private String apiKey = "";
    
    public PyUTClient() {
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }
    
    public static PyUTClient getInstance() {
        return ApplicationManager.getApplication().getService(PyUTClient.class);
    }
    
    public void configure(String serverUrl, String apiKey) {
        this.serverUrl = serverUrl;
        this.apiKey = apiKey;
    }
    
    public CompletableFuture<GenerateResponse> generateTests(
            Project project,
            String filePath,
            String fileContent,
            GenerateOptions options
    ) {
        CompletableFuture<GenerateResponse> future = new CompletableFuture<>();
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("file_path", filePath);
        requestBody.addProperty("content", fileContent);
        requestBody.addProperty("project_path", project.getBasePath());
        requestBody.addProperty("framework", options.getFramework());
        requestBody.addProperty("coverage_target", options.getCoverageTarget());
        requestBody.addProperty("include_display_name", options.isIncludeDisplayName());
        
        Request request = new Request.Builder()
                .url(serverUrl + "/api/generate")
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        requestBody.toString(),
                        MediaType.parse("application/json")
                ))
                .build();
        
        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                LOG.error("Failed to generate tests", e);
                future.completeExceptionally(e);
            }
            
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                try (ResponseBody body = response.body()) {
                    if (!response.isSuccessful() || body == null) {
                        String error = body != null ? body.string() : "Unknown error";
                        LOG.error("API error: " + error);
                        future.completeExceptionally(new Exception(error));
                        return;
                    }
                    
                    String responseBody = body.string();
                    JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();
                    
                    GenerateResponse generateResponse = new GenerateResponse(
                            json.get("success").getAsBoolean(),
                            json.has("test_content") ? json.get("test_content").getAsString() : "",
                            json.has("test_file_path") ? json.get("test_file_path").getAsString() : "",
                            json.has("message") ? json.get("message").getAsString() : "",
                            json.has("coverage") ? json.get("coverage").getAsDouble() : 0.0
                    );
                    
                    future.complete(generateResponse);
                }
            }
        });
        
        return future;
    }
    
    public CompletableFuture<AnalyzeResponse> analyzeProject(Project project) {
        CompletableFuture<AnalyzeResponse> future = new CompletableFuture<>();
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("project_path", project.getBasePath());
        
        Request request = new Request.Builder()
                .url(serverUrl + "/api/analyze")
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        requestBody.toString(),
                        MediaType.parse("application/json")
                ))
                .build();
        
        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                LOG.error("Failed to analyze project", e);
                future.completeExceptionally(e);
            }
            
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                try (ResponseBody body = response.body()) {
                    if (!response.isSuccessful() || body == null) {
                        future.completeExceptionally(new Exception("API error"));
                        return;
                    }
                    
                    String responseBody = body.string();
                    JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();
                    
                    AnalyzeResponse analyzeResponse = new AnalyzeResponse(
                            json.get("total_files").getAsInt(),
                            json.get("testable_files").getAsInt(),
                            json.get("has_tests").getAsInt()
                    );
                    
                    future.complete(analyzeResponse);
                }
            }
        });
        
        return future;
    }
    
    public CompletableFuture<Boolean> checkHealth() {
        CompletableFuture<Boolean> future = new CompletableFuture<>();
        
        Request request = new Request.Builder()
                .url(serverUrl + "/health")
                .get()
                .build();
        
        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                future.complete(false);
            }
            
            @Override
            public void onResponse(Call call, Response response) {
                future.complete(response.isSuccessful());
            }
        });
        
        return future;
    }
    
    public static class GenerateOptions {
        private String framework = "junit5";
        private double coverageTarget = 0.8;
        private boolean includeDisplayName = true;
        
        public String getFramework() { return framework; }
        public void setFramework(String framework) { this.framework = framework; }
        public double getCoverageTarget() { return coverageTarget; }
        public void setCoverageTarget(double coverageTarget) { this.coverageTarget = coverageTarget; }
        public boolean isIncludeDisplayName() { return includeDisplayName; }
        public void setIncludeDisplayName(boolean includeDisplayName) { this.includeDisplayName = includeDisplayName; }
    }
    
    public static class GenerateResponse {
        private final boolean success;
        private final String testContent;
        private final String testFilePath;
        private final String message;
        private final double coverage;
        
        public GenerateResponse(boolean success, String testContent, String testFilePath, String message, double coverage) {
            this.success = success;
            this.testContent = testContent;
            this.testFilePath = testFilePath;
            this.message = message;
            this.coverage = coverage;
        }
        
        public boolean isSuccess() { return success; }
        public String getTestContent() { return testContent; }
        public String getTestFilePath() { return testFilePath; }
        public String getMessage() { return message; }
        public double getCoverage() { return coverage; }
    }
    
    public static class AnalyzeResponse {
        private final int totalFiles;
        private final int testableFiles;
        private final int hasTests;
        
        public AnalyzeResponse(int totalFiles, int testableFiles, int hasTests) {
            this.totalFiles = totalFiles;
            this.testableFiles = testableFiles;
            this.hasTests = hasTests;
        }
        
        public int getTotalFiles() { return totalFiles; }
        public int getTestableFiles() { return testableFiles; }
        public int getHasTests() { return hasTests; }
    }
}
