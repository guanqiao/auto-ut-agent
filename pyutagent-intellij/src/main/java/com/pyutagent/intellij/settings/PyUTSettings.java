package com.pyutagent.intellij.settings;

import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import com.intellij.util.xmlb.XmlSerializerUtil;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

@State(
        name = "PyUTSettings",
        storages = @Storage("PyUTSettings.xml")
)
public class PyUTSettings implements PersistentStateComponent<PyUTSettings.State> {
    
    private State state = new State();
    
    public static PyUTSettings getInstance() {
        return ApplicationManager.getApplication().getService(PyUTSettings.class);
    }
    
    public static class State {
        public String serverUrl = "http://localhost:8080";
        public String apiKey = "";
        public String testFramework = "junit5";
        public double coverageTarget = 0.8;
        public boolean includeDisplayName = true;
        public boolean autoOpenTestFile = true;
        public int maxRetries = 3;
        public int timeoutSeconds = 120;
    }
    
    @Nullable
    @Override
    public State getState() {
        return state;
    }
    
    @Override
    public void loadState(@NotNull State state) {
        XmlSerializerUtil.copyBean(state, this.state);
    }
    
    // Getters and Setters
    public String getServerUrl() { return state.serverUrl; }
    public void setServerUrl(String serverUrl) { state.serverUrl = serverUrl; }
    
    public String getApiKey() { return state.apiKey; }
    public void setApiKey(String apiKey) { state.apiKey = apiKey; }
    
    public String getTestFramework() { return state.testFramework; }
    public void setTestFramework(String testFramework) { state.testFramework = testFramework; }
    
    public double getCoverageTarget() { return state.coverageTarget; }
    public void setCoverageTarget(double coverageTarget) { state.coverageTarget = coverageTarget; }
    
    public boolean isIncludeDisplayName() { return state.includeDisplayName; }
    public void setIncludeDisplayName(boolean includeDisplayName) { state.includeDisplayName = includeDisplayName; }
    
    public boolean isAutoOpenTestFile() { return state.autoOpenTestFile; }
    public void setAutoOpenTestFile(boolean autoOpenTestFile) { state.autoOpenTestFile = autoOpenTestFile; }
    
    public int getMaxRetries() { return state.maxRetries; }
    public void setMaxRetries(int maxRetries) { state.maxRetries = maxRetries; }
    
    public int getTimeoutSeconds() { return state.timeoutSeconds; }
    public void setTimeoutSeconds(int timeoutSeconds) { state.timeoutSeconds = timeoutSeconds; }
}
