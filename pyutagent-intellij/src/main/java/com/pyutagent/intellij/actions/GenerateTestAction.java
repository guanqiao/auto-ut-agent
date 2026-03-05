package com.pyutagent.intellij.actions;

import com.intellij.notification.Notification;
import com.intellij.notification.NotificationType;
import com.intellij.notification.Notifications;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.diagnostic.Logger;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.progress.ProgressIndicator;
import com.intellij.openapi.progress.ProgressManager;
import com.intellij.openapi.progress.Task;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.vfs.LocalFileSystem;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiFile;
import com.intellij.psi.PsiJavaFile;
import com.intellij.psi.PsiManager;
import com.pyutagent.intellij.PyUTClient;
import com.pyutagent.intellij.settings.PyUTSettings;
import org.jetbrains.annotations.NotNull;

import java.nio.file.Path;
import java.nio.file.Paths;

public class GenerateTestAction extends AnAction {
    private static final Logger LOG = Logger.getInstance(GenerateTestAction.class);
    
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Project project = e.getProject();
        if (project == null) return;
        
        VirtualFile file = e.getData(CommonDataKeys.VIRTUAL_FILE);
        if (file == null || !file.getName().endsWith(".java")) {
            showNotification(project, "Please select a Java file", NotificationType.WARNING);
            return;
        }
        
        PsiFile psiFile = PsiManager.getInstance(project).findFile(file);
        if (!(psiFile instanceof PsiJavaFile)) {
            showNotification(project, "Selected file is not a Java file", NotificationType.WARNING);
            return;
        }
        
        PsiJavaFile javaFile = (PsiJavaFile) psiFile;
        String content = javaFile.getText();
        String filePath = file.getPath();
        
        PyUTSettings settings = PyUTSettings.getInstance();
        PyUTClient client = PyUTClient.getInstance();
        client.configure(settings.getServerUrl(), settings.getApiKey());
        
        ProgressManager.getInstance().run(new Task.Backgroundable(project, "Generating Unit Tests...", true) {
            @Override
            public void run(@NotNull ProgressIndicator indicator) {
                indicator.setText("Analyzing code...");
                indicator.setFraction(0.1);
                
                try {
                    PyUTClient.GenerateOptions options = new PyUTClient.GenerateOptions();
                    options.setFramework(settings.getTestFramework());
                    options.setCoverageTarget(settings.getCoverageTarget());
                    options.setIncludeDisplayName(settings.isIncludeDisplayName());
                    
                    indicator.setText("Generating tests...");
                    indicator.setFraction(0.3);
                    
                    PyUTClient.GenerateResponse response = client.generateTests(
                            project, filePath, content, options
                    ).join();
                    
                    indicator.setFraction(0.8);
                    
                    if (response.isSuccess()) {
                        ApplicationManager.getApplication().invokeLater(() -> {
                            writeTestFile(project, file, response.getTestContent(), response.getTestFilePath());
                            showNotification(project, 
                                    String.format("Tests generated! Coverage: %.1f%%", response.getCoverage() * 100),
                                    NotificationType.INFORMATION);
                        });
                    } else {
                        ApplicationManager.getApplication().invokeLater(() -> 
                                showNotification(project, "Failed: " + response.getMessage(), NotificationType.ERROR));
                    }
                    
                } catch (Exception ex) {
                    LOG.error("Test generation failed", ex);
                    ApplicationManager.getApplication().invokeLater(() -> 
                            showNotification(project, "Error: " + ex.getMessage(), NotificationType.ERROR));
                }
            }
        });
    }
    
    private void writeTestFile(Project project, VirtualFile sourceFile, String testContent, String testFilePath) {
        VirtualFile testDir = findOrCreateTestDirectory(project, sourceFile);
        if (testDir == null) return;
        
        String testFileName = getTestFileName(sourceFile.getName());
        
        WriteCommandAction.runWriteCommandAction(project, () -> {
            try {
                VirtualFile existingTest = testDir.findChild(testFileName);
                if (existingTest != null) {
                    int result = Messages.showYesNoDialog(
                            project,
                            "Test file already exists. Overwrite?",
                            "Confirm Overwrite",
                            Messages.getQuestionIcon()
                    );
                    if (result != Messages.YES) return;
                    
                    existingTest.setBinaryContent(testContent.getBytes());
                    FileEditorManager.getInstance(project).openFile(existingTest, true);
                } else {
                    VirtualFile newTestFile = testDir.createChildData(this, testFileName);
                    newTestFile.setBinaryContent(testContent.getBytes());
                    FileEditorManager.getInstance(project).openFile(newTestFile, true);
                }
            } catch (Exception e) {
                LOG.error("Failed to write test file", e);
            }
        });
    }
    
    private VirtualFile findOrCreateTestDirectory(Project project, VirtualFile sourceFile) {
        String sourcePath = sourceFile.getPath();
        String testPath = sourcePath
                .replace("/src/main/java/", "/src/test/java/")
                .replace("/src/main/", "/src/test/");
        
        Path testDirPath = Paths.get(testPath).getParent();
        VirtualFile testDir = LocalFileSystem.getInstance().findFileByPath(testDirPath.toString());
        
        if (testDir == null) {
            testDir = createDirectories(testDirPath.toString());
        }
        
        return testDir;
    }
    
    private VirtualFile createDirectories(String path) {
        try {
            VirtualFile root = LocalFileSystem.getInstance().findFileByPath(
                    path.substring(0, path.indexOf("/src/test/java/") + 15)
            );
            if (root == null) return null;
            
            String[] parts = path.substring(path.indexOf("/src/test/java/") + 16).split("/");
            VirtualFile current = root;
            
            for (String part : parts) {
                VirtualFile child = current.findChild(part);
                if (child == null) {
                    child = current.createChildDirectory(this, part);
                }
                current = child;
            }
            
            return current;
        } catch (Exception e) {
            LOG.error("Failed to create test directory", e);
            return null;
        }
    }
    
    private String getTestFileName(String sourceFileName) {
        String baseName = sourceFileName.replace(".java", "");
        return baseName + "Test.java";
    }
    
    private void showNotification(Project project, String message, NotificationType type) {
        Notification notification = new Notification(
                "PyUT Agent Notifications",
                "PyUT Agent",
                message,
                type
        );
        Notifications.Bus.notify(notification, project);
    }
    
    @Override
    public void update(@NotNull AnActionEvent e) {
        VirtualFile file = e.getData(CommonDataKeys.VIRTUAL_FILE);
        boolean enabled = file != null && file.getName().endsWith(".java");
        e.getPresentation().setEnabledAndVisible(enabled);
    }
}
