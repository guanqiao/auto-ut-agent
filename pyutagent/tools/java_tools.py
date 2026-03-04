"""Java/JDK tools for finding and validating Java installations."""

import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class JavaInfo:
    """Information about a Java installation."""
    java_home: str
    java_version: str
    vendor: str
    java_path: str
    javac_path: Optional[str] = None
    is_jdk: bool = False


def find_java_executable() -> Optional[str]:
    """Find Java executable with smart search strategy.
    
    Search order:
    1. Check JAVA_HOME environment variable
    2. Check JDK_HOME environment variable
    3. Check PATH using shutil.which
    4. Windows-specific locations
    5. Unix/macOS common locations
    
    Returns:
        Path to java executable if found, None otherwise
    """
    java_home = _get_java_home_from_env()
    if java_home:
        java_path = _check_java_bin(java_home)
        if java_path:
            logger.debug(f"[JavaFinder] Found java via JAVA_HOME: {java_path}")
            return java_path
    
    java_path = shutil.which("java")
    if java_path:
        logger.debug(f"[JavaFinder] Found java in PATH: {java_path}")
        return java_path
    
    if platform.system() == "Windows":
        java_path = _find_java_windows()
        if java_path:
            return java_path
    else:
        java_path = _find_java_unix()
        if java_path:
            return java_path
    
    logger.warning("[JavaFinder] Java not found in any standard location")
    return None


def find_javac_executable() -> Optional[str]:
    """Find javac executable with smart search strategy.
    
    Search order:
    1. Check JAVA_HOME environment variable
    2. Check JDK_HOME environment variable
    3. Check PATH using shutil.which
    4. Windows-specific locations
    5. Unix/macOS common locations
    
    Returns:
        Path to javac executable if found, None otherwise
    """
    java_home = _get_java_home_from_env()
    if java_home:
        javac_path = _check_javac_bin(java_home)
        if javac_path:
            logger.debug(f"[JavaFinder] Found javac via JAVA_HOME: {javac_path}")
            return javac_path
    
    javac_path = shutil.which("javac")
    if javac_path:
        logger.debug(f"[JavaFinder] Found javac in PATH: {javac_path}")
        return javac_path
    
    if platform.system() == "Windows":
        javac_path = _find_javac_windows()
        if javac_path:
            return javac_path
    else:
        javac_path = _find_javac_unix()
        if javac_path:
            return javac_path
    
    logger.warning("[JavaFinder] javac not found in any standard location")
    return None


def find_java_home() -> Optional[str]:
    """Find JAVA_HOME directory with smart search strategy.
    
    Returns:
        Path to JAVA_HOME directory if found, None otherwise
    """
    java_home = _get_java_home_from_env()
    if java_home:
        if Path(java_home).exists():
            logger.debug(f"[JavaFinder] Found JAVA_HOME from env: {java_home}")
            return java_home
    
    java_path = find_java_executable()
    if java_path:
        java_home = _derive_java_home_from_executable(java_path)
        if java_home:
            logger.debug(f"[JavaFinder] Derived JAVA_HOME from executable: {java_home}")
            return java_home
    
    if platform.system() == "Windows":
        java_home = _find_java_home_windows()
        if java_home:
            return java_home
    else:
        java_home = _find_java_home_unix()
        if java_home:
            return java_home
    
    logger.warning("[JavaFinder] JAVA_HOME not found")
    return None


def get_java_info() -> Optional[JavaInfo]:
    """Get detailed information about the Java installation.
    
    Returns:
        JavaInfo if Java is found, None otherwise
    """
    java_path = find_java_executable()
    if not java_path:
        return None
    
    java_home = find_java_home()
    javac_path = find_javac_executable()
    
    version, vendor = _get_java_version_and_vendor(java_path)
    
    return JavaInfo(
        java_home=java_home or "",
        java_version=version,
        vendor=vendor,
        java_path=java_path,
        javac_path=javac_path,
        is_jdk=javac_path is not None
    )


def get_configured_java_paths() -> Tuple[Optional[str], Optional[str]]:
    """Get configured Java and javac paths.
    
    Priority:
    1. User configured java_home from settings
    2. Auto-detected paths
    
    Returns:
        Tuple of (java_path, javac_path)
    """
    from ..core.config import get_settings
    
    settings = get_settings()
    configured_home = settings.jdk.java_home
    
    if configured_home and configured_home.strip():
        java_home = Path(configured_home.strip())
        if java_home.exists():
            java_path = _check_java_bin(str(java_home))
            javac_path = _check_javac_bin(str(java_home))
            if java_path:
                logger.info(f"[JavaFinder] Using configured JAVA_HOME: {java_home}")
                return java_path, javac_path
            else:
                logger.warning(f"[JavaFinder] Configured JAVA_HOME invalid: {java_home}")
    
    java_path = find_java_executable()
    javac_path = find_javac_executable()
    
    return java_path, javac_path


def _get_java_home_from_env() -> Optional[str]:
    """Get JAVA_HOME from environment variables."""
    for env_var in ["JAVA_HOME", "JDK_HOME", "JRE_HOME"]:
        java_home = os.environ.get(env_var)
        if java_home and Path(java_home).exists():
            return java_home
    return None


def _check_java_bin(java_home: str) -> Optional[str]:
    """Check if java exists in the bin directory of java home."""
    home_path = Path(java_home)
    
    if platform.system() == "Windows":
        java_exe = home_path / "bin" / "java.exe"
        if java_exe.exists():
            return str(java_exe)
    
    java_bin = home_path / "bin" / "java"
    if java_bin.exists():
        return str(java_bin)
    
    return None


def _check_javac_bin(java_home: str) -> Optional[str]:
    """Check if javac exists in the bin directory of java home."""
    home_path = Path(java_home)
    
    if platform.system() == "Windows":
        javac_exe = home_path / "bin" / "javac.exe"
        if javac_exe.exists():
            return str(javac_exe)
    
    javac_bin = home_path / "bin" / "javac"
    if javac_bin.exists():
        return str(javac_bin)
    
    return None


def _find_java_windows() -> Optional[str]:
    """Search for Java in Windows-specific locations."""
    search_paths = _get_windows_java_search_paths()
    
    for search_path in search_paths:
        java_exe = search_path / "java.exe"
        if java_exe.exists():
            logger.debug(f"[JavaFinder] Found java.exe at {java_exe}")
            return str(java_exe)
    
    return None


def _find_javac_windows() -> Optional[str]:
    """Search for javac in Windows-specific locations."""
    search_paths = _get_windows_java_search_paths()
    
    for search_path in search_paths:
        javac_exe = search_path / "javac.exe"
        if javac_exe.exists():
            logger.debug(f"[JavaFinder] Found javac.exe at {javac_exe}")
            return str(javac_exe)
    
    return None


def _get_windows_java_search_paths() -> List[Path]:
    """Get list of Windows Java bin search paths."""
    paths = []
    
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    program_data = os.environ.get("ProgramData", "C:\\ProgramData")
    user_profile = os.environ.get("USERPROFILE", "C:\\Users\\Default")
    
    java_vendors = ["Java", "Eclipse Adoptium", "AdoptOpenJDK", "Amazon Corretto", "Microsoft", "BellSoft", "Azul Systems", "Zulu"]
    
    for base in [program_files, program_files_x86]:
        for vendor in java_vendors:
            vendor_path = Path(base) / vendor
            if vendor_path.exists():
                for jdk_dir in sorted(vendor_path.iterdir(), reverse=True):
                    if jdk_dir.is_dir():
                        bin_path = jdk_dir / "bin"
                        if bin_path.exists():
                            paths.append(bin_path)
    
    for base in [program_files, program_files_x86]:
        jdk_path = Path(base) / "JDK"
        if jdk_path.exists():
            for jdk_dir in sorted(jdk_path.iterdir(), reverse=True):
                if jdk_dir.is_dir():
                    bin_path = jdk_dir / "bin"
                    if bin_path.exists():
                        paths.append(bin_path)
    
    chocolatey_path = Path(program_data) / "chocolatey" / "bin"
    if chocolatey_path.exists():
        paths.append(chocolatey_path)
    
    scoop_shims = Path(user_profile) / "scoop" / "shims"
    if scoop_shims.exists():
        paths.append(scoop_shims)
    
    scoop_apps = Path(user_profile) / "scoop" / "apps"
    if scoop_apps.exists():
        for openjdk_dir in scoop_apps.glob("openjdk*"):
            current = openjdk_dir / "current" / "bin"
            if current.exists():
                paths.append(current)
            bin_dir = openjdk_dir / "bin"
            if bin_dir.exists():
                paths.append(bin_dir)
    
    common_paths = [
        Path("C:\\Java") / "bin",
        Path("D:\\Java") / "bin",
        Path(user_profile) / ".jdks" / "bin",
    ]
    paths.extend([p for p in common_paths if p.exists()])
    
    return paths


def _find_java_unix() -> Optional[str]:
    """Search for Java in Unix/macOS-specific locations."""
    search_paths = _get_unix_java_search_paths()
    
    for search_path in search_paths:
        java_bin = Path(search_path) / "java"
        if java_bin.exists():
            logger.debug(f"[JavaFinder] Found java at {java_bin}")
            return str(java_bin)
    
    return None


def _find_javac_unix() -> Optional[str]:
    """Search for javac in Unix/macOS-specific locations."""
    search_paths = _get_unix_java_search_paths()
    
    for search_path in search_paths:
        javac_bin = Path(search_path) / "javac"
        if javac_bin.exists():
            logger.debug(f"[JavaFinder] Found javac at {javac_bin}")
            return str(javac_bin)
    
    return None


def _get_unix_java_search_paths() -> List[str]:
    """Get list of Unix/macOS Java bin search paths."""
    paths = []
    
    if platform.system() == "Darwin":
        jvm_base = Path("/Library/Java/JavaVirtualMachines")
        if jvm_base.exists():
            for jdk_dir in sorted(jvm_base.iterdir(), reverse=True):
                if jdk_dir.is_dir():
                    bin_path = jdk_dir / "Contents" / "Home" / "bin"
                    if bin_path.exists():
                        paths.append(str(bin_path))
        
        homebrew_paths = [
            "/opt/homebrew/opt/openjdk/bin",
            "/usr/local/opt/openjdk/bin",
            "/opt/homebrew/Cellar/openjdk",
            "/usr/local/Cellar/openjdk",
        ]
        for hp in homebrew_paths:
            if Path(hp).exists():
                if hp.endswith("bin"):
                    paths.append(hp)
                else:
                    for version_dir in sorted(Path(hp).iterdir(), reverse=True):
                        bin_path = version_dir / "bin"
                        if bin_path.exists():
                            paths.append(str(bin_path))
    else:
        jvm_base = Path("/usr/lib/jvm")
        if jvm_base.exists():
            for jvm_dir in sorted(jvm_base.iterdir(), reverse=True):
                if jvm_dir.is_dir():
                    bin_path = jvm_dir / "bin"
                    if bin_path.exists():
                        paths.append(str(bin_path))
        
        linux_paths = [
            "/usr/share/java/bin",
            "/usr/local/share/java/bin",
            "/opt/java/bin",
            "/opt/jdk/bin",
            "/opt/openjdk/bin",
            "/usr/lib/jvm/default-java/bin",
        ]
        paths.extend([p for p in linux_paths if Path(p).exists()])
    
    snap_path = "/snap/bin"
    if Path(snap_path).exists():
        paths.append(snap_path)
    
    return paths


def _find_java_home_windows() -> Optional[str]:
    """Find JAVA_HOME on Windows."""
    search_paths = _get_windows_java_search_paths()
    
    for bin_path in search_paths:
        java_exe = bin_path / "java.exe"
        if java_exe.exists():
            java_home = _derive_java_home_from_executable(str(java_exe))
            if java_home:
                return java_home
    
    return None


def _find_java_home_unix() -> Optional[str]:
    """Find JAVA_HOME on Unix/macOS."""
    search_paths = _get_unix_java_search_paths()
    
    for bin_path in search_paths:
        java_bin = Path(bin_path) / "java"
        if java_bin.exists():
            java_home = _derive_java_home_from_executable(str(java_bin))
            if java_home:
                return java_home
    
    return None


def _derive_java_home_from_executable(java_path: str) -> Optional[str]:
    """Derive JAVA_HOME from java executable path."""
    java_path = Path(java_path)
    
    if java_path.name in ["java", "java.exe"]:
        bin_dir = java_path.parent
        if bin_dir.name == "bin":
            java_home = bin_dir.parent
            if java_home.exists():
                return str(java_home)
    
    return None


def _get_java_version_and_vendor(java_path: str) -> Tuple[str, str]:
    """Get Java version and vendor by running java -version."""
    try:
        result = subprocess.run(
            [java_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stderr or result.stdout
        
        version = "unknown"
        version_match = re.search(r'version "?(\d+[.\d]*)"?', output)
        if version_match:
            version = version_match.group(1)
        
        vendor = "unknown"
        if "OpenJDK" in output or "openjdk" in output.lower():
            vendor = "OpenJDK"
        elif "Oracle" in output or "Java(TM)" in output:
            vendor = "Oracle"
        elif "Eclipse Adoptium" in output or "Temurin" in output:
            vendor = "Eclipse Adoptium"
        elif "Amazon Corretto" in output:
            vendor = "Amazon Corretto"
        elif "Microsoft" in output:
            vendor = "Microsoft"
        elif "Azul" in output or "Zulu" in output:
            vendor = "Azul"
        elif "BellSoft" in output or "LibERICA" in output:
            vendor = "BellSoft"
        elif "IBM" in output:
            vendor = "IBM"
        
        return version, vendor
        
    except Exception as e:
        logger.warning(f"[JavaFinder] Failed to get Java version: {e}")
        return "unknown", "unknown"
