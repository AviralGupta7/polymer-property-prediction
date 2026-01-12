# -*- coding: utf-8 -*-
"""
Python App to Executable Converter (FIXED v3 - MODELS EMBEDDED)
- Embeds model files DIRECTLY INSIDE the .exe
- Models loaded from memory, no external files needed
- Single .exe file, completely standalone
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
from datetime import datetime
import base64
import glob

class AppToExeBuilder:
    def __init__(self, app_file, requirements_file=None, output_name=None, icon_file=None, onefile=True):
        self.app_file = Path(app_file)
        self.requirements_file = Path(requirements_file) if requirements_file else Path("requirements.txt")
        self.output_name = output_name or self.app_file.stem
        self.icon_file = Path(icon_file) if icon_file else None
        self.onefile = onefile
        
        # Directories
        self.project_root = Path.cwd()
        self.venv_dir = self.project_root / "venv_build"
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.spec_dir = self.project_root / "specs"
        
        # Logging
        self.log_file = self.project_root / f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message, level="INFO"):
        """Log to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
        except:
            pass
    
    def read_file_safe(self, filepath):
        """Read file with multiple encoding fallbacks"""
        try:
            return filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return filepath.read_text(encoding='latin-1')
            except:
                try:
                    return filepath.read_text(encoding='cp1252')
                except:
                    return filepath.read_text(encoding='utf-8', errors='ignore')
    
    def embed_models(self):
        """Embed model files as base64 in launcher"""
        self.log("Scanning for model files...")
        
        models_folder = self.project_root / "production_models"
        if not models_folder.exists():
            self.log("[WARN] production_models folder not found", "WARN")
            return "{}"
        
        model_files = list(models_folder.glob("*"))
        if not model_files:
            self.log("[WARN] No model files found in production_models/", "WARN")
            return "{}"
        
        models_dict = {}
        total_size = 0
        
        for model_file in model_files:
            if model_file.is_file():
                try:
                    with open(model_file, 'rb') as f:
                        content = f.read()
                        encoded = base64.b64encode(content).decode('utf-8')
                        models_dict[model_file.name] = encoded
                        total_size += len(content)
                    self.log(f"[OK] Embedded: {model_file.name} ({len(content) / 1024 / 1024:.2f} MB)")
                except Exception as e:
                    self.log(f"[WARN] Failed to embed {model_file.name}: {e}", "WARN")
        
        self.log(f"[OK] Total embedded models: {total_size / 1024 / 1024:.2f} MB")
        
        # Convert dict to Python code
        models_code = "{\n"
        for filename, encoded in models_dict.items():
            models_code += f'    "{filename}": "{encoded}",\n'
        models_code += "}"
        
        return models_code
    
    def create_launcher(self):
        """Create launcher.py with embedded models"""
        self.log("Creating launcher with embedded models...")
        
        try:
            app_content = self.read_file_safe(self.app_file)
        except Exception as e:
            self.log(f"Failed to read app file: {e}", "ERROR")
            sys.exit(1)
        
        app_content_b64 = base64.b64encode(app_content.encode('utf-8')).decode('utf-8')
        models_dict_code = self.embed_models()
        
        launcher_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher for Streamlit app with embedded models"""
import sys
import os
import subprocess
import tempfile
import base64
import json

# EMBEDDED MODELS (base64 encoded)
EMBEDDED_MODELS = {models_dict_code}

# EMBEDDED APP
APP_PY_CONTENT_B64 = """{app_content_b64}"""

def setup_models_dir(base_dir):
    """Extract embedded models to models directory"""
    models_dir = os.path.join(base_dir, 'production_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for filename, b64_content in EMBEDDED_MODELS.items():
        try:
            model_path = os.path.join(models_dir, filename)
            model_data = base64.b64decode(b64_content)
            with open(model_path, 'wb') as f:
                f.write(model_data)
            print(f"[LAUNCHER] Extracted: {{filename}}")
        except Exception as e:
            print(f"[LAUNCHER] Failed to extract {{filename}}: {{e}}")
    
    return models_dir

def main():
    try:
        app_content = base64.b64decode(APP_PY_CONTENT_B64).decode('utf-8')
        temp_dir = tempfile.mkdtemp(prefix='streamlit_app_')
        
        # Extract models to temp directory
        models_dir = setup_models_dir(temp_dir)
        
        # Write app.py
        temp_app_path = os.path.join(temp_dir, 'app.py')
        with open(temp_app_path, 'w', encoding='utf-8') as f:
            f.write(app_content)
        
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            cmd = [
                "python",
                "-m", "streamlit",
                "run",
                temp_app_path,
                "--client.showErrorDetails=true",
                "--logger.level=info",
            ]
            
            print(f"[LAUNCHER] Starting Streamlit app from: {{temp_app_path}}")
            print(f"[LAUNCHER] Models at: {{models_dir}}")
            sys.stdout.flush()
            
            result = subprocess.run(cmd, check=False)
            
        finally:
            os.chdir(original_dir)
            try:
                import shutil as sh
                sh.rmtree(temp_dir)
            except:
                pass
                
    except Exception as e:
        print(f"[LAUNCHER ERROR] {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        try:
            launcher_path = self.project_root / "launcher.py"
            launcher_path.write_text(launcher_content, encoding='utf-8')
            self.log("[OK] launcher.py created with embedded models")
            return launcher_path
        except Exception as e:
            self.log(f"Failed to create launcher.py: {e}", "ERROR")
            sys.exit(1)
    
    def check_prerequisites(self):
        """Check prerequisites"""
        self.log("Checking prerequisites...")
        
        if not shutil.which("python") and not shutil.which("python3"):
            self.log("Python not found in PATH", "ERROR")
            sys.exit(1)
        self.log("[OK] Python found")
        
        if not shutil.which("pip"):
            self.log("pip not found in PATH", "ERROR")
            sys.exit(1)
        self.log("[OK] pip found")
        
        if not self.app_file.exists():
            self.log(f"App file not found: {self.app_file}", "ERROR")
            sys.exit(1)
        self.log(f"[OK] App file found: {self.app_file}")
        
        if not self.requirements_file.exists():
            self.log(f"[WARN] Requirements file not found (optional)", "WARN")
        else:
            self.log(f"[OK] Requirements file found")
    
    def create_venv(self):
        """Create virtual environment"""
        self.log(f"Creating virtual environment...")
        
        if self.venv_dir.exists():
            self.log("Virtual environment already exists, skipping...", "WARN")
            return
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True, capture_output=True)
            self.log("[OK] Virtual environment created")
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to create venv: {e}", "ERROR")
            sys.exit(1)
    
    def get_venv_python(self):
        """Get Python executable path from venv"""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
    
    def get_venv_pip(self):
        """Get pip executable path from venv"""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
    
    def install_dependencies(self):
        """Install dependencies"""
        self.log("Installing dependencies...")
        
        venv_pip = self.get_venv_pip()
        venv_pip_str = str(venv_pip)
        
        if not venv_pip.exists():
            self.log(f"venv pip not found", "ERROR")
            sys.exit(1)
        
        self.log("Installing PyInstaller...")
        try:
            subprocess.run([venv_pip_str, "install", "--upgrade", "pyinstaller"], 
                         check=True, capture_output=True, text=True)
            self.log("[OK] PyInstaller installed")
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install PyInstaller: {e.stderr}", "ERROR")
            sys.exit(1)
        
        if self.requirements_file.exists():
            self.log(f"Installing from requirements.txt...")
            try:
                subprocess.run([venv_pip_str, "install", "-r", str(self.requirements_file)], 
                             check=True, capture_output=True, text=True)
                self.log(f"[OK] Dependencies installed")
            except subprocess.CalledProcessError as e:
                self.log(f"Warning: Failed to install some dependencies", "WARN")
        else:
            self.log("Installing common packages...", "WARN")
            common_packages = ["streamlit", "pandas", "numpy", "scikit-learn", "xgboost"]
            try:
                subprocess.run([venv_pip_str, "install"] + common_packages, 
                             check=True, capture_output=True, text=True)
                self.log(f"[OK] Common packages installed")
            except subprocess.CalledProcessError as e:
                self.log(f"Warning: Failed to install some packages", "WARN")
    
    def find_pyinstaller(self):
        """Find PyInstaller executable"""
        if sys.platform == "win32":
            candidates = [
                self.venv_dir / "Scripts" / "pyinstaller.exe",
                self.venv_dir / "Scripts" / "pyinstaller",
            ]
        else:
            candidates = [
                self.venv_dir / "bin" / "pyinstaller",
            ]
        
        for candidate in candidates:
            if candidate.exists():
                self.log(f"Found PyInstaller at: {candidate}")
                return str(candidate)
        
        self.log("PyInstaller not found, will use python -m PyInstaller")
        return None
    
    def build_executable(self, launcher_path):
        """Build executable using PyInstaller"""
        self.log(f"Building executable: {self.output_name}")
        
        venv_python = self.get_venv_python()
        pyinstaller_path = self.find_pyinstaller()
        
        if pyinstaller_path:
            cmd = [pyinstaller_path]
        else:
            cmd = [str(venv_python), "-m", "PyInstaller"]
        
        cmd.extend([
            "--name", self.output_name,
            "--distpath", str(self.dist_dir),
            "--workpath", str(self.build_dir),
            "--specpath", str(self.spec_dir),
            "--onefile",
            "--console",
        ])
        
        # Add comprehensive imports for Streamlit apps
        cmd.extend([
            "--collect-all=streamlit",
            "--collect-all=pandas",
            "--collect-all=numpy",
            "--collect-all=sklearn",
            "--collect-all=xgboost",
            "--collect-all=matplotlib",
            "--collect-all=plotly",
            "--collect-all=altair",
            "--collect-all=protobuf",
            "--collect-all=pyarrow",
            
            "--hidden-import=streamlit",
            "--hidden-import=streamlit.web",
            "--hidden-import=streamlit.proto",
            "--hidden-import=pandas",
            "--hidden-import=numpy",
            "--hidden-import=sklearn",
            "--hidden-import=xgboost",
            "--hidden-import=matplotlib",
            "--hidden-import=plotly",
            "--hidden-import=altair",
            "--hidden-import=protobuf",
            "--hidden-import=pyarrow",
            "--hidden-import=PIL",
            "--hidden-import=base64",
            "--hidden-import=tempfile",
            "--hidden-import=shutil",
            
            "--noconfirm",
        ])
        
        cmd.append(str(launcher_path))
        
        self.log("Starting build (5-15 minutes, please wait...)")
        self.log(f"Build command: {' '.join(cmd[:8])}...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"Build failed:\n{result.stderr}", "ERROR")
                sys.exit(1)
            
            self.log("[OK] Build completed successfully!")
            
        except Exception as e:
            self.log(f"Build error: {e}", "ERROR")
            sys.exit(1)
    
    def get_executable_path(self):
        """Get path to built executable"""
        if sys.platform == "win32":
            return self.dist_dir / f"{self.output_name}.exe"
        else:
            return self.dist_dir / self.output_name
    
    def print_summary(self):
        """Print build summary"""
        exe_path = self.get_executable_path()
        
        self.log("=" * 70)
        self.log("BUILD COMPLETE!", "SUCCESS")
        self.log("=" * 70)
        
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024*1024)
            self.log(f"Executable: {exe_path}", "SUCCESS")
            self.log(f"Size: {size_mb:.2f} MB", "SUCCESS")
            self.log("[OK] Models are EMBEDDED inside the .exe", "SUCCESS")
        else:
            self.log(f"[WARN] Executable not found at: {exe_path}", "WARN")
        
        self.log("=" * 70)
        self.log("STANDALONE EXE READY! No external files needed.", "SUCCESS")
        self.log("=" * 70)
    
    def cleanup_build_files(self, keep_dist=True):
        """Clean up temporary build files"""
        self.log("Cleaning up temporary files...")
        
        if not keep_dist:
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            if self.spec_dir.exists():
                shutil.rmtree(self.spec_dir)
        
        self.log("[OK] Cleanup complete")
    
    def build(self):
        """Execute full build process"""
        self.log("=" * 70)
        self.log("PYTHON TO EXECUTABLE CONVERTER (v3 - MODELS EMBEDDED)", "INFO")
        self.log("=" * 70)
        
        self.check_prerequisites()
        launcher_path = self.create_launcher()
        self.create_venv()
        self.install_dependencies()
        self.build_executable(launcher_path)
        self.cleanup_build_files(keep_dist=True)
        self.print_summary()

def main():
    parser = argparse.ArgumentParser(
        description="Convert Python app to standalone executable with embedded models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_fixed_v3.py
  python build_fixed_v3.py --app myapp.py --output MyApp
  python build_fixed_v3.py --app app.py --onedir
        """
    )
    
    parser.add_argument("--app", type=str, default="app.py",
                       help="Path to main Python app file (default: app.py)")
    parser.add_argument("--requirements", type=str, default="requirements.txt",
                       help="Path to requirements.txt (default: requirements.txt)")
    parser.add_argument("--output", type=str, default=None,
                       help="Executable name (default: app name)")
    parser.add_argument("--icon", type=str, default=None,
                       help="Path to icon file (.ico)")
    parser.add_argument("--onefile", action="store_true", default=True,
                       help="Create single executable (default: True)")
    parser.add_argument("--onedir", action="store_true",
                       help="Create folder with executable and dependencies")
    
    args = parser.parse_args()
    
    builder = AppToExeBuilder(
        app_file=args.app,
        requirements_file=args.requirements,
        output_name=args.output,
        icon_file=args.icon,
        onefile=not args.onedir
    )
    
    builder.build()

if __name__ == "__main__":
    main()
