#!/usr/bin/env python3
"""
Enhanced Research Framework Setup Script
Automated installation and configuration for bio-inspired MARL research framework
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required for the research framework")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version compatible: {sys.version}")
    return True

def install_package(package, description=""):
    """Install a Python package with error handling"""
    try:
        print(f"Installing {package}...")
        if description:
            print(f"  → {description}")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, check=True)
        
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}")
        print(f"  Error: {e.stderr}")
        return False

def check_package_installed(package_name):
    """Check if a package is already installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_core_dependencies():
    """Install core dependencies required for basic functionality"""
    print("\n=== Installing Core Dependencies ===")
    
    core_packages = [
        ("psycopg2-binary>=2.9.9", "PostgreSQL database adapter"),
        ("SQLAlchemy>=2.0.23", "Database ORM"),
        ("alembic>=1.13.1", "Database migrations"),
        ("flask>=3.0.0", "Web framework"),
        ("pyyaml>=6.0", "YAML configuration support")
    ]
    
    success_count = 0
    for package, description in core_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\nCore dependencies: {success_count}/{len(core_packages)} installed successfully")
    return success_count == len(core_packages)

def install_research_framework():
    """Install enhanced research framework dependencies"""
    print("\n=== Installing Research Framework Dependencies ===")
    
    # Check if Ray is available in environment
    try:
        import ray
        print("✓ Ray already available")
        ray_available = True
    except ImportError:
        print("Ray not found, will attempt installation...")
        ray_available = False
    
    research_packages = [
        ("ray[rllib]==2.9.3", "Ray RLlib for distributed RL"),
        ("torch>=2.0.0,<2.6.0", "PyTorch for neural networks"),
        ("numpy>=1.24.0,<2.0.0", "Numerical computing"),
        ("gymnasium==0.28.1", "RL environment framework"),
        ("pettingzoo>=1.24.0", "Multi-agent environments"),
        ("pandas>=1.5.0", "Data analysis"),
        ("scipy>=1.9.0", "Scientific computing"),
        ("scikit-learn>=1.2.0", "Machine learning utilities"),
        ("statsmodels>=0.14.0", "Statistical analysis"),
        ("pingouin>=0.5.0", "Advanced statistical tests"),
        ("networkx>=3.0", "Graph analysis for communication"),
    ]
    
    success_count = 0
    failed_packages = []
    
    for package, description in research_packages:
        if install_package(package, description):
            success_count += 1
        else:
            failed_packages.append(package.split(">=")[0].split("==")[0])
    
    print(f"\nResearch framework: {success_count}/{len(research_packages)} packages installed")
    
    if failed_packages:
        print(f"⚠️  Failed packages: {', '.join(failed_packages)}")
        print("The system will use fallback implementations for missing packages.")
    
    return success_count >= len(research_packages) * 0.7  # 70% success rate

def install_optional_packages():
    """Install optional packages for enhanced functionality"""
    print("\n=== Installing Optional Enhancement Packages ===")
    
    optional_packages = [
        ("transformers>=4.21.0", "Attention mechanisms for pheromone networks"),
        ("einops>=0.6.0", "Tensor operations for neural plasticity"),
        ("tensorboard>=2.13.0", "Experiment tracking"),
        ("matplotlib>=3.6.0", "Visualization"),
        ("seaborn>=0.12.0", "Statistical visualization"),
        ("plotly>=5.15.0", "Interactive plots"),
    ]
    
    success_count = 0
    for package, description in optional_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\nOptional packages: {success_count}/{len(optional_packages)} installed")
    return True

def verify_installation():
    """Verify that key packages can be imported"""
    print("\n=== Verifying Installation ===")
    
    test_imports = [
        ("psycopg2", "PostgreSQL adapter"),
        ("sqlalchemy", "Database ORM"),
        ("flask", "Web framework"),
        ("yaml", "YAML support"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data analysis"),
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        try:
            importlib.import_module(module)
            print(f"✓ {module} - {description}")
        except ImportError:
            print(f"✗ {module} - {description}")
            failed_imports.append(module)
    
    # Test advanced packages with fallback support
    advanced_imports = [
        ("torch", "PyTorch"),
        ("ray", "Ray framework"),
        ("gymnasium", "RL environments"),
        ("scipy", "Scientific computing"),
        ("sklearn", "Machine learning"),
    ]
    
    print("\nAdvanced packages (with fallback support):")
    for module, description in advanced_imports:
        try:
            importlib.import_module(module)
            print(f"✓ {module} - {description}")
        except ImportError:
            print(f"⚠️  {module} - {description} (fallback will be used)")
    
    if failed_imports:
        print(f"\n⚠️  Some core packages failed to import: {', '.join(failed_imports)}")
        print("Please check the installation manually.")
        return False
    
    print("\n✅ Installation verification completed successfully!")
    return True

def create_environment_template():
    """Create a template .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n=== Creating Environment Configuration ===")
        try:
            import shutil
            shutil.copy(env_example, env_file)
            print("✓ Created .env file from template")
            print("📝 Please edit .env file with your database credentials")
        except Exception as e:
            print(f"⚠️  Could not create .env file: {e}")
            print("Please manually copy .env.example to .env and configure your database")

def main():
    """Main setup function"""
    print("🧠 Enhanced Research Framework Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies in order of importance
    print("\nStarting installation process...")
    
    # Core dependencies (required)
    if not install_core_dependencies():
        print("\n❌ Core dependency installation failed!")
        print("Some features may not work properly.")
    
    # Research framework (recommended)
    research_success = install_research_framework()
    if not research_success:
        print("\n⚠️  Research framework installation had issues.")
        print("The system will use fallback implementations where needed.")
    
    # Optional enhancements
    install_optional_packages()
    
    # Verify installation
    verification_success = verify_installation()
    
    # Create environment configuration
    create_environment_template()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Configure your .env file with database credentials")
    print("2. Run: npm install (for Node.js dependencies)")
    print("3. Run: npm run db:push (to setup database schema)")
    print("4. Run: npm run dev (to start the application)")
    print("5. Navigate to /research for the research dashboard")
    
    if not verification_success:
        print("\n⚠️  Note: Some packages failed to install.")
        print("The system includes fallback implementations and will work with reduced functionality.")
    
    print("\n📚 For detailed usage instructions, see:")
    print("   - README.md")
    print("   - docs/RESEARCH_FRAMEWORK_GUIDE.md")

if __name__ == "__main__":
    main()