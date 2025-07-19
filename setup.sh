#!/bin/bash

# Brain-Inspired Multi-Agent RL System Setup Script
# This script helps set up the development environment locally

echo "🧠 Setting up Brain-Inspired Multi-Agent RL System..."
echo "=================================================="

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if required tools are installed
echo ""
echo "Checking prerequisites..."

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js found: $NODE_VERSION"
else
    print_error "Node.js is not installed. Please install Node.js v20+ from https://nodejs.org/"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_status "npm found: $NPM_VERSION"
else
    print_error "npm is not installed. Please install npm."
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 is not installed. Please install Python 3.11+ from https://python.org/"
    exit 1
fi

# Check PostgreSQL
if command -v psql &> /dev/null; then
    print_status "PostgreSQL client found"
else
    print_warning "PostgreSQL client not found. Install PostgreSQL 12+ for database features."
fi

echo ""
echo "Installing dependencies..."

# Install Node.js dependencies
print_info "Installing Node.js packages..."
if npm install; then
    print_status "Node.js dependencies installed successfully"
else
    print_error "Failed to install Node.js dependencies"
    exit 1
fi

# Install core Python dependencies
print_info "Installing core Python dependencies..."

# Create a requirements list for essential packages
CORE_PYTHON_DEPS=(
    "psycopg2-binary>=2.9.9"
    "SQLAlchemy>=2.0.23"
    "alembic>=1.13.1"
    "flask>=3.0.0"
)

for dep in "${CORE_PYTHON_DEPS[@]}"; do
    print_info "Installing $dep..."
    if python3 -m pip install "$dep"; then
        print_status "Installed $dep"
    else
        print_warning "Failed to install $dep (this might be due to environment constraints)"
    fi
done

echo ""
print_info "Setting up environment configuration..."

# Create .env.example file if it doesn't exist
if [ ! -f ".env.example" ]; then
    cat > .env.example << EOL
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/marl_system
PGHOST=localhost
PGPORT=5432
PGUSER=your_username
PGPASSWORD=your_password
PGDATABASE=marl_system

# Optional: Session Configuration
SESSION_SECRET=your-super-secret-session-key

# Development Settings
NODE_ENV=development
EOL
    print_status "Created .env.example file"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_warning "No .env file found. Please copy .env.example to .env and configure your settings:"
    print_info "  cp .env.example .env"
    print_info "  # Then edit .env with your database credentials"
fi

echo ""
print_info "Database setup instructions:"
echo ""
echo "1. Make sure PostgreSQL is running:"
echo "   sudo systemctl start postgresql  # Linux"
echo "   brew services start postgresql   # macOS"
echo ""
echo "2. Create database and user:"
echo "   sudo -u postgres psql"
echo "   CREATE DATABASE marl_system;"
echo "   CREATE USER marl_user WITH PASSWORD 'your_password';"
echo "   GRANT ALL PRIVILEGES ON DATABASE marl_system TO marl_user;"
echo "   \\q"
echo ""
echo "3. Update your .env file with the database credentials"
echo ""
echo "4. Push database schema:"
echo "   npm run db:push"

echo ""
echo "🎉 Setup complete!"
echo ""
print_status "Your Brain-Inspired Multi-Agent RL System is ready!"
echo ""
echo "Next steps:"
echo "1. Configure your .env file with database settings"
echo "2. Set up your PostgreSQL database (see instructions above)"
echo "3. Run: npm run db:push"
echo "4. Start development server: npm run dev"
echo ""
echo "The application will be available at http://localhost:5000"
echo ""
print_info "For detailed setup instructions, see README.md"
echo "For troubleshooting, check the documentation or logs"
echo ""
echo "Happy coding! 🚀"