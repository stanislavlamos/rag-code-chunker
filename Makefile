.PHONY: backend frontend run clean install

# Default target
all: install run

# Install dependencies
install:
	pip install -r requirements.txt

# Run backend server
backend:
	cd backend && uvicorn api.main:app --reload

# Run frontend server
frontend:
	cd frontend && streamlit run app.py

# Run both servers (in separate terminals)
run:
	@echo "Starting backend server..."
	@cd backend && uvicorn api.main:app --reload & \
	echo "Starting frontend server..." && \
	cd frontend && streamlit run app.py

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +

# Help command
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make backend    - Run backend server only"
	@echo "  make frontend   - Run frontend server only"
	@echo "  make run        - Run both servers"
	@echo "  make clean      - Clean up Python cache files"
	@echo "  make help       - Show this help message" 