# Contributing Guidelines

Thank you for your interest in contributing to the Real Estate Price Prediction System!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Write meaningful commit messages

## Testing

- Write tests for all new features
- Ensure test coverage remains ≥80%
- Run tests before submitting PR:
  ```bash
  pytest tests/ -v --cov
  ```

## Pull Request Process

1. Update CHANGELOG.md with your changes
2. Ensure all tests pass
3. Update documentation if needed
4. Request review from maintainers

## Areas for Contribution

- Model improvements
- Feature engineering
- API enhancements
- Frontend UI/UX
- Documentation
- Test coverage
- Performance optimization
- Bug fixes

## Questions?

Open an issue for questions or discussions.

Thank you for contributing! 🎉

