```markdown
# verl Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you how to contribute to the `verl` Python codebase, focusing on its coding conventions, file organization, and common development workflows. You'll learn how to maintain consistency in code style, update CI workflows, and apply targeted bugfixes, all while following best practices observed in the repository.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files and modules.
  ```
  verl/model_utils.py
  verl/rollout_manager.py
  ```

- **Import Style:**  
  Prefer relative imports within the package.
  ```python
  from .utils import load_config
  from .trainer import Trainer
  ```

- **Export Style:**  
  Use named exports; avoid wildcard (`*`) exports.
  ```python
  __all__ = ['Trainer', 'RolloutManager']
  ```

- **Commit Messages:**  
  - Use freeform messages, sometimes prefixed with tags such as `[docker]`, `[fully_async]`, `[ci]`.
  - Clearly describe the change, especially for bugfixes (include bug, root cause, and fix).

## Workflows

### CI Workflow Update
**Trigger:** When a new model is supported, a dependency is updated, or CI coverage needs to be adjusted.  
**Command:** `/update-ci`

1. Edit or add CI job files under `.github/workflows/` (YAML format).
2. Update or add test scripts under `tests/` or related directories to match new CI logic.
3. Optionally update documentation under `docs/` to reflect CI changes.
4. Run or trigger CI to verify the changes.

**Example:**
```yaml
# .github/workflows/python-ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

### Bugfix: Single Module
**Trigger:** When a specific error or bug is reported in a module (e.g., AttributeError, KeyError, missing guard).  
**Command:** `/bugfix-module`

1. Identify the problematic function or class in the relevant Python file (e.g., `verl/trainer.py`).
2. Edit the file to fix the bug (add guard, initialize variable, handle missing file, etc.).
3. Clearly describe the bug, root cause, and fix in the commit message.
4. Optionally add or update a test to cover the bugfix.
5. Validate the fix manually or via CI.

**Example:**
```python
# Before
def get_reward(self, data):
    return data['reward']

# After (add guard)
def get_reward(self, data):
    if 'reward' not in data:
        return 0
    return data['reward']
```
Commit message example:
```
[bugfix] Add guard for missing 'reward' key in get_reward()
Root cause: KeyError when 'reward' absent in data dict.
Fix: Return 0 if key is missing.
```

## Testing Patterns

- **Framework:** Not explicitly specified; likely uses `pytest` or standard Python testing.
- **File Pattern:** Test files are named with `.test.ts` (TypeScript), but Python tests likely reside in `tests/**/*.py`.
- **Location:** All tests are under the `tests/` directory.
- **CI Integration:** Tests are run as part of the CI workflow.

**Example:**
```python
# tests/test_trainer.py
from verl.trainer import Trainer

def test_trainer_initialization():
    trainer = Trainer()
    assert trainer is not None
```

## Commands

| Command        | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| /update-ci     | Update CI workflow files and related test scripts            |
| /bugfix-module | Apply a targeted bugfix to a single Python module            |
```
