# Lessons Learned: Perplexity-based K-S Test Implementation

This document captures key learnings from implementing perplexity-based Kolmogorov-Smirnov tests for the Model Equality Testing framework.

## 1. KenLM and Boost Dependency Issues

### Problem
When attempting to use the `lmplz` binary (from KenLM) to train language models, encountered the error:
```
dyld[86795]: Library not found: @rpath/libboost_system.dylib
```

### Root Cause
- **Boost 1.90.0 breaking change**: Starting with Boost 1.90.0, the `boost::system` library became header-only
- The library file `libboost_system.dylib` no longer exists, but older KenLM builds still linked against it
- KenLM's `CMakeLists.txt` was requesting `boost_system` as a required component, causing the build to link against a non-existent library

### Investigation Path
1. Initial hypothesis: Missing boost installation → ❌ Boost was installed via Homebrew
2. Second hypothesis: pip-installed kenlm package issue → ❌ The venv lmplz was actually a symlink to `/Users/dima/Build/kenlm/build/bin/lmplz`
3. Used `otool -L` to inspect library dependencies and discovered the missing `libboost_system.dylib` reference
4. Researched Boost changelog and discovered the 1.90.0 breaking change

### Solution
**Modified `/Users/dima/Build/kenlm/CMakeLists.txt`:**

```cmake
# BEFORE (broken with Boost 1.90.0+):
find_package(Boost 1.41.0 REQUIRED COMPONENTS
  program_options
  system          # ← This component no longer exists as a library
  thread
  unit_test_framework
)

# AFTER (working):
# Note: system is header-only since Boost 1.69
find_package(Boost 1.41.0 REQUIRED COMPONENTS
  program_options
  thread
  unit_test_framework
)
```

Then rebuilt KenLM:
```bash
cd /Users/dima/Build/kenlm/build
cmake ..
make -j 4
```

### Key Takeaway
When working with native dependencies:
- **Check library dependencies with `otool -L` (macOS) or `ldd` (Linux)** to understand what's missing
- **Research upstream breaking changes** in major version bumps
- **Modify build configuration rather than downgrading dependencies** when possible
- **Document the fix** for future reference

---

## 2. lmplz Validation and Exit Codes

### Problem
The `lmplz` binary exits with code 1 when invoked with `--help`, which our validation code initially treated as an error.

### Initial Approach (Broken)
```python
result = subprocess.run(['lmplz', '--help'], check=True)  # ❌ Raises exception
```

### Solution
Check the **content** of the help text, not just the exit code:

```python
result = subprocess.run(['lmplz', '--help'], capture_output=True, text=True, timeout=5)
if "Builds unpruned language models" not in result.stderr and \
   "Builds unpruned language models" not in result.stdout:
    raise RuntimeError("lmplz binary found but not working properly")
```

### Key Takeaway
- Exit codes are not always reliable indicators of success/failure
- For validation, **check the actual output content** when possible
- Add timeouts to prevent hangs during validation

---

## 3. Kneser-Ney Smoothing and Small Datasets

### Problem
When training KenLM models on small test datasets:
```
ERROR: Could not calculate Kneser-Ney discounts...
Is this small or artificial data?
```

### Root Cause
Kneser-Ney smoothing requires sufficient statistics to calculate discount parameters. Small datasets don't have enough n-gram counts.

### Solution
Add the `--discount_fallback` flag to lmplz:

```python
subprocess.run([
    'lmplz',
    '-o', str(order),
    '--text', temp_file,
    '--arpa', output_path,
    '--discount_fallback'  # ← Enables fallback for small datasets
])
```

### Key Takeaway
- Always test with **realistic data sizes** early in development
- For production systems, consider **warning users about minimum dataset requirements**
- Provide **fallback options** for edge cases when possible

---

## 4. Python Package Import Structure

### Problem
Import errors after restructuring code:
```python
ModuleNotFoundError: No module named 'model_equality_testing.src'
```

### Root Cause
The package is structured with `model_equality_testing/src/` as the source directory, but when installed, the `src/` directory maps directly to the `model_equality_testing` package namespace.

```
model_equality_testing/
├── src/              # This becomes the package root
│   ├── features.py
│   ├── tests.py
│   └── pvalue.py
└── pyproject.toml
```

### Solution
Use **relative imports** within the package:

```python
# ❌ WRONG (causes ModuleNotFoundError):
from model_equality_testing.src.features import get_vader_scores

# ✅ CORRECT:
from .features import get_vader_scores
```

### Key Takeaway
- Within a package, **use relative imports** (`.features`, `..utils`)
- From outside the package, use absolute imports (`model_equality_testing.features`)
- Test imports both **within the package** and **from external code**

---

## 5. Multiple Test Registries and State

### Problem
After adding new test functions to `registry.py`, they still weren't available via `run_two_sample_test()`.

### Root Cause
Discovered a **duplicate `IMPLEMENTED_TESTS` dictionary** at the end of `tests.py` that was the actual registry being imported by `algorithm.py`:

```python
# In algorithm.py (line 80):
from .tests import IMPLEMENTED_TESTS  # ← Not from registry.py!
```

Both `registry.py` and `tests.py` had their own `IMPLEMENTED_TESTS` dictionary, but only the one in `tests.py` was being used.

### Solution
Updated **both** dictionaries to include the new test functions:
- `registry.py`: Added for organizational clarity
- `tests.py`: Added for actual functionality (this is what `algorithm.py` imports)

### Long-term Fix
Consider consolidating to a single registry in a future refactoring.

### Key Takeaway
- **Search for duplicates** when state isn't updating as expected
- **Follow the import chain** to find what code is actually using
- Use `grep` or IDE "Find Usages" to track down where things are imported
- Consider **eliminating duplicate state** when refactoring

---

## 6. Python Module Cache Issues

### Problem
After fixing code, imports still failed or old behavior persisted.

### Root Cause
Python caches compiled bytecode in `__pycache__` directories. When module structure changes significantly, stale cache files can cause issues.

### Solution
```bash
# Clear all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Reinstall the package in development mode
pip install -e . --force-reinstall --no-deps
```

### Key Takeaway
- When debugging mysterious import issues, **clear Python cache first**
- The `--no-deps` flag prevents reinstalling dependencies (faster for development iteration)
- Use `--force-reinstall` to ensure package metadata is updated

---

## 7. Design Patterns: Flexibility vs Simplicity

### Context
Implemented two approaches for perplexity-based K-S testing:
- **Option A**: Standalone `two_sample_perplexity_ks()` function (simple, direct)
- **Option B**: Generic `two_sample_ks_statistic(feature_fn=...)` (flexible, reusable)

### Decision
Implemented **both** approaches:
- Option A for users who want a quick, dedicated perplexity test
- Option B for users who want to experiment with custom features

### Usage Comparison

**Option A (Simple):**
```python
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_perplexity_ks",
    kenlm_model_path="model.arpa"
)
```

**Option B (Flexible):**
```python
from functools import partial
from model_equality_testing.src.features import get_perplexity_scores

feature_fn = partial(
    get_perplexity_scores,
    kenlm_model_path="model.arpa",
    granularity="word"
)

pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_ks_statistic",
    feature_fn=feature_fn
)
```

### Key Takeaway
- **Provide both simple and flexible interfaces** when the use cases differ significantly
- Simple interfaces reduce friction for common cases
- Flexible interfaces enable power users to extend functionality
- The marginal cost of maintaining both is often worth the UX improvement
- Document both approaches clearly (see `OPTION_B_USAGE.md`)

---

## 8. Test-Driven Development for Scientific Code

### Approach
Created comprehensive test suites early:
- `test_perplexity_ks.py`: Tests basic functionality (training, scoring, K-S computation)
- `test_option_b.py`: Tests generic K-S with various feature functions

### Benefits
1. **Caught integration issues early** (registry problems, import errors)
2. **Provided executable documentation** of how to use the API
3. **Enabled confident refactoring** (knew immediately when something broke)
4. **Validated edge cases** (empty texts, zero tokens, small datasets)

### Key Takeaway
- For research code, **write tests that validate scientific correctness**, not just code correctness
- Include tests that use **realistic data** (not just synthetic examples)
- Test **integration with the full framework** (e.g., via `run_two_sample_test()`)
- Tests serve as **usage examples** for future users

---

## Summary: Key Principles

1. **Native dependencies are brittle** - Always validate, provide clear error messages, and document workarounds
2. **Exit codes lie** - Check actual output content when validating external binaries
3. **Small data breaks algorithms** - Test with realistic data sizes and provide fallbacks
4. **Import structure matters** - Use relative imports within packages, absolute from outside
5. **Search for duplicate state** - When updates don't take effect, look for multiple sources of truth
6. **Clear cache when confused** - Python's `__pycache__` can mask bugs during rapid iteration
7. **Flexibility costs complexity** - Provide both simple and flexible interfaces when appropriate
8. **Tests are documentation** - Write tests that show how to use your API correctly
