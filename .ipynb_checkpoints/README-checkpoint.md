# 🔢 NumPy Functions — Complete Reference for Data Scientists

> A **complete, categorized reference** of every NumPy function a Data Scientist or Data Analyst needs — with use-case annotations, priority markers, and a daily-use Top 20 cheat sheet.

---

## 📌 Table of Contents

- [1️⃣ Array Creation](#1️⃣-array-creation)
- [2️⃣ Array Inspection](#2️⃣-array-inspection)
- [3️⃣ Indexing & Slicing](#3️⃣-indexing--slicing)
- [4️⃣ Math & Arithmetic](#4️⃣-math--arithmetic)
- [5️⃣ Statistics](#5️⃣-statistics--most-important-for-ds)
- [6️⃣ Array Manipulation](#6️⃣-array-manipulation)
- [7️⃣ Sorting & Searching](#7️⃣-sorting--searching)
- [8️⃣ Linear Algebra](#8️⃣-linear-algebra--critical-for-ml)
- [9️⃣ Random Module](#9️⃣-random-module)
- [🔟 Data Cleaning & Handling](#-data-cleaning--handling)
- [1️⃣1️⃣ Boolean Operations](#1️⃣1️⃣-boolean-operations)
- [1️⃣2️⃣ Set Operations](#1️⃣2️⃣-set-operations)
- [⭐ Top 20 Daily Functions](#-top-20-functions-used-daily-in-ds)
- [Getting Started](#-getting-started)

---

## 1️⃣ Array Creation

> The starting point of every NumPy workflow.


np.array()          # create array from list
np.zeros()          # array of zeros
np.ones()           # array of ones
np.full()           # array filled with a specific value
np.arange()         # evenly spaced values (like Python range)
np.linspace()       # evenly spaced values (fixed count)
np.eye()            # identity matrix
np.random.rand()    # random array between 0 and 1
np.random.randn()   # random array (normal distribution)
np.random.randint() # random integers
np.empty()          # uninitialized array (fastest creation)
```

---

## 2️⃣ Array Inspection

> Understand what's inside your array before working with it.

arr.shape           # dimensions of array (rows, cols)
arr.ndim            # number of dimensions
arr.size            # total number of elements
arr.dtype           # data type of elements
arr.reshape()       # change shape of array
arr.flatten()       # collapse to 1D (returns copy)
arr.ravel()         # flatten (returns view — faster)
arr.T               # transpose rows and columns
np.info()           # detailed info about array or function
```

---

## 3️⃣ Indexing & Slicing

> Access, filter, and extract exactly what you need from data.

arr[0]              # single element
arr[1:5]            # slice from index 1 to 4
arr[::2]            # every 2nd element (step slicing)
arr[arr > 5]        # ⭐ boolean indexing — used heavily in DA
arr[[1, 3, 5]]      # fancy indexing — pick specific indices
arr[0, 1]           # element at row 0, col 1 (2D array)
arr[:, 0]           # ⭐ entire first column — very common in DS
arr[0, :]           # entire first row
np.where()          # ⭐ conditional indexing — very important
```

---

## 4️⃣ Math & Arithmetic

> Vectorized math operations — much faster than Python loops.

```python
np.add()            # addition
np.subtract()       # subtraction
np.multiply()       # multiplication
np.divide()         # division
np.power()          # exponentiation (x^n)
np.mod()            # modulus (remainder)
np.sqrt()           # square root
np.abs()            # absolute value
np.exp()            # e^x — used in ML activation functions
np.log()            # ⭐ natural log — used in loss functions
np.log2()           # log base 2
np.log10()          # log base 10
np.sign()           # sign of each element (+1, 0, -1)
np.floor()          # round down to nearest integer
np.ceil()           # round up to nearest integer
np.round()          # round to specified decimal places
np.clip()           # ⭐ limit values to a range — important in ML
```

---

## 5️⃣ Statistics ← Most Important for DS

> The heart of Data Analysis. These functions are used every single day.

np.mean()           # ⭐ average value
np.median()         # ⭐ middle value (robust to outliers)
np.std()            # ⭐ standard deviation
np.var()            # variance
np.min()            # minimum value
np.max()            # maximum value
np.sum()            # sum of all elements
np.cumsum()         # cumulative sum
np.cumprod()        # cumulative product
np.percentile()     # ⭐ percentile — used in outlier detection
np.quantile()       # quantile value (0.0 to 1.0 scale)
np.corrcoef()       # ⭐ correlation matrix — heavily used in EDA
np.cov()            # covariance matrix
np.histogram()      # frequency distribution of values
np.bincount()       # count occurrences of each integer
np.unique()         # ⭐ unique values in array
np.count_nonzero()  # count non-zero elements
```

---

## 6️⃣ Array Manipulation

> Reshape, combine, and restructure your data.

```python
np.reshape()        # change array dimensions
np.resize()         # resize array (repeats data if needed)
np.append()         # add elements to end of array
np.insert()         # insert elements at specific position
np.delete()         # delete elements by index
np.concatenate()    # ⭐ join arrays along existing axis
np.stack()          # stack arrays along a new axis
np.hstack()         # ⭐ horizontal stack (column-wise join)
np.vstack()         # ⭐ vertical stack (row-wise join)
np.split()          # split array into sub-arrays
np.hsplit()         # split array horizontally
np.vsplit()         # split array vertically
np.flip()           # reverse array elements
np.roll()           # shift elements along an axis
np.squeeze()        # ⭐ remove size-1 dimensions — common in ML
np.expand_dims()    # ⭐ add a new dimension — common in ML
```

---

## 7️⃣ Sorting & Searching

> Find, rank, and organize data efficiently.

np.sort()           # sort array elements
np.argsort()        # ⭐ indices that would sort the array
np.argmin()         # ⭐ index of the minimum value
np.argmax()         # ⭐ index of the maximum value
np.searchsorted()   # find position to insert value (binary search)
np.where()          # find indices matching a condition
np.nonzero()        # indices of all non-zero elements
np.extract()        # extract elements based on a condition
```

---

## 8️⃣ Linear Algebra ← Critical for ML

> Matrix operations that power Machine Learning algorithms.

np.dot()                 # ⭐ dot product — very heavily used
np.matmul()              # ⭐ matrix multiplication (@ operator)
np.linalg.inv()          # matrix inverse
np.linalg.det()          # determinant of a matrix
np.linalg.eig()          # ⭐ eigenvalues & eigenvectors — used in PCA
np.linalg.svd()          # ⭐ singular value decomposition — used in ML
np.linalg.norm()         # vector or matrix norm
np.linalg.solve()        # solve system of linear equations (Ax = b)
np.linalg.matrix_rank()  # rank of a matrix
np.trace()               # sum of diagonal elements
np.diag()                # extract diagonal or create diagonal matrix
np.cross()               # cross product of two vectors
```

---

## 9️⃣ Random Module

> Generate reproducible random data for experiments and simulations.

np.random.seed()        # ⭐ set seed for reproducibility — always use
np.random.rand()        # uniform distribution [0, 1)
np.random.randn()       # standard normal distribution
np.random.randint()     # random integers in a range
np.random.choice()      # ⭐ random sample from array
np.random.shuffle()     # shuffle array in place
np.random.permutation() # return shuffled copy of array
np.random.normal()      # normal (Gaussian) distribution
np.random.uniform()     # uniform distribution
np.random.binomial()    # binomial distribution
np.random.poisson()     # Poisson distribution
np.random.exponential() # exponential distribution
```

---

## 🔟 Data Cleaning & Handling

> Handle missing values and invalid data — essential for real-world datasets.

np.isnan()          # ⭐ check for NaN values — used daily
np.isinf()          # check for infinity values
np.isfinite()       # check that values are finite numbers
np.nan_to_num()     # replace NaN and inf with numbers
np.nanmean()        # ⭐ mean — ignores NaN values
np.nanmedian()      # median — ignores NaN values
np.nanstd()         # standard deviation — ignores NaN
np.nansum()         # sum — ignores NaN values
np.nanmax()         # maximum — ignores NaN values
np.nanmin()         # minimum — ignores NaN values
```

---

## 1️⃣1️⃣ Boolean Operations

> Apply logic across entire arrays in one line.

np.any()            # True if any element is True
np.all()            # True if all elements are True
np.logical_and()    # element-wise AND
np.logical_or()     # element-wise OR
np.logical_not()    # element-wise NOT
np.isin()           # ⭐ check membership in a list — used in filtering
```

---

## 1️⃣2️⃣ Set Operations

> Work with unique values and compare datasets.

np.unique()         # unique elements (sorted)
np.intersect1d()    # common elements between two arrays
np.union1d()        # all unique elements from both arrays
np.setdiff1d()      # elements in A but not in B
np.in1d()           # test if elements of A are in B
```

---

## ⭐ Top 20 Functions Used Daily in DS

> Bookmark this. These are the functions you'll use in almost every project.

| Rank | Function | Use Case |
|------|----------|----------|
| 1  | `np.array()`         | Create arrays from data        |
| 2  | `np.mean()`          | Calculate average              |
| 3  | `np.std()`           | Measure spread of data         |
| 4  | `np.where()`         | Conditional operations         |
| 5  | `arr[:, col]`        | Access entire column           |
| 6  | `np.reshape()`       | Shape arrays for ML models     |
| 7  | `np.dot()`           | Matrix math                    |
| 8  | `np.isnan()`         | Find missing values            |
| 9  | `np.nanmean()`       | Mean while ignoring NaN        |
| 10 | `np.corrcoef()`      | Correlation between features   |
| 11 | `np.unique()`        | Get unique values              |
| 12 | `np.argmax()`        | Find best prediction index     |
| 13 | `np.random.seed()`   | Ensure reproducibility         |
| 14 | `np.concatenate()`   | Join datasets together         |
| 15 | `np.percentile()`    | Outlier detection              |
| 16 | `np.log()`           | Used in ML loss functions      |
| 17 | `np.clip()`          | Bound prediction values        |
| 18 | `np.linalg.eig()`    | PCA & dimensionality reduction |
| 19 | `np.squeeze()`       | Fix shape errors in ML         |
| 20 | `np.isin()`          | Filter rows by value list      |

---

## 🚀 Getting Started

### Install NumPy

pip install numpy
```

### Import Convention

import numpy as np   # always use this alias
```

### Quick Test

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))        # 3.0
print(np.std(arr))         # 1.414...
print(arr[arr > 3])        # [4 5]
print(np.where(arr > 3))   # (array([3, 4]),)
```

---

## 💡 Pro Tips

- Always set `np.random.seed(42)` at the start of your notebook for reproducible results
- Prefer `np.nanmean()` over `np.mean()` when working with real-world datasets that may have missing values
- Use **boolean indexing** `arr[arr > value]` instead of loops — it's 100x faster
- Master `reshape()`, `squeeze()`, and `expand_dims()` early — they solve 90% of shape errors in ML
- `np.where(condition, x, y)` is a vectorized if-else — use it everywhere

---

