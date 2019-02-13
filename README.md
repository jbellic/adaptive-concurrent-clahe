# Adaptive Concurrent CLAHE
Very fast contrast limited adaptive histogram equalization (CLAHE) implementation for Java based on OpenCV
------------------------------------------------------
## Introduction

with option for adaptive check if image to process is suitable for CLAHE. Implementation is fully threadsafe, multithreaded and much faster than the C++ port / original implementation. Early benchmarks with 4000x6000 pixel images and default settings shows up to ~2.66x speed increase.

## Cloning GIT Repository

The bleeding edge source code can be obtained by cloning the git repository.

```
git clone https://github.com/jbellic/adaptive-concurrent-clahe.git
```
