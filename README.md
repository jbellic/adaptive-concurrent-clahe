# Adaptive Concurrent CLAHE
Very fast contrast limited adaptive histogram equalization (CLAHE) implementation for Java based on OpenCV
------------------------------------------------------
CLAHE algorithm with option for adaptive check if image to process is suitable for CLAHE by determining the darkness of an image. Implementation is fully threadsafe, multithreaded and much faster than the initial C++ port for OpenCV 2.4.9. Early benchmarks with 4000x6000 pixel images and default settings shows up to ~2.66x speed increase. This implementation is used within the project **Katib-Engine**, a fast multi layered image search engine with high accuracy.

### Usage example with Clip Limit = 1

![AC-CLAHE](https://raw.githubusercontent.com/jbellic/adaptive-concurrent-clahe/master/ac-clahe.jpg)

### Adaptive Check

Allows the user to check if image candidate is suitable for CLAHE application by analysing image pixels for darkness by utilizing histogram. If image contains x percent amount of too dark pixels, where darkness is defined by a custom threshold value, then CLAHE is applied.

### Options

CLAHE: Color/Gray output, Clip Limit  
Adaptive Check: Pixel Brightness Threshold, Dark Pixels Percentage Threshold

### Cloning GIT Repository

```
git clone https://github.com/jbellic/adaptive-concurrent-clahe.git
```


