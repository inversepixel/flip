# A note about precision

We have several different implementations of FLIP (Python, PyTorch, C++, and CUDA) and we have tried to make
the implementations as similar as possible. However, there are several facts about these that make it very hard
to get perfect matches between the implementations.
These include:
1. Our computations are made using 32-bit floating-point arithmetic.
2. The order of operations matter, with respect to the result.
    * We are using different versions of functions in the different versions of FLIP, and these may not all use similar implementations.
      Even computing the mean of an array can give different results because of this.
    * As an example, if a 2D filter implementation's outer loop is on `x` and the inner loop is on `y`, that will in the majority
      of cases give a different floating-point result compared to have the outer loop be `y` and the inner `x`.
4. GPUs attempt to try to use fused multiply-and-add (FMA) operations, i.e., `a*b+c`, as much as possible. These are faster, but the entire
   operation is also computed at higher precision. Since the CPU implementation may not use FMA, this is another source of difference
   between implementations.
5. Depending on compiler flags, `sqrt()` may be computed using lower precision on GPUs.
6. For the C++ and CUDA implementations, we have changed to using separated filters for faster performance.
   This has given rise to small differences compared to previous versions. For our tests,
   we have therefore updated the `images/correct_{ldr|hdr}flip_{cpp|cuda}.{png|exr}` images.

That said, we have tried to make the results of our different implementations as close to each other as we could. There may still be differences.

Furthermore, the Python version of FLIP, installed using `pip install flip_evaluator`, runs on Windows, Linux (tested on Ubuntu 24.04),
and OS X ($\ge$ 10.15). However, its output sometimes differ slightly between the different operative systems.
The references used for `flip_evaluator/tests/test.py` are made for Windows. While the mean tests (means compared up to six decimal points)
pass on each mentioned operative system, not all error map pixels are identical.
