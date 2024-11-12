#################################################################################
# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################

# Visualizing and Communicating Errors in Rendered Images
# Ray Tracing Gems II, 2021,
# by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
# Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

# Visualizing Errors in Rendered High Dynamic Range Images
# Eurographics 2021,
# by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
# Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics 2020,
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
# Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
# Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

# Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

import subprocess
import os
import sys
import numpy as np

def get_time(string):
    colon_pos = string.find(':')
    sec_pos = string.find(' seconds')
    return float(string[colon_pos + 2:sec_pos])

def huber_estimate(samples, sigma_hat, k = 1.345, num_iter = 10):
	# https://www.sciencedirect.com/topics/mathematics/location-estimator#:~:text=The%20most%20popular%20robust%20location,not%20change%20the%20median%20much.
	# k from https://cran.r-project.org/web/packages/robustbase/vignettes/psi_functions.pdf
	mu_hat = np.median(samples)
	for _ in range(num_iter):
		z = (samples - mu_hat) / sigma_hat
		nom = np.minimum(k, np.maximum(-k, z)).sum()
		denom = np.where(abs(z) > k, 0, 1).sum()
		mu_hat = mu_hat + sigma_hat * nom / denom
	return mu_hat

def compute_stable_average(samples):
    samples_std = 1.4826 * np.median(abs(samples - np.median(samples)))
    return huber_estimate(samples, samples_std)



if __name__ == '__main__':
    """
    Test script. Runs FLIP for both LDR and HDR using one of CUDA/CPP/PYTHON based on the commandline argument.
    Both the mean FLIP is tested and the pixel values from the resulting FLIP images.
    """

    if(len(sys.argv) != 3):
        print("Usage: python test.py numLoops whichAlgorithms")
        print("   whichAlgorithms (>=1 and <=15) is a sum of:")
        print("       LDR-CPP : 1")
        print("       HDR-CPP : 2")
        print("       LDR-CUDA: 4")
        print("       HDR-CUDA: 8")
        sys.exit()

    num_loops = int(sys.argv[1])    
    which_algorithms = int(sys.argv[2])

    mask = 1;

    print("Num loops = ", num_loops)
    for i in range(4):
        skip = True
        if i == 0 and which_algorithms & mask:
            cmd = "../cpp/x64/release/flip.exe --reference ../../images/reference.png --test ../../images/test.png"
            helper_string = " CPP-LDR"
            skip = False
        elif i == 1 and which_algorithms & mask:
            cmd = "../cpp/x64/release/flip.exe --reference ../../images/reference.exr --test ../../images/test.exr --no-exposure-map"
            helper_string = " CPP-HDR"
            skip = False
        elif i == 2 and which_algorithms & mask:
            cmd = "../cpp/x64/release/flip-cuda.exe --reference ../../images/reference.png --test ../../images/test.png"
            helper_string = "CUDA-LDR"
            skip = False
        elif i == 3 and which_algorithms & mask:
            cmd = "../cpp/x64/release/flip-cuda.exe --reference ../../images/reference.exr --test ../../images/test.exr --no-exposure-map"
            helper_string = "CUDA-HDR"
            skip = False
        mask = mask << 1

        if not skip:
            sum_FLIP_time = [] 
            sum_total_time = []
            for i in range(num_loops):
                process_results = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
                result_strings = process_results.stdout.split('\n')

                if helper_string[helper_string.find('-') + 1:] == "LDR":
                    FLIP_time = get_time(result_strings[10])
                    total_time = get_time(result_strings[13])
                else:
                    FLIP_time = get_time(result_strings[10+4])
                    total_time = get_time(result_strings[13+4])
                sum_FLIP_time.append(FLIP_time)
                sum_total_time.append(total_time)

            #print(f"{helper_string}-FLIP evaluation time: {sum_FLIP_time / num_loops:.4f} seconds.")
            #print(f"{helper_string}-FLIP total time     : {sum_total_time / num_loops:.4f} seconds.")

            sum_FLIP_time = np.array(sum_FLIP_time)
            sum_total_time = np.array(sum_total_time)

#            print(f"{np.average(sum_FLIP_time):.4f}  {compute_stable_average(sum_FLIP_time):.4f}")
#            print(f"{np.average(sum_total_time):.4f}  {compute_stable_average(sum_total_time):.4f}")
            print(f"{compute_stable_average(sum_FLIP_time):.4f}")
            print(f"{compute_stable_average(sum_total_time):.4f}")





