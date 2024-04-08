# ap-vast-unofficial
An unofficial implementation of ap-vast based on the article T. Lee et al. "Signal-Adaptive and Perceptually Optimized Sound Zones With Variable Span Trade-Off Filters", IEEE/ACM Transactions on Audio, Speech, and Language Processing, Vol. 28, 2020, pp. 2412-2426. DOI: 10.1109/TASLP.2020.3013397

Please note that the jdiag.m function in the MATLAB implementation for calculating the joint diagonalization of the two correlation matrices, is copied from the original work of the authors of the article cited above (https://github.com/nightmoonbridge/vast_dft). That work is under the BSD-2-Clause license, which has been included into the description of the function.

This repo presents a brute-force implementation of the ap-vast method as described in the cited article. This is used in our own publication work (link to publication will follow) as a reference method for comparison. The presented implementation is not fast by any means. For inspirations regarding fast implementations, people are encouraged to go check out https://github.com/nightmoonbridge/vast_dft and the associated publication.

# Existing patents
While the code for this unofficial implementation is available in the efforts of promoting open science, people should be aware of the following patents associated with the authors of the original publication:
 - EP 3 797 528: https://patents.google.com/patent/EP3797528A1/en?oq=ep3797528
 - US 2021/0235213: https://patents.google.com/patent/US20210235213A1/en?oq=us20210235213
 - US 11,516,614: https://patents.google.com/patent/US11516614B2/en?oq=us11516614
