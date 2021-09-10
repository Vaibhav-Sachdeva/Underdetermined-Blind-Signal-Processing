# Underdetermined-Blind-Signal-Processing
Blind Source Separation (BSS) is the separation of a set of source signals from a set of mixed signals, without the aid of information (or with very little information) about the
source signals or the mixing process. 

## Problem Statement 
The goal here is to identify the mixing matrix A by Blind Mixing model Recovery (BMMR) and the n-dimensional source random vectors by Blind Source Recovery (BSR) from an observed mixture random vector x where x is:
<p align="center">

  
                                                                         x = As
                                                        where,
                                                        x = m-dimensional mixture random vector
                                                        s = n-dimensional source random vector with component density p(s)
                                                        A = mixing matrix having independent columns each of unit norm
</p>
More sources are to be extracted from less observed mixtures in Underdetermined Blind Source Separation (n < m) without knowing both the sources and the mixing matrix. Blind Source Separation (BSS) is predominantly based on independent source assumptions.
