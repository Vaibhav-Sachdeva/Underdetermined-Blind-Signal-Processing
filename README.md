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
More sources are to be extracted from less observed mixtures in Underdetermined Blind Source Separation (n < m) without knowing both the sources and the mixing matrix. Blind Source Separation (BSS) is predominantly based on independent source assumptions.Assuming that statistically independent sources have at most one Gaussian variable,
it is known that A is uniquely determined by x. The most commonly used algorithms are based on sparse sources that are correctly identified by a clustering of k-means. But, meanbased clustering can only classify the appropriate A if the data density approaches a delta 1 distribution.

## Methods 
- Geometric Matrix Recovery
- Blind Source Recovery (BSR)
- K-Means
- K-medoids

## Results
<p align ="center">
<img src="https://github.com/Vaibhav-Sachdeva/Vaibhav-Sachdeva/blob/main/Images/digi_1.PNG" alt="centered image" height="598" width="398">

K-medoids seems to be a better model here than K-means, since it gives more consistent result and is more robust to noise and outliers. Although, there are few cases in which K-means performs better than K-medoids, but consistency wise K-medoids is much better.

