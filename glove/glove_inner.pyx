#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log, pow, sqrt

ctypedef np.float64_t REAL_t
ctypedef np.uint32_t  INT_t

cdef void train_glove_thread(
        REAL_t * W,       REAL_t * ContextW,
        REAL_t * gradsqW, REAL_t * gradsqContextW,
        REAL_t * bias,       REAL_t * ContextB,
        REAL_t * gradsqb, REAL_t * gradsqContextB,
        REAL_t * error,
        INT_t * job_key, INT_t * job_subkey, REAL_t * job_target,
        int vector_size, int batch_size, REAL_t x_max, REAL_t alpha, REAL_t step_size) nogil:

    cdef long long a, b, l1, l2
    cdef int example_idx = 0
    cdef REAL_t temp1, temp2, diff, fdiff

    for example_idx in range(batch_size):
        # Calculate cost, save diff for gradients
        l1 = job_key[example_idx]    * vector_size # cr word indices start at 1
        l2 = job_subkey[example_idx] * vector_size

        diff = 0.0;
        for b in range(vector_size):
            diff += W[b + l1] * ContextW[b + l2] # dot product of word and context word vector
        diff += bias[job_key[example_idx]] + ContextB[job_subkey[example_idx]] - log(job_target[example_idx]) # add separate bias for each word
        fdiff = diff if (job_target[example_idx] > x_max) else pow(job_target[example_idx] / x_max, alpha) * diff # multiply weighting function (f) with diff
        error[0] += 0.5 * fdiff * diff # weighted squared error

        # # Adaptive gradient updates
        fdiff *= step_size # for ease in calculating gradient
        for b in range(vector_size):
            # learning rate times gradient for word vectors
            temp1 = fdiff * ContextW[b + l2]
            temp2 = fdiff * W[b + l1]
            # adaptive updates
            W[b + l1]              -= (temp1 / sqrt(gradsqW[b + l1]))
            ContextW[b + l2]       -= (temp2 / sqrt(gradsqContextW[b + l2]))
            gradsqW[b + l1]        += temp1 * temp1
            gradsqContextW[b + l2] += temp2 * temp2
        # updates for bias terms
        bias[job_key[example_idx]]        -= fdiff / sqrt(gradsqb[job_key[example_idx]]);
        ContextB[job_subkey[example_idx]] -= fdiff / sqrt(gradsqContextB[job_subkey[example_idx]]);

        fdiff *= fdiff;
        gradsqb[job_key[example_idx]]           += fdiff
        gradsqContextB[job_subkey[example_idx]] += fdiff

def train_glove(model, jobs, float _step_size, _error):
    cdef REAL_t *W              = <REAL_t *>(np.PyArray_DATA(model.W))
    cdef REAL_t *ContextW       = <REAL_t *>(np.PyArray_DATA(model.ContextW))
    cdef REAL_t *gradsqW        = <REAL_t *>(np.PyArray_DATA(model.gradsqW))
    cdef REAL_t *gradsqContextW = <REAL_t *>(np.PyArray_DATA(model.gradsqContextW))

    cdef REAL_t *b              = <REAL_t *>(np.PyArray_DATA(model.b))
    cdef REAL_t *ContextB       = <REAL_t *>(np.PyArray_DATA(model.ContextB))
    cdef REAL_t *gradsqb        = <REAL_t *>(np.PyArray_DATA(model.gradsqb))
    cdef REAL_t *gradsqContextB = <REAL_t *>(np.PyArray_DATA(model.gradsqContextB))

    cdef REAL_t *error          = <REAL_t *>(np.PyArray_DATA(_error))

    cdef INT_t  *job_key        = <INT_t  *>(np.PyArray_DATA(jobs[0]))
    cdef INT_t  *job_subkey     = <INT_t  *>(np.PyArray_DATA(jobs[1]))
    cdef REAL_t *job_target     = <REAL_t *>(np.PyArray_DATA(jobs[2]))

    # configuration and parameters
    cdef REAL_t step_size = _step_size
    cdef int vector_size = model.d
    cdef int batch_size = len(jobs[0])
    cdef REAL_t x_max   = model.x_max
    cdef REAL_t alpha   = model.alpha

    # release GIL & train on the sentence
    with nogil:
        train_glove_thread(
            W,\
            ContextW,\
            gradsqW,\
            gradsqContextW,\
            b,\
            ContextB,\
            gradsqb,\
            gradsqContextB,\
            error,\
            job_key,\
            job_subkey,\
            job_target, \
            vector_size,\
            batch_size, \
            x_max, \
            alpha, \
            step_size
        )
