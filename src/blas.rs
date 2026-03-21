#[cfg(any(feature = "accelerate", feature = "blas"))]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// C = alpha * op(A) * op(B) + beta * C
///
/// All matrices are row-major.
/// A is m x k (or k x m if transposed), B is k x n (or n x k if transposed), C is m x n.
#[cfg(any(feature = "accelerate", feature = "blas"))]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    trans_a: bool,
    b: &[f32],
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            if trans_a { TRANS } else { NO_TRANS },
            if trans_b { TRANS } else { NO_TRANS },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
}

/// Integer GEMM: C_i32[m,n] += A_i16[m,k] * B_i16[k,n]
///
/// Inputs are i16 (zero-point-subtracted quantized values, range -255..255),
/// accumulation is i32. i16 storage is 2x smaller than f32/i32.
/// Result is exact regardless of accumulation order.
///
/// On aarch64, uses NEON vmlal_s16 (widening i16×i16→i32 MAC) to process
/// 8 output columns per inner iteration.
pub fn i16_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[i16],
    lda: usize,
    b: &[i16],
    ldb: usize,
    c: &mut [i32],
    ldc: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        unsafe {
            return i16_gemm_neon(m, n, k, a, lda, b, ldb, c, ldc);
        }
    }
    #[allow(unreachable_code)]
    i16_gemm_scalar(m, n, k, a, lda, b, ldb, c, ldc);
}

fn i16_gemm_scalar(
    m: usize,
    n: usize,
    k: usize,
    a: &[i16],
    lda: usize,
    b: &[i16],
    ldb: usize,
    c: &mut [i32],
    ldc: usize,
) {
    for i in 0..m {
        for pk in 0..k {
            let a_val = a[i * lda + pk] as i32;
            let c_row = &mut c[i * ldc..i * ldc + n];
            let b_row = &b[pk * ldb..pk * ldb + n];
            for j in 0..n {
                c_row[j] += a_val * b_row[j] as i32;
            }
        }
    }
}

/// NEON-accelerated i16 GEMM using vmlal_s16 (widening i16×i16→i32 MAC).
///
/// For each (i, pk) pair: broadcast a[i,pk] to i16x4, load B values as i16,
/// then widening multiply-accumulate into int32x4 accumulators.
/// Processes 8 output columns per inner iteration (two int32x4 registers).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn i16_gemm_neon(
    m: usize,
    n: usize,
    k: usize,
    a: &[i16],
    lda: usize,
    b: &[i16],
    ldb: usize,
    c: &mut [i32],
    ldc: usize,
) {
    use std::arch::aarch64::*;

    for i in 0..m {
        let a_row = &a[i * lda..];
        let c_row = &mut c[i * ldc..];
        let mut j = 0;

        // Process 8 output columns at a time
        while j + 8 <= n {
            // SAFETY: pointers are valid and aligned (NEON handles unaligned)
            let mut acc0 = unsafe { vld1q_s32(c_row[j..].as_ptr()) };
            let mut acc1 = unsafe { vld1q_s32(c_row[j + 4..].as_ptr()) };

            for pk in 0..k {
                unsafe {
                    let a_val = vdup_n_s16(a_row[pk]);
                    let b8 = vld1q_s16(b[(pk * ldb + j)..].as_ptr());
                    acc0 = vmlal_s16(acc0, a_val, vget_low_s16(b8));
                    acc1 = vmlal_s16(acc1, a_val, vget_high_s16(b8));
                }
            }

            unsafe {
                vst1q_s32(c_row[j..].as_mut_ptr(), acc0);
                vst1q_s32(c_row[j + 4..].as_mut_ptr(), acc1);
            }
            j += 8;
        }

        // Process 4 columns at a time
        while j + 4 <= n {
            let mut acc = unsafe { vld1q_s32(c_row[j..].as_ptr()) };
            for pk in 0..k {
                unsafe {
                    let a_val = vdup_n_s16(a_row[pk]);
                    let b4 = vld1_s16(b[(pk * ldb + j)..].as_ptr());
                    acc = vmlal_s16(acc, a_val, b4);
                }
            }
            unsafe { vst1q_s32(c_row[j..].as_mut_ptr(), acc) };
            j += 4;
        }

        // Remaining columns: scalar
        for (pk, &a_val) in a_row.iter().enumerate().take(k) {
            let a_val = a_val as i32;
            let b_base = pk * ldb;
            for jj in j..n {
                c_row[jj] += a_val * b[b_base + jj] as i32;
            }
        }
    }
}

/// Default sgemm using the matrixmultiply crate (optimized SIMD kernels).
#[cfg(not(any(feature = "accelerate", feature = "blas")))]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    trans_a: bool,
    b: &[f32],
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    // matrixmultiply::sgemm uses column/row strides.
    // Row-major A[m,k]: row stride = lda, col stride = 1
    // Transposed A: logical A is A^T, so physical layout has row stride = 1, col stride = lda
    let (a_rs, a_cs) = if trans_a { (1, lda) } else { (lda, 1) };
    let (b_rs, b_cs) = if trans_b { (1, ldb) } else { (ldb, 1) };

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            a_rs as isize,
            a_cs as isize,
            b.as_ptr(),
            b_rs as isize,
            b_cs as isize,
            beta,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}
