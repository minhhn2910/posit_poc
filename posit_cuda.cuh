//#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <fstream>

//#include "fp16.hpp"
//#include "fp16.cuh"
//#include "posit_constants.hpp"
#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t

#define _G_NBITS 8
#define _G_ESIZE 2

#define SIGN_MASK 0x8000
#define FLOAT_SIGN_MASK 0x80000000
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF
#define SECOND_BIT_MASK 0x4000
#define POSIT_INF 0x0000
#define POSIT_LIMB_ALL_BITS_SET 0xffff
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_SIZE 32
#define FLOAT_EXPONENT_MASK 0x7f800000
#define FLOAT_FRACTION_MASK 0x007fffff
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8
#define TEMP_TYPE uint64_t
#define UNSIGNED_LONG_LONG_SIZE 64
#define EDP_ACC_SIZE 63
#define POSIT_EXP_SHIFT 41 //64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define POSIT_LENGTH_PLUS_ONE 17

#define GET_MAX(a, b)                                                          \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define _G_INFP 32768

#if _G_NBITS == 16
#define _G_POSIT_SHIFT_AMOUNT 0
#define _G_MAXREALP 32767
#define _G_MINREALP 1
#define POSIT_EXTRA_BITS_SHIFT 49 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x0000FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0001000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 2.684354560e+8
#define _G_MINREAL 3.725290298e-9
#define _G_MAXREAL_INT 0x8D800000
#define _G_MINREAL_INT 0x31800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 7.205759e+16
#define _G_MINREAL 1.387779e-17
#define _G_MAXREAL_INT 0x5B800000
#define _G_MINREAL_INT 0x23800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 5.192296859e+33
#define _G_MINREAL 1.925929944e-34
#define _G_MAXREAL_INT 0x77800000
#define _G_MINREAL_INT 0x07800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 2.695994667e+67
#define _G_MINREAL 3.709206151e-68

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_POSIT_SHIFT_AMOUNT (FP16_LIMB_SIZE - _G_NBITS)
#define _G_MAXREALP ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT
#define _G_MINREALP (1 << _G_POSIT_SHIFT_AMOUNT)
#define _G_INFP 1 << (FP16_LIMB_SIZE - 1)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 8
#define _G_POSIT_SHIFT_AMOUNT 8
#define _G_MAXREALP 32512
#define _G_MINREALP 256
#define POSIT_EXTRA_BITS_SHIFT 57
#define POSIT_EXTRA_BITS_MASK 0x00FFFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0100000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 4096
#define _G_MINREAL 0.0002441406250
#define _G_MAXREAL_INT 0x45800000
#define _G_MINREAL_INT 0x39800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 1.677721600e+7
#define _G_MINREAL 5.960464478e-8
#define _G_MAXREAL_INT 0x4B800000
#define _G_MINREAL_INT 0x33800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 2.814749767e+14
#define _G_MINREAL 3.552713679e-15
#define _G_MAXREAL_INT 0x57800000
#define _G_MINREAL_INT 0x27800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 7.922816251e+28
#define _G_MINREAL 1.262177448e-29

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif


#elif _G_NBITS == 4
#define _G_POSIT_SHIFT_AMOUNT 12
#define _G_MAXREALP ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT
#define _G_MINREALP (1 << _G_POSIT_SHIFT_AMOUNT)
#define POSIT_EXTRA_BITS_SHIFT (64 - _G_NBITS + 1)
#define POSIT_EXTRA_BITS_MASK 0x0FFFFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x1000000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 16
#define _G_MINREAL 0.0625
#define _G_MAXREAL_INT 0x41800000
#define _G_MINREAL_INT 0x3d800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 256
#define _G_MINREAL 0.00390625
#define _G_MAXREAL_INT 0x43800000
#define _G_MINREAL_INT 0x3b800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 65536
#define _G_MINREAL 0.00001525878
#define _G_MAXREAL_INT 0x47800000
#define _G_MINREAL_INT 0x377ffff6

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif



#elif _G_NBITS == 6
#define _G_POSIT_SHIFT_AMOUNT 10
#define _G_MAXREALP ((1 << (_G_NBITS - 1)) - 1) << _G_POSIT_SHIFT_AMOUNT
#define _G_MINREALP (1 << _G_POSIT_SHIFT_AMOUNT)
#define POSIT_EXTRA_BITS_SHIFT (64 - _G_NBITS + 1)
#define POSIT_EXTRA_BITS_MASK 0x03FFFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0400000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define POSIT_EXPONENT_MASK 1
#define _G_MAXREAL 256
#define _G_MINREAL 0.00390625
#define _G_MAXREAL_INT 0x43800000
#define _G_MINREAL_INT 0x3b800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 65536
#define _G_MINREAL 0.00001525878
#define _G_MAXREAL_INT 0x47800000
#define _G_MINREAL_INT 0x377ffff6

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 4294967296
#define _G_MINREAL 2.3283064e-10
#define _G_MAXREAL_INT 0x4f800000
#define _G_MINREAL_INT 0x2f800000

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 9
#define _G_POSIT_SHIFT_AMOUNT 7
#define _G_MAXREALP 32640
#define _G_MINREALP 128
#define POSIT_EXTRA_BITS_SHIFT 56 // 64 - _G_NBITS + 1
#define POSIT_EXTRA_BITS_MASK 0x007FFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0080000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 16384
#define _G_MINREAL 0.00006103515625
#define _G_MAXREAL_INT 0x46800000
#define _G_MINREAL_INT 0x38800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 2.684354560e+8
#define _G_MINREAL 3.725290298e-9
#define _G_MAXREAL_INT 0x4D800000
#define _G_MINREAL_INT 0x31800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 7.205759404e+16
#define _G_MINREAL 1.387778781e-17
#define _G_MAXREAL_INT 0x5B800000
#define _G_MINREAL_INT 0x23800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 5.192296859e+33
#define _G_MINREAL 1.925929944e-34

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 10
#define _G_POSIT_SHIFT_AMOUNT 6
#define _G_MAXREALP 32704
#define _G_MINREALP 64
#define POSIT_EXTRA_BITS_SHIFT 55
#define POSIT_EXTRA_BITS_MASK 0x003FFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0040000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 65536
#define POSIT_EXPONENT_MASK 1
#define _G_MINREAL 0.00001525878906
#define _G_MAXREAL_INT 0x47800000
#define _G_MINREAL_INT 0x37800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define POSIT_EXPONENT_MASK 3
#define _G_MAXREAL 4.294967296e+9
#define _G_MINREAL 2.328306437e-10
#define _G_MAXREAL_INT 0x4F800000
#define _G_MINREAL_INT 0x2F800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define POSIT_EXPONENT_MASK 7
#define _G_MAXREAL 1.844674407e+19
#define _G_MINREAL 5.421010862e-20
#define _G_MAXREAL_INT 0x5F800000
#define _G_MINREAL_INT 0x1F800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 3.402823669e+38
#define _G_MINREAL 2.938735877e-39

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 11
#define _G_POSIT_SHIFT_AMOUNT 5
#define _G_MAXREALP 32736
#define _G_MINREALP 32
#define POSIT_EXTRA_BITS_SHIFT 54
#define POSIT_EXTRA_BITS_MASK 0x001FFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0020000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 262144
#define _G_MINREAL 3.814697266e-6
#define _G_MAXREAL_INT 0x48800000
#define _G_MINREAL_INT 0x36800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 6.871947674e+10
#define _G_MINREAL 1.455191523e-11
#define _G_MAXREAL_INT 0x51800000
#define _G_MINREAL_INT 0x2D800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 4.722366483e+21
#define _G_MINREAL 2.117582368e-22
#define _G_MAXREAL_INT 0x63800000
#define _G_MINREAL_INT 0x1B800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 2.230074520e+43
#define _G_MINREAL 4.484155086e-44

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 12
#define _G_POSIT_SHIFT_AMOUNT 4
#define _G_MAXREALP 32752
#define _G_MINREALP 16
#define POSIT_EXTRA_BITS_SHIFT 53
#define POSIT_EXTRA_BITS_MASK 0x000FFFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0010000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 1.048576000e+6
#define _G_MINREAL 9.536743164e-7
#define _G_MAXREAL_INT 0x49800000
#define _G_MINREAL_INT 0x35800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 1.099511628e+12
#define _G_MINREAL 9.094947018e-13
#define _G_MAXREAL_INT 0x53800000
#define _G_MINREAL_INT 0x2B800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 1.208925820e+24
#define _G_MINREAL 8.271806126e-25
#define _G_MAXREAL_INT 0x67800000
#define _G_MINREAL_INT 0x17800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 1.461501637e+48
#define _G_MINREAL 6.842277658e-49

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 13
#define _G_POSIT_SHIFT_AMOUNT 3
#define _G_MAXREALP 32760
#define _G_MINREALP 8
#define POSIT_EXTRA_BITS_SHIFT 52
#define POSIT_EXTRA_BITS_MASK 0x0007FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0008000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 4.194304000e+6
#define _G_MINREAL 2.384185791e-7
#define _G_MAXREAL_INT 0x4A800000
#define _G_MINREAL_INT 0x34800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 1.759218604e+13
#define _G_MINREAL 5.684341886e-14
#define _G_MAXREAL_INT 0x55800000
#define _G_MINREAL_INT 0x29800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 3.094850098e+26
#define _G_MINREAL 3.231174268e-27
#define _G_MAXREAL_INT 0x6B800000
#define _G_MINREAL_INT 0x13800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 9.578097130e+52
#define _G_MINREAL 1.044048715e-53

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 14
#define _G_POSIT_SHIFT_AMOUNT 2
#define _G_MAXREALP 32764
#define _G_MINREALP 4
#define POSIT_EXTRA_BITS_SHIFT 51
#define POSIT_EXTRA_BITS_MASK 0x0003FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0004000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 1.677721600e+7
#define _G_MINREAL 5.960464478e-8
#define _G_MAXREAL_INT 0x4B800000
#define _G_MINREAL_INT 0x33800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 2.814749767e+14
#define _G_MINREAL 3.552713679e-15
#define _G_MAXREAL_INT 0x57800000
#define _G_MINREAL_INT 0x27800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 7.922816251e+28
#define _G_MINREAL 1.262177448e-29
#define _G_MAXREAL_INT 0x6F800000
#define _G_MINREAL_INT 0x0F800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 6.277101735e+57
#define _G_MINREAL 1.593091911e-58

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#elif _G_NBITS == 15
#define _G_POSIT_SHIFT_AMOUNT 1
#define _G_MAXREALP 32766
#define _G_MINREALP 2
#define POSIT_EXTRA_BITS_SHIFT 50
#define POSIT_EXTRA_BITS_MASK 0x0001FFFFFFFFFFFF
#define POSIT_HALFWAY_BIT_MASK 0x0002000000000000

#if _G_ESIZE == 1
#define _G_USEED 4
#define _G_USEED_ZEROS 2
#define _G_MAXREAL 6.710886400e+7
#define _G_MINREAL 1.490116119e-8
#define _G_MAXREAL_INT 0x4C800000
#define _G_MINREAL_INT 0x32800000

#elif _G_ESIZE == 2
#define _G_USEED 16
#define _G_USEED_ZEROS 4
#define _G_MAXREAL 4.503599627e+15
#define _G_MINREAL 2.220446049e-16
#define _G_MAXREAL_INT 0x59800000
#define _G_MINREAL_INT 0x25800000

#elif _G_ESIZE == 3
#define _G_USEED 256
#define _G_USEED_ZEROS 8
#define _G_MAXREAL 2.028240960e+31
#define _G_MINREAL 4.930380658e-32
#define _G_MAXREAL_INT 0x73800000
#define _G_MINREAL_INT 0x0B800000

#elif _G_ESIZE == 4
#define _G_USEED 512
#define _G_USEED_ZEROS 16
#define _G_MAXREAL 4.113761393e+62
#define _G_MINREAL 2.430865343e-63

#else
#define _G_USEED 1 << (1 << _G_ESIZE)
#define _G_USEED_ZEROS (1 << _G_ESIZE)
#define _G_MAXREAL pow(_G_USEED, (_G_NBITS - 2))
#define _G_MINREAL (1 / pow(_G_USEED, (_G_NBITS - 2)))
#endif

#endif

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

typedef FP16_TYPE fp16;

__device__ __inline__ float fp16tofp32_gpu(fp16 p) {
  union Bits v;

  // get sign
  bool sign = p & SIGN_MASK;
  p = (p ^ -sign) + sign;

  // get the regime sign
  bool regime_sign = p & SECOND_BIT_MASK;

  // get regime
  v.ui = p << POSIT_LENGTH_PLUS_ONE;
  int regime_length = (__clz(v.ui) & -!regime_sign) + (__clz(~v.ui) & -regime_sign);
  int regime = (regime_length - regime_sign) << _G_ESIZE;
  regime = (regime ^ -regime_sign) + regime_sign;

  // assemble
  v.ui <<= (regime_length + 1);
  v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
  v.ui += ((SINGLE_PRECISION_BIAS - regime) << FLOAT_EXPONENT_SHIFT);

  v.si ^= (FLOAT_INF ^ v.si) & -(p == _G_INFP);
  v.si ^= (0 ^ v.si) & -(p == 0);

  v.ui |= (sign << FLOAT_SIGN_SHIFT);
  return v.f;

}

__device__ __inline__ fp16 fp32tofp16_gpu(float f) {
  fp16 p = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;

  p = _G_MAXREALP & -(v.si >= _G_MAXREAL_INT);
  p = _G_INFP & -(v.si >= FLOAT_INF);
  p = _G_MINREALP & -(v.si <= _G_MINREAL_INT);

  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_ESIZE + 1)) | (exp & POSIT_EXPONENT_MASK);;
  //if exponent is negative
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign) >> ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & POSIT_EXPONENT_MASK))) & (bool) exp);

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (POSIT_EXP_SHIFT - regime_and_exp_length));
  fp16 temp_p = (regime_and_exp >> POSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & POSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & POSIT_EXTRA_BITS_MASK));
#if _G_NBITS != 16
  temp_p <<= _G_POSIT_SHIFT_AMOUNT;
#endif
  p = temp_p & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  p = (p ^ -sign) + sign;

  return p;
}


/*
//template <typename scalar_t>
__global__ void posit_cuda_kernel(
     float* input,
    size_t input_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    fp16 temp = fp32tofp16_gpu(input[index]);
    input[index] = fp16tofp32_gpu(temp);

  }
}

__global__ void p2f_cuda_kernel(
     uint_8* input,
     float *output,
    size_t input_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    fp16 temp = input[index] << 8; //fp32tofp16_gpu(input[index]);
    output[index] = fp16tofp32_gpu(temp);

  }
}
__global__ void f2p_cuda_kernel(
     float* input,
     uint16_t* output, 
    size_t input_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    fp16 temp = fp32tofp16_gpu(input[index]);
    output[index] = temp;

  }
}
*/
/*
torch::Tensor posit_cuda(
    torch::Tensor input) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  //auto output = torch::ones_like(input);
//  std::cout<< " posit format "<< _G_NBITS << " " << _G_ESIZE << "\n";
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

//  AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    posit_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
        input_size );
//  }));



  return input;
}
*/
