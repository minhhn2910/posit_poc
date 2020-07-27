#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

#define FP8_LIMB_SIZE 8
#define FP8_TYPE uint8_t
typedef FP8_TYPE fp8;

union Bits {
	float f;
	int32_t si;
	uint32_t ui;
};

#include <stdio.h> 
  
void printBinary(int n, int i) 
{ 
  
    // Prints the binary representation 
    // of a number n up to i-bits. 
    int k; 
    for (k = i - 1; k >= 0; k--) { 
  
        if ((n >> k) & 1) 
            printf("1"); 
        else
            printf("0"); 
    } 
} 
  
typedef union { 
  
    float f; 
    struct
    { 
  
        // Order is important. 
        // Here the members of the union data structure 
        // use the same memory (32 bits). 
        // The ordering is taken 
        // from the LSB to the MSB. 
        unsigned int mantissa : 23; 
        unsigned int exponent : 8; 
        unsigned int sign : 1; 
  
    } raw; 
} myfloat; 
  
// Function to convert real value 
// to IEEE foating point representation 
void printIEEE(myfloat var) 
{ 
  
    // Prints the IEEE 754 representation 
    // of a float value (32 bits) 
  
    printf("%d | ", var.raw.sign); 
    printBinary(var.raw.exponent, 8); 
    printf(" | "); 
    printBinary(var.raw.mantissa, 23); 
    //printf("\n"); 
} 

#define _G_NBITS 8
#define _G_ESIZE 2
#define SIGN_MASK 0x80
#define SECOND_BIT_MASK 0x40
#define POSIT_TO_INT_SHIFT 25
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9
#define SINGLE_PRECISION_BIAS 127
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_INF 0x7F800000
#define _G_INFP 128
#define FLOAT_SIGN_SHIFT 31

float posit8tofp32(fp8 p) {
	union Bits v;

	// get sign
	bool sign = p & SIGN_MASK;
	p = (p ^ -sign) + sign;

	// get the regime sign
	bool regime_sign = p & SECOND_BIT_MASK;

	// get regime
	v.ui = p << POSIT_TO_INT_SHIFT;
	//int regime_length = (__builtin_clz(v.ui) & -!regime_sign) + (__builtin_clz(~v.ui) & -regime_sign);
	int regime_length;
	  if(regime_sign)
	    regime_length = (__builtin_clz(~v.ui));
	  else
	    regime_length = (__builtin_clz(v.ui));
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

int main() {
    uint16_t lookup_table[256];
	printf("Posit environment - nbits = %d, esize = %d\n", _G_NBITS, _G_ESIZE);
	for (int i = 0; i <= 255; i++) {
         myfloat var; 
        var.f = posit8tofp32(i); 
        printf("posit = %d ", i);
        printIEEE(var); 
        Bits bit_ ;
        bit_.f = var.f;
        uint16_t temp = bit_.ui >> 16;
        lookup_table[i] = temp;
        printf(" %x ",lookup_table[i]);
        printf("\n"); 
		//printf("posit = %d, float value = %e\n", i, );
	}
    printf("{");
    for (int i = 0; i <= 255; i++) 
        printf("0x%x, ",lookup_table[i]);
    printf("}");
    uint16_t new_table[256] = {0x0, 0x3380, 0x3580, 0x3680, 0x3780, 0x3800, 0x3880, 0x3900, 0x3980, 0x39c0, 0x3a00, 0x3a40, 0x3a80, 0x3ac0, 0x3b00, 0x3b40, 0x3b80, 0x3ba0, 0x3bc0, 0x3be0, 0x3c00, 0x3c20, 0x3c40, 0x3c60, 0x3c80, 0x3ca0, 0x3cc0, 0x3ce0, 0x3d00, 0x3d20, 0x3d40, 0x3d60, 0x3d80, 0x3d90, 0x3da0, 0x3db0, 0x3dc0, 0x3dd0, 0x3de0, 0x3df0, 0x3e00, 0x3e10, 0x3e20, 0x3e30, 0x3e40, 0x3e50, 0x3e60, 0x3e70, 0x3e80, 0x3e90, 0x3ea0, 0x3eb0, 0x3ec0, 0x3ed0, 0x3ee0, 0x3ef0, 0x3f00, 0x3f10, 0x3f20, 0x3f30, 0x3f40, 0x3f50, 0x3f60, 0x3f70, 0x3f80, 0x3f90, 0x3fa0, 0x3fb0, 0x3fc0, 0x3fd0, 0x3fe0, 0x3ff0, 0x4000, 0x4010, 0x4020, 0x4030, 0x4040, 0x4050, 0x4060, 0x4070, 0x4080, 0x4090, 0x40a0, 0x40b0, 0x40c0, 0x40d0, 0x40e0, 0x40f0, 0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170, 0x4180, 0x41a0, 0x41c0, 0x41e0, 0x4200, 0x4220, 0x4240, 0x4260, 0x4280, 0x42a0, 0x42c0, 0x42e0, 0x4300, 0x4320, 0x4340, 0x4360, 0x4380, 0x43c0, 0x4400, 0x4440, 0x4480, 0x44c0, 0x4500, 0x4540, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780, 0x4880, 0x4980, 0x4b80, 0xff80, 0xcb80, 0xc980, 0xc880, 0xc780, 0xc700, 0xc680, 0xc600, 0xc580, 0xc540, 0xc500, 0xc4c0, 0xc480, 0xc440, 0xc400, 0xc3c0, 0xc380, 0xc360, 0xc340, 0xc320, 0xc300, 0xc2e0, 0xc2c0, 0xc2a0, 0xc280, 0xc260, 0xc240, 0xc220, 0xc200, 0xc1e0, 0xc1c0, 0xc1a0, 0xc180, 0xc170, 0xc160, 0xc150, 0xc140, 0xc130, 0xc120, 0xc110, 0xc100, 0xc0f0, 0xc0e0, 0xc0d0, 0xc0c0, 0xc0b0, 0xc0a0, 0xc090, 0xc080, 0xc070, 0xc060, 0xc050, 0xc040, 0xc030, 0xc020, 0xc010, 0xc000, 0xbff0, 0xbfe0, 0xbfd0, 0xbfc0, 0xbfb0, 0xbfa0, 0xbf90, 0xbf80, 0xbf70, 0xbf60, 0xbf50, 0xbf40, 0xbf30, 0xbf20, 0xbf10, 0xbf00, 0xbef0, 0xbee0, 0xbed0, 0xbec0, 0xbeb0, 0xbea0, 0xbe90, 0xbe80, 0xbe70, 0xbe60, 0xbe50, 0xbe40, 0xbe30, 0xbe20, 0xbe10, 0xbe00, 0xbdf0, 0xbde0, 0xbdd0, 0xbdc0, 0xbdb0, 0xbda0, 0xbd90, 0xbd80, 0xbd60, 0xbd40, 0xbd20, 0xbd00, 0xbce0, 0xbcc0, 0xbca0, 0xbc80, 0xbc60, 0xbc40, 0xbc20, 0xbc00, 0xbbe0, 0xbbc0, 0xbba0, 0xbb80, 0xbb40, 0xbb00, 0xbac0, 0xba80, 0xba40, 0xba00, 0xb9c0, 0xb980, 0xb900, 0xb880, 0xb800, 0xb780, 0xb680, 0xb580, 0xb380 };
    
    	for (int i = 0; i <= 255; i++) {
         myfloat var; 
         Bits bit_new;
         bit_new.ui = new_table[i] << 16 ;
         
         printf("posit = %d, float = %e -- %e\n", i,posit8tofp32(i) , bit_new.f );
	}
return 0;
}


