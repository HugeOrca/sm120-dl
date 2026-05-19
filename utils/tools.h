#pragma once
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>

using namespace cute;
#define PRINT64(x) if(thread(64, 0)) {print(#x"64 "); print(x); print("\n");   }
#define PRINT63(x) if(thread(63, 0)) {print(#x"63 "); print(x); print("\n");   }
#define PRINT(x) if(thread0()) {print(#x" "); print(x); print("\n");   }
#define PRINT_4(x) if(thread(128)) {print(#x" "); print(x); print("\n");   }
#define PRINT0_1(x) if(thread(0,1)) {print(#x"0_1 "); print(x); print("\n");   }
#define PRINT1_1(x) if(thread(1,1)) {print(#x"1_1 "); print(x); print("\n");   }
#define PRINT1(x) if(thread(1,0)) {print(#x"1 "); print(x); print("\n");   }
#define PRINT2(x) if(thread(2,0)) {print(#x"2 "); print(x); print("\n");   }
#define PRINT3(x) if(thread(3,0)) {print(#x"3 "); print(x); print("\n");   }
#define PRINT8(x) if(thread(8,0)) {print(#x"8 "); print(x); print("\n");   }
#define PRINT16(x) if(thread(16,0)) {print(#x"16 "); print(x); print("\n");   }
#define PRINT17(x) if(thread(17,0)) {print(#x"17 "); print(x); print("\n");   }
#define PRINT18(x) if(thread(18,0)) {print(#x"18 "); print(x); print("\n");   }
#define PRINT19(x) if(thread(19,0)) {print(#x"19 "); print(x); print("\n");   }
#define PRINT20(x) if(thread(20,0)) {print(#x"20 "); print(x); print("\n");   }
#define PRINT21(x) if(thread(21,0)) {print(#x"21 "); print(x); print("\n");   }
#define PRINT32(x) if(thread(32,0)) {print(#x"32 "); print(x); print("\n");   }
#define PRINT33(x) if(thread(33,0)) {print(#x"33 "); print(x); print("\n");   }
#define PTENSOR(x) if(thread0()) {print(#x" "); print_tensor(x); print("\n");   }
#define PTENSOR0_1(x) if(thread(0,1)) {print(#x"0_1 "); print_tensor(x); print("\n");   }
#define PTENSOR1_1(x) if(thread(1,1)) {print(#x"1_1 "); print_tensor(x); print("\n");   }
#define PTENSOR17(x) if(thread(17,0)) {print(#x"17 "); print_tensor(x); print("\n");   }
#define PTENSOR18(x) if(thread(18,0)) {print(#x"18 "); print_tensor(x); print("\n");   }
#define PTENSOR19(x) if(thread(19,0)) {print(#x"19 "); print_tensor(x); print("\n");   }
#define PTENSOR20(x) if(thread(20,0)) {print(#x"20 "); print_tensor(x); print("\n");   }
#define PTENSOR21(x) if(thread(21,0)) {print(#x"21 "); print_tensor(x); print("\n");   }
#define PTENSOR22(x) if(thread(22,0)) {print(#x"22 "); print_tensor(x); print("\n");   }
#define PTENSOR23(x) if(thread(23,0)) {print(#x"23 "); print_tensor(x); print("\n");   }
#define PTENSOR24(x) if(thread(24,0)) {print(#x"24 "); print_tensor(x); print("\n");   }
#define PTENSOR33(x) if(thread(33,0)) {print(#x"33 "); print_tensor(x); print("\n");   }
#define PTENSOR32(x) if(thread(32,0)) {print(#x"32 "); print_tensor(x); print("\n");   }
#define PTENSOR64(x) if(thread(64,0)) {print(#x"64 "); print_tensor(x); print("\n");   }
#define PTENSOR63(x) if(thread(63,0)) {print(#x"63 "); print_tensor(x); print("\n");   }
#define PLAYOUT(x) if(thread0()) {print(#x" "); print_layout(x); print("\n");   }
#define PLAYOUT_4(x) if(thread(0, 4)) {print(#x" "); print_layout(x); print("\n");   }
#define PLATEX(x) if(thread0()) {print(#x" "); print_latex(x); print("\n");   }


#define CEILDIV(x, align) (x+align-1)/align
#define ALIGNUP(x, align) (((x+align-1)/align)*align)
