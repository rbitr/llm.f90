#include <stdlib.h>

// Function to allocate memory aligned to a 16-byte boundary
void* aligned_alloc_16(size_t size) {
    return aligned_alloc(16, size);
}

// Function to free the allocated memory
void aligned_free(void* ptr) {
    free(ptr);
}
