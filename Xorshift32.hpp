#ifndef XORSHIFT32_HPP
#define XORSHIFT32_HPP

#include <cstdint>

class Xorshift32 {
public:
    Xorshift32(uint32_t seed);
    inline uint32_t next();
    inline double next_double();
private:
    uint32_t state;
};

// Inline definitions (put these in the header to avoid needing a separate .cpp file)
inline Xorshift32::Xorshift32(uint32_t seed) : state(seed) {}

inline uint32_t Xorshift32::next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline double Xorshift32::next_double() {
    return static_cast<double>(next()) / static_cast<double>(UINT32_MAX);
}

#endif