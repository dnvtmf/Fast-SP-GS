#pragma once
#include "cuda.h"

namespace OPS_3D {
#define FUNC_DECL __device__ __host__ __forceinline__

template <typename scalar_t>
struct TypeSelecotr {
  using T  = float;
  using T2 = float2;
  using T3 = float3;
  using T4 = float4;
};

template <>
struct TypeSelecotr<float> {
  using T  = float;
  using T2 = float2;
  using T3 = float3;
  using T4 = float4;
};

template <>
struct TypeSelecotr<double> {
  using T  = double;
  using T2 = double2;
  using T3 = double3;
  using T4 = double4;
};

template <>
struct TypeSelecotr<int32_t> {
  using T  = int;
  using T2 = int2;
  using T3 = int3;
  using T4 = int4;
};

template <>
struct TypeSelecotr<int64_t> {
  using T  = int64_t;
  using T2 = longlong2;
  using T3 = longlong3;
  using T4 = longlong4;
};

template <typename T = float>
struct vec3 : public TypeSelecotr<T>::T3 {
  FUNC_DECL vec3() { this->x = 0, this->y = 0, this->z = 0; }
  FUNC_DECL vec3(T x) { this->x = x, this->y = x, this->z = x; }
  FUNC_DECL vec3(T x, T y, T z) { this->x = x, this->y = y, this->z = z; }
  FUNC_DECL vec3(const vec3<T> &b) { this->x = b.x, this->y = b.y, this->z = b.z; }
  FUNC_DECL vec3(const T *ptr) { this->x = ptr[0], this->y = ptr[1], this->z = ptr[2]; }

  FUNC_DECL T &operator[](const int i) { return i == 0 ? this->x : (i == 1 ? this->y : this->z); }

  FUNC_DECL const vec3<T> operator+(const T &b) const { return {this->x + b, this->y + b, this->z + b}; }
  FUNC_DECL friend const vec3<T> operator+(const T &a, const vec3<T> &b) { return {a + b.x, a + b.y, a + b.z}; }
  FUNC_DECL const vec3<T> operator+(const vec3<T> &b) const { return {this->x + b.x, this->y + b.y, this->z + b.z}; }

  FUNC_DECL void operator+=(const vec3<T> &b) { this->x += b.x, this->y += b.y, this->z += b.z; }
  FUNC_DECL void operator+=(const T &b) { this->x += b, this->y += b, this->z += b; }

  FUNC_DECL const vec3<T> operator-(const T &b) const { return {this->x - b, this->y - b, this->z - b}; }
  FUNC_DECL friend const vec3<T> operator-(const T &a, const vec3<T> &b) { return {a - b.x, a - b.y, a - b.z}; }
  FUNC_DECL const vec3<T> operator-(const vec3<T> &b) const { return {this->x - b.x, this->y - b.y, this->z - b.z}; }

  FUNC_DECL void operator-=(const vec3<T> &b) { this->x -= b.x, this->y -= b.y, this->z -= b.z; }
  FUNC_DECL void operator-=(const T &b) { this->x -= b, this->y -= b, this->z -= b; }

  FUNC_DECL const vec3<T> operator*(const T &b) const { return {this->x * b, this->y * b, this->z * b}; }
  FUNC_DECL friend const vec3<T> operator*(const T &a, const vec3<T> &b) { return {a * b.x, a * b.y, a * b.z}; }
  FUNC_DECL const vec3<T> operator*(const vec3<T> &b) const { return {this->x * b.x, this->y * b.y, this->z * b.z}; }

  FUNC_DECL void operator*=(const vec3<T> &b) { this->x *= b.x, this->y *= b.y, this->z *= b.z; }
  FUNC_DECL void operator*=(const T &b) { this->x *= b, this->y *= b, this->z *= b; }

  FUNC_DECL const vec3<T> operator/(const T &b) const { return {this->x / b, this->y / b, this->z / b}; }
  FUNC_DECL friend const vec3<T> operator/(const T &a, const vec3<T> &b) { return {a / b.x, a / b.y, a / b.z}; }
  FUNC_DECL const vec3<T> operator/(const vec3<T> &b) const { return {this->x / b.x, this->y / b.y, this->z / b.z}; }

  FUNC_DECL void operator/=(const vec3<T> &b) { this->x /= b.x, this->y /= b.y, this->z /= b.z; }
  FUNC_DECL void operator/=(const T &b) { this->x /= b, this->y /= b, this->z /= b; }

  FUNC_DECL T dot(const vec3<T> &b) const { return this->x * b.x + this->y * b.y + this->z * b.z; }
  FUNC_DECL T operator^(const vec3<T> &b) const { return this->x * b.x + this->y * b.y + this->z * b.z; }

  FUNC_DECL vec3<T> cross(const vec3<T> &b) const {
    return {this->y * b.z - this->z * b.y, this->z * b.x - this->x * b.z, this->x * b.y - this->y * b.x};
  }
};

template <typename T = float>
struct vec4 : public TypeSelecotr<T>::T4 {
  FUNC_DECL vec4() { this->x = this->y = this->z = 0; }
  FUNC_DECL vec4(T x) { this->x = this->y = this->z = this->w = x; }
  FUNC_DECL vec4(T x, T y, T z, T w) { this->x = x, this->y = y, this->z = z, this->w = w; }
  FUNC_DECL vec4(const vec4<T> &b) { this->x = b.x, this->y = b.y, this->z = b.z, this->w = b.w; }
  FUNC_DECL vec4(const vec3<T> &b, T w = 0) { this->x = b.x, this->y = b.y, this->z = b.z, this->w = w; }
  FUNC_DECL vec4(const T *ptr) { this->x = ptr[0], this->y = ptr[1], this->z = ptr[2], this->w = ptr[3]; }

  FUNC_DECL T &operator[](const int i) { return i == 0 ? this->x : (i == 1 ? this->y : (i == 3 ? this->z : this->w)); }

  FUNC_DECL const vec4<T> operator+(const T &b) const { return {this->x + b, this->y + b, this->z + b, this->w + b}; }
  FUNC_DECL friend const vec4<T> operator+(const T &a, const vec4<T> &b) {
    return {a + b.x, a + b.y, a + b.z, a + b.w};
  }
  FUNC_DECL const vec4<T> operator+(const vec4<T> &b) const {
    return {this->x + b.x, this->y + b.y, this->z + b.z, this->w + b.w};
  }

  FUNC_DECL void operator+=(const vec4<T> &b) { this->x += b.x, this->y += b.y, this->z += b.z, this->w += b.w; }
  FUNC_DECL void operator+=(const T &b) { this->x += b, this->y += b, this->z += b, this->w += b; }

  FUNC_DECL const vec4<T> operator-(const T &b) const { return {this->x - b, this->y - b, this->z - b, this->w - b}; }
  FUNC_DECL friend const vec4<T> operator-(const T &a, const vec4<T> &b) {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
  }
  FUNC_DECL const vec4<T> operator-(const vec4<T> &b) const {
    return {this->x - b.x, this->y - b.y, this->z - b.z, this->w - b.w};
  }

  FUNC_DECL void operator-=(const vec4<T> &b) { this->x -= b.x, this->y -= b.y, this->z -= b.z, this->w -= b.w; }
  FUNC_DECL void operator-=(const T &b) { this->x -= b, this->y -= b, this->z -= b, this->w -= b; }

  FUNC_DECL const vec4<T> operator*(const T &b) const { return {this->x * b, this->y * b, this->z * b, this->w * b}; }
  FUNC_DECL friend const vec4<T> operator*(const T &a, const vec4<T> &b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
  }
  FUNC_DECL const vec4<T> operator*(const vec4<T> &b) const {
    return {this->x * b.x, this->y * b.y, this->z * b.z, this->w * b.w};
  }

  FUNC_DECL void operator*=(const vec4<T> &b) { this->x *= b.x, this->y *= b.y, this->z *= b.z, this->w *= b.w; }
  FUNC_DECL void operator*=(const T &b) { this->x *= b, this->y *= b, this->z *= b, this->w *= b; }

  FUNC_DECL const vec4<T> operator/(const T &b) const { return {this->x / b, this->y / b, this->z / b, this->w / b}; }
  FUNC_DECL friend const vec4<T> operator/(const T &a, const vec4<T> &b) {
    return {a / b.x, a / b.y, a / b.z, a / b.w};
  }
  FUNC_DECL const vec4<T> operator/(const vec4<T> &b) const {
    return {this->x / b.x, this->y / b.y, this->z / b.z, this->w / b.w};
  }

  FUNC_DECL void operator/=(const vec4<T> &b) { this->x /= b.x, this->y /= b.y, this->z /= b.z, this->w /= b.w; }
  FUNC_DECL void operator/=(const T &b) { this->x /= b, this->y /= b, this->z /= b, this->w /= b; }

  FUNC_DECL T dot(const vec4<T> &b) const { return this->x * b.x + this->y * b.y + this->z * b.z + this->w * b.w; }
  FUNC_DECL T operator^(const vec4<T> &b) const {
    return this->x * b.x + this->y * b.y + this->z * b.z + this->w * b.w;
  }
  FUNC_DECL T dot(const vec3<T> &b) const { return this->x * b.x + this->y * b.y + this->z * b.z + this->w; }
  FUNC_DECL T operator^(const vec3<T> &b) const { return this->x * b.x + this->y * b.y + this->z * b.z + this->w; }
  FUNC_DECL friend T dot(const vec3<T> &a, const vec4<T> &b) { return a.x * b.x + a.y * b.y + a.z * b.z + b.w; }
  FUNC_DECL friend T operator^(const vec3<T> &a, const vec4<T> &b) { return a.x * b.x + a.y * b.y + a.z * b.z + b.w; }
};

template <typename T>
struct mat3 {
  union {
    struct {
      vec3<T> value[3];
    };
    T _data[9];
  };
  FUNC_DECL mat3() {
    for (int i = 0; i < 9; ++i) _data[i] = 0;
  }
  FUNC_DECL mat3(const mat3<T> &b) {
    for (int i = 0; i < 9; ++i) _data[i] = b._data[i];
  }
  FUNC_DECL mat3(T *data) {
    for (int i = 0; i < 9; ++i) _data[i] = data[i];
  }
  FUNC_DECL mat3(T x1, T y1, T z1, T x2, T y2, T z2, T x3, T y3, T z3) { _data = {x1, y1, z1, x2, y2, z2, x3, y3, z3}; }
  FUNC_DECL vec3<T> &operator[](const int i) { return value[i]; }
  FUNC_DECL T &operator()(int row, int col) { return _data[row * 3 + col]; }
  FUNC_DECL T &at(int row, int col) { return _data[row * 4 + col]; }
  FUNC_DECL mat3<T> static I() { return {T(1), 0, 0, 0, T(1), 0, 0, 0, T(1)}; }

  FUNC_DECL vec3<T> mul(const vec3<T> &p) {
    return {
        p.x * _data[0] + p.y * _data[1] + p.z * _data[2],
        p.x * _data[3] + p.y * _data[4] + p.z * _data[5],
        p.x * _data[6] + p.y * _data[7] + p.z * _data[8],
    };
  }
  FUNC_DECL vec3<T> operator*(const vec3<T> &p) { return mul(p); }
};

template <typename T>
struct mat4 {
  union {
    struct {
      vec4<T> value[4];
    };
    T _data[16];
  };

  FUNC_DECL mat4() {
    for (int i = 0; i < 16; ++i) _data[i] = 0;
  }
  FUNC_DECL mat4(T *data) {
    for (int i = 0; i < 16; ++i) _data[i] = data[i];
  }
  FUNC_DECL mat4(T x1, T y1, T z1, T w1, T x2, T y2, T z2, T w2, T x3, T y3, T z3, T w3, T x4, T y4, T z4, T w4) {
    value[0] = {x1, y1, z1, w1};
    value[1] = {x2, y2, z2, w2};
    value[2] = {x3, y3, z3, w3};
    value[3] = {x4, y4, z4, w4};
  }
  FUNC_DECL mat4(const mat3<T> &R, const vec3<T> &t) {
    value[0] = {R._data[0], R._data[1], R._data[2], t.x};
    value[1] = {R._data[3], R._data[4], R._data[5], t.y};
    value[2] = {R._data[6], R._data[7], R._data[8], t.z};
    value[3] = {0, 0, 0, T(1)};
  }
  FUNC_DECL mat4(const mat4<T> &M) {
    for (int i = 0; i < 16; ++i) this->_data[i] = M._data[i];
  }

  FUNC_DECL vec4<T> &operator[](const int &i) { return value[i]; }
  FUNC_DECL T &operator()(const int &row, const int &col) { return _data[row * 4 + col]; }
  FUNC_DECL vec4<T> &operator()(int row) { return {_data[row + 0], _data[row + 1], _data[row + 2], _data[row + 3]}; }
  FUNC_DECL T &at(int row, int col) { return _data[row * 4 + col]; }

  FUNC_DECL mat4<T> static I() { return {T(1), 0, 0, 0, 0, T(1), 0, 0, 0, 0, T(1), 0, 0, 0, 0, T(1)}; }

  FUNC_DECL mat4<T> operator*(const mat4<T> &B) const {
    mat4<T> C;
    C._data[0 * 4 + 0] = _data[0 * 4 + 0] * B._data[0 * 4 + 0] + _data[0 * 4 + 1] * B._data[1 * 4 + 0] +
                         _data[0 * 4 + 2] * B._data[2 * 4 + 0] + _data[0 * 4 + 3] * B._data[3 * 4 + 0];
    C._data[0 * 4 + 1] = _data[0 * 4 + 0] * B._data[0 * 4 + 1] + _data[0 * 4 + 1] * B._data[1 * 4 + 1] +
                         _data[0 * 4 + 2] * B._data[2 * 4 + 1] + _data[0 * 4 + 3] * B._data[3 * 4 + 1];
    C._data[0 * 4 + 2] = _data[0 * 4 + 0] * B._data[0 * 4 + 2] + _data[0 * 4 + 1] * B._data[1 * 4 + 2] +
                         _data[0 * 4 + 2] * B._data[2 * 4 + 2] + _data[0 * 4 + 3] * B._data[3 * 4 + 2];
    C._data[0 * 4 + 3] = _data[0 * 4 + 0] * B._data[0 * 4 + 3] + _data[0 * 4 + 1] * B._data[1 * 4 + 3] +
                         _data[0 * 4 + 2] * B._data[2 * 4 + 3] + _data[0 * 4 + 3] * B._data[3 * 4 + 3];

    C._data[1 * 4 + 0] = _data[1 * 4 + 0] * B._data[0 * 4 + 0] + _data[1 * 4 + 1] * B._data[1 * 4 + 0] +
                         _data[1 * 4 + 2] * B._data[2 * 4 + 0] + _data[1 * 4 + 3] * B._data[3 * 4 + 0];
    C._data[1 * 4 + 1] = _data[1 * 4 + 0] * B._data[0 * 4 + 1] + _data[1 * 4 + 1] * B._data[1 * 4 + 1] +
                         _data[1 * 4 + 2] * B._data[2 * 4 + 1] + _data[1 * 4 + 3] * B._data[3 * 4 + 1];
    C._data[1 * 4 + 2] = _data[1 * 4 + 0] * B._data[0 * 4 + 2] + _data[1 * 4 + 1] * B._data[1 * 4 + 2] +
                         _data[1 * 4 + 2] * B._data[2 * 4 + 2] + _data[1 * 4 + 3] * B._data[3 * 4 + 2];
    C._data[1 * 4 + 3] = _data[1 * 4 + 0] * B._data[0 * 4 + 3] + _data[1 * 4 + 1] * B._data[1 * 4 + 3] +
                         _data[1 * 4 + 2] * B._data[2 * 4 + 3] + _data[1 * 4 + 3] * B._data[3 * 4 + 3];

    C._data[2 * 4 + 0] = _data[2 * 4 + 0] * B._data[0 * 4 + 0] + _data[2 * 4 + 1] * B._data[1 * 4 + 0] +
                         _data[2 * 4 + 2] * B._data[2 * 4 + 0] + _data[2 * 4 + 3] * B._data[3 * 4 + 0];
    C._data[2 * 4 + 1] = _data[2 * 4 + 0] * B._data[0 * 4 + 1] + _data[2 * 4 + 1] * B._data[1 * 4 + 1] +
                         _data[2 * 4 + 2] * B._data[2 * 4 + 1] + _data[2 * 4 + 3] * B._data[3 * 4 + 1];
    C._data[2 * 4 + 2] = _data[2 * 4 + 0] * B._data[0 * 4 + 2] + _data[2 * 4 + 1] * B._data[1 * 4 + 2] +
                         _data[2 * 4 + 2] * B._data[2 * 4 + 2] + _data[2 * 4 + 3] * B._data[3 * 4 + 2];
    C._data[2 * 4 + 3] = _data[2 * 4 + 0] * B._data[0 * 4 + 3] + _data[2 * 4 + 1] * B._data[1 * 4 + 3] +
                         _data[2 * 4 + 2] * B._data[2 * 4 + 3] + _data[2 * 4 + 3] * B._data[3 * 4 + 3];

    C._data[3 * 4 + 0] = _data[3 * 4 + 0] * B._data[0 * 4 + 0] + _data[3 * 4 + 1] * B._data[1 * 4 + 0] +
                         _data[3 * 4 + 2] * B._data[2 * 4 + 0] + _data[3 * 4 + 3] * B._data[3 * 4 + 0];
    C._data[3 * 4 + 1] = _data[3 * 4 + 0] * B._data[0 * 4 + 1] + _data[3 * 4 + 1] * B._data[1 * 4 + 1] +
                         _data[3 * 4 + 2] * B._data[2 * 4 + 1] + _data[3 * 4 + 3] * B._data[3 * 4 + 1];
    C._data[3 * 4 + 2] = _data[3 * 4 + 0] * B._data[0 * 4 + 2] + _data[3 * 4 + 1] * B._data[1 * 4 + 2] +
                         _data[3 * 4 + 2] * B._data[2 * 4 + 2] + _data[3 * 4 + 3] * B._data[3 * 4 + 2];
    C._data[3 * 4 + 3] = _data[3 * 4 + 0] * B._data[0 * 4 + 3] + _data[3 * 4 + 1] * B._data[1 * 4 + 3] +
                         _data[3 * 4 + 2] * B._data[2 * 4 + 3] + _data[3 * 4 + 3] * B._data[3 * 4 + 3];

    return C;
  }
};

}  // namespace OPS_3D