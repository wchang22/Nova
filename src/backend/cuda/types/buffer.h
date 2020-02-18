#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

template <typename T>
class Buffer {
public:
  Buffer(size_t length, const T* data = nullptr);
  ~Buffer();

  T*& data() { return buffer; };

private:
  T* buffer;
};

#endif // CUDA_BUFFER_H