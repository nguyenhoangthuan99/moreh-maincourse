#include <cstdio>
#include <random>
#include <pthread.h>

class MyQueue {
public:
  MyQueue(int capacity);
  ~MyQueue();
  void Enqueue(float val);
  float Dequeue();
private:
  /* IMPLEMENT */
  int capacity_;
  int size_;
  int head_;
  float* buf_;
  pthread_mutex_t mutex_;
  pthread_cond_t not_empty_;
  pthread_cond_t not_full_;
};

MyQueue::MyQueue(int capacity) : capacity_(capacity), size_(0), head_(0) {
  /* IMPLEMENT */
}

MyQueue::~MyQueue() {
  /* IMPLEMENT */
}

void MyQueue::Enqueue(float val) {
  /* IMPLEMENT */
}

float MyQueue::Dequeue() {
  /* IMPLEMENT */
}

const int queue_size = 16;
const int elem_to_test = 1024;
const int err_threshold = 10;
MyQueue my_queue(queue_size);

void* producer_thread(void *data) {
  std::default_random_engine generator(42);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  for (int i = 0; i < elem_to_test; ++i) {
    printf("Enqueue %d\n", i);
    my_queue.Enqueue(distribution(generator));
  }
  return NULL;
}

void* consumer_thread(void *data) {
  std::default_random_engine generator(42);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  int cnt = 0;
  for (int i = 0; i < elem_to_test; ++i) {
    printf("Dequeue %d\n", i);
    float expected = distribution(generator);
    float dequeued = my_queue.Dequeue();
    if (expected != dequeued) {
      ++cnt;
      if (cnt <= err_threshold) {
        printf("Item %d: Expected %f, but got %f\n", i, expected, dequeued);
      }
      if (cnt == err_threshold + 1) {
        printf("Too many errors, only first %d values are printed.\n", err_threshold);
        exit(1);
      }
    }
  }
  printf("Result: VALID\n");
  return NULL;
}

int main() {
  pthread_t producer, consumer;
  pthread_create(&consumer, NULL, consumer_thread, NULL);
  pthread_create(&producer, NULL, producer_thread, NULL);
  pthread_join(consumer, NULL);
  pthread_join(producer, NULL);
  return 0;
}