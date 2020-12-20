#include <knn.h>
#include <prog.h>

#include <cmath>
#include <data_handler.h>
#include <limits>
#include <map>
#include <stdint.h>

double knn::MaxDouble = std::numeric_limits<double>::max();

knn::knn(uint32_t n) { k = n; }

knn::~knn() { SAFE_DELETE(neighbors); }

bool knn::find_knearest(data *query_point) {
  double min = knn::MaxDouble;
  double prev_min = min;
  uint32_t index = 0u;
  neighbors = new std::vector<data *>;

  for (uint32_t i = 0u; i < k; i++) {
    if (i == 0) {
      for (size_t j = 0; j < training_data->size(); j++) {
        double distance = 0.0;
        // we can fail here, so let's check:
        if (calculate_distance(query_point, training_data->at(j), distance)) {
          training_data->at(j)->set_distance(distance);
          if (distance < min) {
            min = distance;
            index = j;
          }
        } else {
          return false;
        }
      }
      neighbors->push_back(training_data->at(index));
      prev_min = min;
      min = knn::MaxDouble;
    } else {
      for (uint32_t j = 0u; j < training_data->size(); j++) {
        double distance = 0.0;
        if (calculate_distance(query_point, training_data->at(j), distance)) {
          training_data->at(j)->get_distance();
          if (distance > prev_min && distance < min) {
            min = distance;
            index = j;
          }
        } else {
          return false;
        }
      }
      neighbors->push_back(training_data->at(index));
      prev_min = min;
      min = knn::MaxDouble;
    }
  }
  return true;
}

void knn::set_training_data(std::vector<data *> *data) { training_data = data; }

void knn::set_test_data(std::vector<data *> *data) { test_data = data; }

void knn::set_validation_data(std::vector<data *> *data) {
  validation_data = data;
}

void knn::set_k(uint32_t n) { k = n; }

uint32_t knn::predict() {
  std::map<uint8_t, uint32_t> class_freq;
  for (size_t i = 0; i < neighbors->size(); i++) {
    if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end()) {
      class_freq[neighbors->at(i)->get_label()] = 1;
    } else {
      class_freq[neighbors->at(i)->get_label()]++;
    }
  }

  uint32_t best = 0;
  uint32_t max = 0;
  for (auto kv : class_freq) {

    if (kv.second > max) {
      max = kv.second;
      best = kv.first;
    }
  }
  SAFE_DELETE(neighbors);
  return best;
}

bool knn::calculate_distance(data *query_point, data *input, double &distance) {
  uint32_t qp_size = query_point->get_feature_vector_size();
  uint32_t in_size = input->get_feature_vector_size();

  if (qp_size != in_size) {
    printf("ERROR: query_point feature vector size (%d) != input feature "
           "vector size (%d)\n",
           qp_size, in_size);
    return false;
  }

  double d = 0.0;
  for (uint32_t i = 0u; i < query_point->get_feature_vector_size(); i++) {
    double x = 0.0;
    x = query_point->get_feature_vector()->at(i);
    x -= input->get_feature_vector()->at(i);
    d += (x * x);
  }
  double result = sqrt(d);
  distance = result;
  return true;
}

double knn::validate_performance() {
  double current_performance = 0.0;
  uint32_t count = 0;
  uint32_t data_index = 0;
  for (data *query_point : *validation_data) {
    bool bfound = find_knearest(query_point);
    if (bfound) {
      uint32_t prediction = predict();
      uint32_t label = query_point->get_label();
      if (prediction == query_point->get_label()) {
        count++;
      }
      data_index++;
#ifdef _DEBUG
      printf("INFO: Current Performance  %c(P: %d  L: %d) = %.3f %%\n",
             ((prediction != label) ? '*' : ' '), prediction, label,
             ((double)(count * 100)) / ((double)(data_index)));
#endif
    } else {
      printf("Failed to find?\n");
    }
  }
  current_performance =
      ((double)(count * 100)) / ((double)(validation_data->size()));
#ifdef _DEBUG
  printf("INFO: Validation Performance for K = %d:  %.3f %%\n", k,
         current_performance);
#endif
  return current_performance;
}

double knn::test_performance() {
  double current_performance = 0.0;
  uint32_t count = 0u;
  for (data *query_point : *test_data) {
    find_knearest(query_point);
    uint32_t prediction = predict();
    if (prediction == query_point->get_label()) {
      count++;
    }
  }
  current_performance = ((double)(count * 100)) / ((double)(test_data->size()));
#ifdef _DEBUG
  printf("Tested Performance = %.3f %%\n", current_performance);
#endif
  return current_performance;
}
