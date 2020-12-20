#include <data.h>
#include <prog.h>

data::data() {
  distance = 0.0;
  label = 0u;
  enum_label = 0u;
  feature_vector = new std::vector<uint8_t>;
}

data::~data() { SAFE_DELETE(feature_vector); }

double data::get_distance() { return distance; }

void data::set_distance(double d) { distance = d; }

void data::set_feature_vector(std::vector<uint8_t> *pvec) {
  feature_vector = pvec;
}

void data::append_to_feature_vector(uint8_t n) { feature_vector->push_back(n); }

void data::set_label(uint8_t n) { label = n; }

void data::set_enum_label(int n) { enum_label = n; }

uint32_t data::get_feature_vector_size() { return feature_vector->size(); }

uint8_t data::get_label() { return label; }

int data::get_enum_label() { return enum_label; }

std::vector<uint8_t> *data::get_feature_vector() { return feature_vector; }
