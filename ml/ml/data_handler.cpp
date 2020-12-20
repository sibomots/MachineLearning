#include <data_handler.h>
#include <iostream>
#include <prog.h>

data_handler::data_handler() {
  data_array = new std::vector<data *>;
  test_data = new std::vector<data *>;
  training_data = new std::vector<data *>;
  validation_data = new std::vector<data *>;
  num_classes = 0u;
  feature_vector_size = 0u;
}

data_handler::~data_handler() {
  SAFE_DELETE(data_array);
  SAFE_DELETE(test_data);
  SAFE_DELETE(training_data);
  SAFE_DELETE(validation_data);
}

bool data_handler::read_feature_vector(std::string path) {
  uint32_t header[4] = {0};
  unsigned char bytes[4] = {0};

  FILE *fp = fopen(path.c_str(), "rb");
  if (fp == NULL) {
    printf("ERROR: Could not find the file: %s\n", path.c_str());
    return false;
  }

  for (uint32_t n = 0u; n < 4u; n++) {
    if (fread(bytes, sizeof(bytes), 1u, fp)) {
      header[n] = convert_to_little_endian(bytes);
    }
  }

  uint32_t image_size = header[2] * header[3];
#ifdef _DEBUG
  printf("INFO: Done getting file header\n");
  printf("INFO: Image_size = %d\n", image_size);
#endif
  for (uint32_t i = 0u; i < header[1]; i++) {
    data *pData = new data;
    // we can read faster by reading image_size bytes at a time
    uint8_t *pElements = new uint8_t[image_size];
    if (size_t r = fread(pElements, sizeof(uint8_t), image_size, fp)) {
      // now that we read that set, append it.
      for (uint32_t j = 0; j < image_size; j++) {
        pData->append_to_feature_vector(pElements[j]);
      }
    } else {
      printf("ERROR: Unsuccessful trying to read data.\n");
      SAFE_DELETE(pData);
      SAFE_DELETEA(pElements);
      return false;
    }
    data_array->push_back(pData);
    SAFE_DELETEA(pElements);
  }
#ifdef _DEBUG
  printf("INFO: Size of data_array %lu\n", data_array->size());
  printf("INFO: Successfully read/stored feature vectors.\n");
#endif
  return true;
}

bool data_handler::read_feature_labels(std::string path) {
  uint32_t header[4] = {0};
  unsigned char bytes[4] = {0};

  FILE *fp = fopen(path.c_str(), "rb");
  if (fp == NULL) {
    printf("ERROR: Could not find the file %s\n", path.c_str());
    return false;
  }

  for (uint32_t n = 0; n < 2; n++) {
    if (fread(bytes, sizeof(bytes), 1, fp)) {
      header[n] = convert_to_little_endian(bytes);
    }
  }
#ifdef _DEBUG
  printf("INFO: Done getting label file header\n");
#endif
  uint8_t *pElements = new uint8_t[header[1]];
  if (fread(pElements, sizeof(uint8_t), header[1], fp)) {
    for (uint32_t i = 0; i < header[1]; i++) {
      data_array->at(i)->set_label(pElements[i]);
    }
  } else {
    printf("ERROR: error reading from file.\n");
    SAFE_DELETEA(pElements);
    return false;
  }
  SAFE_DELETEA(pElements);
#ifdef _DEBUG
  printf("INFO: Successfully read/stored label.\n");
#endif
  return true;
}

void data_handler::split_data() {
  std::unordered_set<int> used_indexes;
  uint32_t train_size = (data_array->size() * TRAIN_SET_PERCENT) / 100;
  uint32_t test_size = (data_array->size() * TEST_SET_PERCENT) / 100;
  uint32_t valid_size = (data_array->size() * VALIDATION_PERCENT) / 100;
  uint32_t count = 0;
  // seed the rand
  srand((unsigned int)time(NULL));

  while (count < train_size) {
    uint32_t rand_index = rand() % data_array->size();
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      training_data->push_back(data_array->at(rand_index));
      used_indexes.insert(rand_index);
      count++;
    }
  }
  count = 0u;
  while (count < test_size) {
    uint32_t rand_index = rand() % data_array->size();
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      test_data->push_back(data_array->at(rand_index));
      used_indexes.insert(rand_index);
      count++;
    }
  }

  count = 0;
  while (count < valid_size) {
    uint32_t rand_index = rand() % data_array->size();
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      validation_data->push_back(data_array->at(rand_index));
      used_indexes.insert(rand_index);
      count++;
    }
  }

#ifdef _DEBUG
  printf("INFO: Training data size: %lu\n", training_data->size());
  printf("INFO: Test data size: %lu\n", test_data->size());
  printf("INFO: Validation data size: %lu\n", validation_data->size());
#endif
}

void data_handler::count_classes() {
  uint32_t count = 0;
  for (uint32_t i = 0; i < data_array->size(); i++) {
    if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
      class_map[data_array->at(i)->get_label()] = count;
      data_array->at(i)->set_enum_label(count);
      count++;
    }
  }
  num_classes = count;
#ifdef _DEBUG
  printf("INFO: Successfully extracted %d unique classes\n", num_classes);
#endif
}

inline uint32_t
data_handler::convert_to_little_endian(unsigned char *bytes) const {
  uint32_t result = (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) |
                               (bytes[2] << 8) | (bytes[3]));
  return result;
}

std::vector<data *> *data_handler::get_training_data() { return training_data; }

std::vector<data *> *data_handler::get_test_data() { return test_data; }

std::vector<data *> *data_handler::get_validation_data() {
  return validation_data;
}
