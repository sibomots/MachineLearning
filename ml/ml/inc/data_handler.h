#ifndef _DATA_HANDLER_H__
#define _DATA_HANDLER_H__

#include <fstream>
#include <stdint.h>
#include "data.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

class data_handler
{
    std::vector<data*>* data_array;
    std::vector<data*>* training_data;
    std::vector<data*>* test_data;
    std::vector<data*>* validation_data;
    uint32_t num_classes;
    uint32_t feature_vector_size;
    std::map<uint8_t, int> class_map;

    const uint32_t TRAIN_SET_PERCENT = 75u;
    const uint32_t TEST_SET_PERCENT = 20u;
    const uint32_t VALIDATION_PERCENT = 5u;
public:
    data_handler();
    virtual ~data_handler();

    bool read_feature_vector(std::string path);
    bool read_feature_labels(std::string path);

    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(unsigned char* bytes) const;
    std::vector<data*>* get_training_data();
    std::vector<data*>* get_test_data();
    std::vector<data*>* get_validation_data();
};
#endif
