#ifndef _DATA_H__
#define _DATA_H__

#include <vector>
#include <stdio.h>
#include <stdint.h>

class data
{
    std::vector<uint8_t>* feature_vector;
    uint8_t label;
    uint32_t enum_label;
    double distance;
public:
    data();
    virtual ~data();

    void set_feature_vector(std::vector<uint8_t>* pvec);
    void append_to_feature_vector(uint8_t n);
    void set_label(uint8_t n);
    void set_enum_label(int n);
    void set_distance(double d);
    double get_distance();
    uint32_t get_feature_vector_size();
    uint8_t get_label();
    int get_enum_label();
    std::vector<uint8_t>* get_feature_vector();
};

#endif
