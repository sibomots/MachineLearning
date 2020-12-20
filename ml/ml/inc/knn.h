#ifndef _KNN_H__
#define _KNN_H__

#include <vector>
#include "data.h"

class knn
{
   uint32_t k;
   std::vector<data*>* neighbors;
   std::vector<data*>* training_data;
   std::vector<data*>* test_data;
   std::vector<data*>* validation_data;
   static double MaxDouble;
   public:
      knn(uint32_t n = 1);
      ~knn();

      bool find_knearest(data* query_point);
      void set_training_data(std::vector<data*> *data);
      void set_test_data(std::vector<data*> *data);
      void set_validation_data(std::vector<data*> *data);
      void set_k(uint32_t n);
      uint32_t predict();
      bool calculate_distance(data* query_point, data* input, double& distance);
      double validate_performance();
      double test_performance();
};
#endif
