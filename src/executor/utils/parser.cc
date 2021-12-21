//
// Created by root on 12/22/21.
//

#include <falcon/utils/parser.h>

falcon::AlgorithmName parse_algorithm_name(const std::string& name) {
  falcon::AlgorithmName output = falcon::LOG_REG;
  if ("logistic_regression" == name) output = falcon::LOG_REG;
  if ("linear_regression" == name) output = falcon::LINEAR_REG;
  if ("decision_tree" == name) output = falcon::DT;
  if ("random_forest" == name) output = falcon::RF;
  if ("gbdt" == name) output = falcon::GBDT;
  if ("lime_compute_prediction" == name) output = falcon::LIME_COMP_PRED;
  if ("lime_compute_weights" == name) output = falcon::LIME_COMP_WEIGHT;
  if ("lime_feature_selection" == name) output = falcon::LIME_FEAT_SEL;
  if ("lime_interpret" == name) output = falcon::LIME_INTERPRET;
  return output;
}