//
// Created by nailixing on 03/07/21.
//

#include "falcon/distributed/parameter_server_base.h"
#include <falcon/algorithm/vertical/linear_model/logistic_regression_ps.h>
#include <falcon/utils/pb_converter/common_converter.h>
#include <falcon/utils/pb_converter/alg_params_converter.h>
#include <falcon/utils/pb_converter/phe_keys_converter.h>
#include <falcon/algorithm/vertical/linear_model/logistic_regression_builder.h>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <random>

using namespace std;


LRParameterServer::LRParameterServer(const LRParameterServer &obj):alg_builder(obj.alg_builder), party(obj.party){}

LRParameterServer::LRParameterServer(
        const LogisticRegressionBuilder& m_alg_builder,
        const Party& m_party,
        const std::string& ps_network_config_pb_str):
        ParameterServer(ps_network_config_pb_str),
        alg_builder(m_alg_builder), party(m_party){}

LRParameterServer::LRParameterServer(
        const Party& m_party,
        const string &ps_network_config_pb_str):
        ParameterServer(ps_network_config_pb_str),
        party(m_party) {}


LRParameterServer::~LRParameterServer() = default;

void LRParameterServer::broadcast_train_test_data(const std::vector<std::vector<double>> &training_data,
                                                  const std::vector<std::vector<double>> &testing_data,
                                                  const std::vector<double> &training_labels,
                                                  const std::vector<double> &testing_labels) {
  // every ps sends the training data and testing data to workers
  std::string training_data_str, testing_data_str, training_labels_str, testing_labels_str;
  serialize_double_matrix(training_data, training_data_str);
  serialize_double_matrix(testing_data, testing_data_str);
  for (int wk_index = 0; wk_index < this->worker_channels.size(); wk_index++) {
    this->send_long_message_to_worker(wk_index, training_data_str);
    this->send_long_message_to_worker(wk_index, testing_data_str);
  }
  // only active ps sends the training labels and testing labels to workers
  if (party.party_type == falcon::ACTIVE_PARTY) {
    serialize_double_array(training_labels, training_labels_str);
    serialize_double_array(testing_labels, testing_labels_str);
    for (int wk_index = 0; wk_index < this->worker_channels.size(); wk_index++) {
      this->send_long_message_to_worker(wk_index, training_labels_str);
      this->send_long_message_to_worker(wk_index, testing_labels_str);
    }
  }
}

void LRParameterServer::broadcast_phe_keys() {
  std::string phe_keys_str = party.export_phe_key_string();
  for (int wk_index = 0; wk_index < this->worker_channels.size(); wk_index++) {
    this->send_long_message_to_worker(wk_index, phe_keys_str);
  }
}

void LRParameterServer::broadcast_encrypted_weights(){
  std::string weight_str;
  serialize_encoded_number_array(
      this->alg_builder.log_reg_model.local_weights,
      this->alg_builder.log_reg_model.weight_size,
      weight_str);

  for(int wk_index=0; wk_index< this->worker_channels.size(); wk_index++){
    this->send_long_message_to_worker(wk_index, weight_str);
  }
}

std::vector<int> LRParameterServer::select_batch_idx(std::vector<int> training_data_indexes) const{
  // push to batch_indexes
  std::vector<int> batch_indexes;
  if (party.party_type == falcon::ACTIVE_PARTY) {
    //randomly select batch indexes and send to other parties
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::shuffle(std::begin(training_data_indexes), std::end(training_data_indexes), rng);

    batch_indexes.reserve(this->alg_builder.batch_size);
    for (int i = 0; i < this->alg_builder.batch_size; i++) {
      batch_indexes.push_back(training_data_indexes[i]);
    }

    // broadcast to other passive ps
    std::string batch_indexes_str;
    serialize_int_array(batch_indexes, batch_indexes_str);
    for (int ps_index = 0; ps_index < party.party_num; ps_index++) {
      if (ps_index != party.party_id) {
        party.send_long_message(ps_index, batch_indexes_str);
      }
    }
  } else {
    std::string recv_batch_indexes_str;
    party.recv_long_message(ACTIVE_PARTY_ID, recv_batch_indexes_str);
    deserialize_int_array(batch_indexes, recv_batch_indexes_str);
  }

  return batch_indexes;
}

std::vector<int> LRParameterServer::partition_examples(std::vector<int> batch_indexes){

  int mini_batch_size = int(batch_indexes.size()/this->worker_channels.size());

  std::cout << "ps worker size = " << this->worker_channels.size() << std::endl;
  std::cout << "mini batch size = " << mini_batch_size << std::endl;

  std::vector<int> message_sizes;

  // deterministic partition given the batch indexes
  int index = 0;
  for(int wk_index = 0; wk_index < this->worker_channels.size(); wk_index++){

    // generate mini-batch for this worker
    std::vector<int>::const_iterator first1 = batch_indexes.begin() + index;
    std::vector<int>::const_iterator last1  = batch_indexes.begin() + index + mini_batch_size;

    if (wk_index == this->worker_channels.size() - 1){
      last1  = batch_indexes.end();
    }
    std::vector<int> mini_batch_indexes(first1, last1);

    // serialize mini_batch_indexes to str
    std::string mini_batch_indexes_str;
    serialize_int_array(mini_batch_indexes, mini_batch_indexes_str);

    // record size of mini_batch_indexes, used in deserialization process
    message_sizes.push_back((int) mini_batch_indexes.size());

    // send to worker
    this->send_long_message_to_worker(wk_index, mini_batch_indexes_str);
    // update index
    index += mini_batch_size;
  }

  for (int i = 0; i < message_sizes.size(); i++) {
    std::cout << "message_sizes[" << i << "] = " << message_sizes[i] << std::endl;
    LOG(INFO) << "message_sizes[" << i << "] = " << message_sizes[i];
  }
  std::cout << "Broadcast client's batch requests to other workers" << std::endl;
  LOG(INFO) << "Broadcast client's batch requests to other workers";

  return message_sizes;

}

std::vector<string> LRParameterServer::wait_worker_complete(){

  std::vector<string> encoded_messages;
  // wait until all received str has been stored to encrypted_gradient_strs
  while (this->worker_channels.size() != encoded_messages.size()){
    for(int wk_index=0; wk_index< this->worker_channels.size(); wk_index++){
      std::string message;
      this->recv_long_message_from_worker(wk_index, message);
      encoded_messages.push_back(message);
    }
  }
  return encoded_messages;
}

EncodedNumber* LRParameterServer::update_encrypted_weights(
  const std::vector< string >& encoded_messages) const{

  djcs_t_public_key* phe_pub_key = djcs_t_init_public_key();
  party.getter_phe_pub_key(phe_pub_key);

  int weight_size = this->alg_builder.log_reg_model.weight_size;
  int weight_phe_precision = abs(this->alg_builder.log_reg_model.local_weights[0].getter_exponent());
  auto encrypted_aggregated_gradients = new EncodedNumber[weight_size];
  // first, need to initialize and encrypt the gradients
  for (int i = 0; i < weight_size; i++) {
    encrypted_aggregated_gradients[i].set_double(phe_pub_key->n[0], 0.0, weight_phe_precision);
    djcs_t_aux_encrypt(phe_pub_key, party.phe_random,
                       encrypted_aggregated_gradients[i],
                       encrypted_aggregated_gradients[i]);
  }

  // deserialize encrypted message, and add to encrypted
  for (const std::string& message: encoded_messages){
    auto tmp = new EncodedNumber[this->alg_builder.log_reg_model.weight_size];
    deserialize_encoded_number_array(
        tmp,
        weight_size,
        message);
    for (auto i=0; i < weight_size; i++) {
      djcs_t_aux_ee_add(phe_pub_key,
                        encrypted_aggregated_gradients[i],
                        encrypted_aggregated_gradients[i],
                        tmp[i]);
    }
    delete [] tmp;
  }

  // after aggregating the gradients, update the local weights
  if (!this->alg_builder.with_regularization) {
    // update each local weight j
    for (int j = 0; j < weight_size; j++) {
      // update the j-th weight in local_weight vector
      // need to make sure that the exponents of inner_product
      // and local weights are the same
      djcs_t_aux_ee_add(
          phe_pub_key,
          this->alg_builder.log_reg_model.local_weights[j],
          this->alg_builder.log_reg_model.local_weights[j],
          encrypted_aggregated_gradients[j]);
    }
  } else {
    // TODO: handle the update formula when using regularization,
    // currently only l2
  }

  djcs_t_free_public_key(phe_pub_key);
  delete [] encrypted_aggregated_gradients;
}

void LRParameterServer::distributed_train(){

  // step 1: init encrypted local weights
  // (here use 3 * precision for consistence in the following)
  int encrypted_weights_precision = 3 * PHE_FIXED_POINT_PRECISION;
  int plaintext_samples_precision = PHE_FIXED_POINT_PRECISION;

  // create encrypted weight and store it to alg_builder 's local_weight
  this->alg_builder.init_encrypted_weights(this->party, encrypted_weights_precision);

  // record training data ids in training_data_indexes for iteratively batch selection
  std::vector<int> training_data_indexes;
  for (int i = 0; i < this->alg_builder.getter_training_data().size(); i++) {
    training_data_indexes.push_back(i);
  }

  std::cout << "Current channel size is " << this->worker_channels.size() <<std::endl;

  for (int iter = 0; iter < this->alg_builder.max_iteration; iter++) {
    LOG(INFO) << "--------PS Iteration " << iter << " --------";
    std::cout << "--------PS Iteration " << iter << " --------" << std::endl;
     // step 2: broadcast weight
     this->broadcast_encrypted_weights();

     LOG(INFO) << "--------PS Iteration " << iter << ", ps broadcast_encrypted_weights successful" <<" --------";
     std::cout << "--------PS Iteration " << iter << ", ps broadcast_encrypted_weights successful" <<" --------" << std::endl;

     // step 3: randomly select a batch of samples from training data
     // active ps select batch idx and broadcast to passive ps
     std::vector<int> batch_indexes = this->select_batch_idx(training_data_indexes);

     LOG(INFO) << "--------PS Iteration " << iter << ", ps select_batch_idx successful" <<" --------";
     std::cout << "--------PS Iteration " << iter << ", ps select_batch_idx successful" <<" --------" << std::endl;

     // step 4: partition sample ids, since the partition method is deterministic
     // all ps can do this step in its own cluster and broadcast to its own workers
     this->partition_examples(batch_indexes);

     LOG(INFO) << "--------PS Iteration " << iter << ", ps partition_examples successful" <<" --------";
     std::cout << "--------PS Iteration " << iter << ", ps partition_examples successful" <<" --------" << std::endl;

     // step 5: wait worker finish execution
     auto encoded_message = this->wait_worker_complete();
     cout << "LRParameterServer received all loss * X_{ij} from falcon_worker" << endl;

     LOG(INFO) << "--------PS Iteration " << iter << ", LRParameterServer received all loss * X_{ij} from falcon_worker successful" <<" --------";
     std::cout << "--------PS Iteration " << iter << ", LRParameterServer received all loss * X_{ij} from falcon_worker successful" <<" --------" << std::endl;

     // step 6: receive loss, and update weight
     this->update_encrypted_weights(encoded_message);

     LOG(INFO) << "--------PS Iteration " << iter << ", ps update_encrypted_weights successful" <<" --------";
     std::cout << "--------PS Iteration " << iter << ", ps update_encrypted_weights successful" <<" --------" << std::endl;

  }
}

void LRParameterServer::distributed_eval(
    falcon::DatasetType eval_type,
    const std::string& report_save_path) {

  std::string dataset_str = (eval_type == falcon::TRAIN ? "training dataset" : "testing dataset");
  LOG(INFO) << "************* Evaluation on " << dataset_str << " Start *************";

  // step 1: init test data
  int dataset_size =
      (eval_type == falcon::TRAIN) ?
      (int) this->alg_builder.getter_training_data().size() :
      (int) this->alg_builder.getter_testing_data().size();

  std::vector<int> cur_test_data_indexes;
  cur_test_data_indexes.reserve(dataset_size);
  for (int i = 0; i < dataset_size; i++) {
    cur_test_data_indexes.push_back(i);
  }

  // step 2: do the prediction
  auto* decrypted_labels = new EncodedNumber[dataset_size];
  distributed_predict(cur_test_data_indexes, decrypted_labels);

  // step 3: compute and save matrix
  if (party.party_type == falcon::ACTIVE_PARTY) {
    this->alg_builder.eval_matrix_computation_and_save(decrypted_labels,
                                                       dataset_size,
                                                       eval_type,
                                                       report_save_path);
  }

  delete [] decrypted_labels;
}

void LRParameterServer::distributed_predict(
    const std::vector<int>& cur_test_data_indexes,
    EncodedNumber* decrypted_labels) {

  std::cout << "Current channel size is " << this->worker_channels.size() <<std::endl;

  // step 1: partition sample ids, every ps partition in the same way
  std::vector<int> message_sizes = this->partition_examples(cur_test_data_indexes);

  std::cout << "cur_test_data_indexes.size = " << cur_test_data_indexes.size() << std::endl;
  LOG(INFO) << "cur_test_data_indexes.size = " << cur_test_data_indexes.size();

  for (int i = 0; i < message_sizes.size(); i++) {
    std::cout << "message_sizes[" << i << "] = " << message_sizes[i] << std::endl;
    LOG(INFO) << "message_sizes[" << i << "] = " << message_sizes[i];
  }

  // step 2: if active party, wait worker finish execution
  if (party.party_type == falcon::ACTIVE_PARTY) {
    std::vector< string > encoded_messages = this->wait_worker_complete();

    int cur_index = 0;
    // deserialize encrypted predicted labels
    for (int i=0; i < encoded_messages.size(); i++){
      auto partial_predicted_labels = new EncodedNumber[message_sizes[i]];

      deserialize_encoded_number_array(
          partial_predicted_labels,
          message_sizes[i],
          encoded_messages[i]);
      for (int k=0; k < message_sizes[i]; k++){
        decrypted_labels[cur_index] = partial_predicted_labels[k];
        cur_index += 1;
      }
      delete[] partial_predicted_labels;
    }
  }
}


void LRParameterServer::save_model(const std::string& model_save_file){
  // auto* model_weights = new EncodedNumber[this->alg_builder.log_reg_model.weight_size];
  std::string pb_lr_model_string;
  serialize_lr_model(this->alg_builder.log_reg_model, pb_lr_model_string);
  save_pb_model_string(pb_lr_model_string, model_save_file);
}


void launch_lr_parameter_server(
    Party* party,
    const std::string& params_pb_str,
    const std::string& ps_network_config_pb_str,
    const std::string& model_save_file,
    const std::string& model_report_file) {

  LogisticRegressionParams params;
  deserialize_lr_params(params, params_pb_str);

  std::cout << "deserialize lr params on ps success" << std::endl;
  LOG(INFO) << "deserialize lr params on ps success";
  google::FlushLogFiles(google::INFO);

  int weight_size = party->getter_feature_num();
  double training_accuracy = 0.0;
  double testing_accuracy = 0.0;

  std::vector< std::vector<double> > training_data;
  std::vector< std::vector<double> > testing_data;
  std::vector<double> training_labels;
  std::vector<double> testing_labels;

  // ps split the train and test data, after splitting, the ps on all parties
  // have the train and test data, later ps will broadcast the split info to workers
  double split_percentage = SPLIT_TRAIN_TEST_RATIO;
  party->split_train_test_data(split_percentage,
      training_data,
      testing_data,
      training_labels,
      testing_labels);

  std::cout << "PS Init logistic regression model";
  std::cout << "params.batch_size = " << params.batch_size;
  std::cout << "params.max_iteration = " << params.max_iteration;
  std::cout << "params.converge_threshold = " << params.converge_threshold;
  std::cout << "params.with_regularization = " << params.with_regularization;
  std::cout << "params.alpha = " << params.alpha;
  std::cout << "params.learning_rate = " << params.learning_rate;
  std::cout << "params.decay = " << params.decay;
  std::cout << "params.penalty = " << params.penalty;
  std::cout << "params.optimizer = " << params.optimizer;
  std::cout << "params.multi_class = " << params.multi_class;
  std::cout << "params.metric = " << params.metric;
  std::cout << "params.dp_budget = " << params.dp_budget;

  LOG(INFO) << "PS Init logistic regression model";
  LOG(INFO) << "params.batch_size = " << params.batch_size;
  LOG(INFO) << "params.max_iteration = " << params.max_iteration;
  LOG(INFO) << "params.converge_threshold = " << params.converge_threshold;
  LOG(INFO) << "params.with_regularization = " << params.with_regularization;
  LOG(INFO) << "params.alpha = " << params.alpha;
  LOG(INFO) << "params.learning_rate = " << params.learning_rate;
  LOG(INFO) << "params.decay = " << params.decay;
  LOG(INFO) << "params.penalty = " << params.penalty;
  LOG(INFO) << "params.optimizer = " << params.optimizer;
  LOG(INFO) << "params.multi_class = " << params.multi_class;
  LOG(INFO) << "params.metric = " << params.metric;
  LOG(INFO) << "params.dp_budget = " << params.dp_budget;

  std::cout << "PS Init logistic regression model" << std::endl;
  LOG(INFO) << "PS Init logistic regression model";

  // in distributed train, fit bias should be handled here
  LOG(INFO) << "original weight_size = " << weight_size << std::endl;
  // retrieve the fit_bias term
  bool m_fit_bias = params.fit_bias;
  // if this is active party, and fit_bias is true
  // fit_bias or fit_intercept, for whether to plus the
  // constant _bias_ or _intercept_ term
  if ((party->party_type == falcon::ACTIVE_PARTY) && m_fit_bias) {
    LOG(INFO) << "will insert x1=1 to features\n";
    // insert x1=1 to front of the features
    double x1 = 1.0;
    for (int i = 0; i < training_data.size(); i++) {
      training_data[i].insert(training_data[i].begin(), x1);
    }
    for (int i = 0; i < testing_data.size(); i++) {
      testing_data[i].insert(testing_data[i].begin(), x1);
    }
    // update the new feature_num for the active party
    // also update the weight_size value +1
    // before passing weight_size to LogisticRegression instance below
    party->setter_feature_num(++weight_size);
    LOG(INFO) << "updated weight_size = " << weight_size << std::endl;
    LOG(INFO) << "party getter feature_num = " << party->getter_feature_num() << std::endl;
  }

  // init log_reg_model instance
  auto log_reg_model_builder = new LogisticRegressionBuilder (
      params,
      weight_size,
      training_data,
      testing_data,
      training_labels,
      testing_labels,
      training_accuracy,
      testing_accuracy);

  LOG(INFO) << "Init logistic regression model success";
  std::cout << "Init logistic regression model success" << std::endl;

  auto ps = new LRParameterServer(*log_reg_model_builder, *party, ps_network_config_pb_str);

  // here need to send the train/test data/labels to workers
  // also, need to send the phe keys to workers
  // accordingly, workers have to receive and deserialize these
  ps->broadcast_train_test_data(training_data, testing_data, training_labels, testing_labels);
  ps->broadcast_phe_keys();

  google::FlushLogFiles(google::INFO);

  // start to train the task in a distributed way
  ps->distributed_train();

  // evaluate the model on the training and testing datasets
  ps->distributed_eval(falcon::TRAIN, model_report_file);
  ps->distributed_eval(falcon::TEST, model_report_file);

  // save the trained model
  ps->save_model(model_save_file);

  delete log_reg_model_builder;
  delete ps;
}