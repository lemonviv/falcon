//
// Created by root on 11/11/21.
//

#include <falcon/inference/interpretability/lime/lime.h>
#include <falcon/utils/logger/logger.h>
#include <falcon/utils/parser.h>
#include <falcon/utils/pb_converter/interpretability_converter.h>
#include <falcon/utils/io_util.h>
#include <falcon/algorithm/model_builder_helper.h>
#include <falcon/inference/interpretability/lime/scaler.h>
#include <falcon/utils/pb_converter/common_converter.h>
#include <falcon/utils/math/math_ops.h>
#include <Networking/ssl_sockets.h>
#include <falcon/operator/mpc/spdz_connector.h>
#include <falcon/algorithm/vertical/linear_model/linear_regression_builder.h>
#include <falcon/algorithm/vertical/linear_model/linear_regression_ps.h>
#include <falcon/utils/pb_converter/alg_params_converter.h>
#include <falcon/utils/logger/log_alg_params.h>
#include <falcon/utils/base64.h>

#include <glog/logging.h>

#include <utility>
#include <falcon/model/model_io.h>
#include <falcon/utils/pb_converter/lr_converter.h>
#include <random>

std::vector<std::vector<double>> LimeExplainer::generate_random_samples(const Party &party,
                                                                        StandardScaler* scaler,
                                                                        bool sample_around_instance,
                                                                        const std::vector<double> &data_row,
                                                                        int sample_instance_num,
                                                                        const std::string &sampling_method) {
  // TODO: assume all the features are numerical, categorical features not implemented
  int feature_num = (int) data_row.size();
  std::vector<std::vector<double>> sampled_data;
  sampled_data.push_back(data_row);

  // random initialization with a normal distribution
  if (sampling_method == "gaussian") {
    std::mt19937 mt(RANDOM_SEED);
    std::normal_distribution<> nd(0, 1);
    for (int i = 0; i < sample_instance_num; i++) {
      std::vector<double> sample;
      sample.reserve(feature_num);
      for (int j = 0; j < feature_num; j++) {
        sample.push_back(nd(mt));
      }
      sampled_data.push_back(sample);
    }
  } else {
    log_error("Sampling method other than gaussian is not implemented");
    exit(EXIT_FAILURE);
  }

  // scale the sampled data, note that the first row is origin_data, no need to scale
  for (int i = 0; i < sample_instance_num; i++) {
    for (int j = 0; j < feature_num; j++) {
      if (sample_around_instance) {
        sampled_data[i+1][j] = (sampled_data[i+1][j] * (scaler->scale[j]) + data_row[j]);
      } else {
        sampled_data[i+1][j] = (sampled_data[i+1][j] * (scaler->scale[j]) + scaler->mean[j]);
      }
    }
  }
  return sampled_data;
}

void LimeExplainer::load_predict_origin_model(const Party &party,
                                              const std::string& origin_model_name,
                                              const std::string& origin_model_saved_file,
                                              std::vector<std::vector<double>> generated_samples,
                                              std::string model_type,
                                              int class_num,
                                              EncodedNumber **predictions) {
  int num_total_samples = (int) generated_samples.size();
  falcon::AlgorithmName model_name = parse_algorithm_name(origin_model_name);
  switch (model_name) {
    case falcon::LOG_REG:
    {
      LogisticRegressionModel saved_log_reg_model;
      std::string saved_model_string;
      load_pb_model_string(saved_model_string, origin_model_saved_file);
      deserialize_lr_model(saved_log_reg_model, saved_model_string);
      saved_log_reg_model.predict_proba(party,
                                        generated_samples,
                                        predictions);
      break;
    }
    case falcon::RF:
    {
      log_error("Random forest model not integrated yet.");
      break;
    }
    case falcon::GBDT:
    {
      log_error("GBDT model not integrated yet.");
      break;
    }
    default:
      break;
  }
}

void LimeExplainer::compute_sample_weights(
    const Party &party,
    const std::string &generated_sample_file,
    const std::string &computed_prediction_file,
    bool is_precompute,
    int num_samples,
    int class_num,
    const std::string &distance_metric,
    const std::string &kernel,
    double kernel_width,
    const std::string &sample_weights_file,
    const std::string &selected_sample_file,
    const std::string &selected_prediction_file) {
  // the parties do the following steps for computing sample weights
  //  1. read the generated sample file
  //  2. read the computed_prediction file (note that
  //      we assume all parties have the same prediction file now)
  //  3. check whether is_precompute is true:
  //      if true: parties randomly select num_samples and update
  //          selected_sample_file and selected_prediction_file
  //      else: parties directly write the generated_sample_file
  //          and computed_prediction_file as the above two files
  //  4. compute the distance between the first row and the rest given
  //      the distance metric, need to call spdz program
  //  5. compute the sample weights based on kernel and kernel_width
  //      also need to call spdz program
  //  6. write the encrypted sample weights to the sample_weights_file

  // 1. read the generated sample file
  char delimiter = ',';
  std::vector<std::vector<double>> generated_samples =
      read_dataset(generated_sample_file, delimiter);
  std::vector<double> origin_data = generated_samples[0];
  int generated_samples_size = (int) generated_samples.size();
  log_info("Read the generated samples finished");

  // 2. read the computed_prediction file (note that
  //     we assume all parties have the same prediction file now)
  auto** computed_predictions = new EncodedNumber*[generated_samples_size];
  for (int i = 0; i < generated_samples_size; i++) {
    computed_predictions[i] = new EncodedNumber[class_num];
  }
  read_encoded_number_matrix_file(computed_predictions,
                                  generated_samples_size,
                                  class_num,
                                  computed_prediction_file);
  log_info("Read the computed predictions finished");

  // 3. differentiate is_precompute situation
  std::vector<std::vector<double>> selected_samples;
  if (is_precompute && (generated_samples_size > num_samples)) {
    // random select sample idx
    std::vector<int> selected_samples_idx = random_select_sample_idx(
        party, generated_samples_size, num_samples);
    // re-aggregate selected samples, size should be num_samples + 1
    for (int i : selected_samples_idx) {
      selected_samples.push_back(generated_samples[i]);
    }
    // select the computed_predictions
    auto** selected_predictions = new EncodedNumber*[selected_samples_idx.size()];
    for (int i = 0; i < selected_samples_idx.size(); i++) {
      selected_predictions[i] = new EncodedNumber[class_num];
    }
    for (int i = 0; i < selected_samples_idx.size(); i++) {
      for (int j = 0; j < class_num; j++) {
        int idx = selected_samples_idx[i];
        selected_predictions[i][j] = computed_predictions[idx][j];
      }
    }

    // write the selected_samples and selected_predictions
    write_dataset_to_file(selected_samples,
                          delimiter,
                          selected_sample_file);
    write_encoded_number_matrix_to_file(selected_predictions,
                                        (int) selected_samples_idx.size(),
                                        class_num,
                                        selected_prediction_file);
    // free memory
    for (int i = 0; i < selected_samples_idx.size(); i++) {
      delete [] selected_predictions[i];
    }
    delete [] selected_predictions;
  } else {
    // directly write the generated_samples and computed_predictions
    selected_samples = generated_samples;
    write_dataset_to_file(generated_samples,
                          delimiter,
                          selected_sample_file);
    write_encoded_number_matrix_to_file(computed_predictions,
                                        generated_samples_size,
                                        class_num,
                                        selected_prediction_file);
  }
  log_info("Select samples and computed predictions finished");

  //  4. compute the distance between the first row and the rest given
  //      the distance metric, need to call spdz program
  //  5. compute the sample weights based on kernel and kernel_width
  //      also need to call spdz program
  int selected_sample_size = (int) selected_samples.size();
  auto* encrypted_weights = new EncodedNumber[selected_sample_size];
  compute_dist_weights(party,
                       encrypted_weights,
                       selected_samples[0],
                       selected_samples,
                       distance_metric,
                       kernel,
                       kernel_width);
  log_info("Compute dist and weights finished");

//  // for debug
//  auto* dec_weights = new EncodedNumber[selected_sample_size];
//  party.collaborative_decrypt(encrypted_weights, dec_weights, selected_sample_size, ACTIVE_PARTY_ID);
//  for (int i = 0; i < selected_sample_size; i++) {
//    double x;
//    dec_weights[i].decode(x);
//    log_info("[compute_sample_weights]: dec_weights[" + std::to_string(i) + "] = " + std::to_string(x));
//  }
//  delete [] dec_weights;

  //  6. write the encrypted sample weights to the sample_weights_file
  write_encoded_number_array_to_file(encrypted_weights,
                                     selected_sample_size,
                                     sample_weights_file);
  log_info("Write encrypted sample weights finished");

  for (int i = 0; i < generated_samples_size; i++) {
    delete [] computed_predictions[i];
  }
  delete [] computed_predictions;
  delete [] encrypted_weights;
}

std::vector<int> LimeExplainer::random_select_sample_idx(
    const Party& party,
    int generated_samples_size,
    int num_samples) {
  // active party random select samples and predictions
  std::vector<int> selected_samples_idx;
  // the first row should always in the selected data
  if (party.party_type == falcon::ACTIVE_PARTY) {
    selected_samples_idx.push_back(0);
    std::vector<int> generated_samples_idx;
    for (int i = 1; i < generated_samples_size; i++) {
      generated_samples_idx.push_back(i);
    }
    std::default_random_engine rng(RANDOM_SEED);
    std::shuffle(std::begin(generated_samples_idx), std::end(generated_samples_idx), rng);
    for (int i = 0; i < num_samples; i++) {
      selected_samples_idx.push_back(generated_samples_idx[i]);
    }
    // serialize and broadcast to the other parties
    std::string selected_samples_idx_str;
    serialize_int_array(selected_samples_idx, selected_samples_idx_str);
    for (int id = 0; id < party.party_num; id++) {
      if (id != party.party_id) {
        party.send_long_message(id, selected_samples_idx_str);
      }
    }
  } else {
    // receive selected_samples_idx_str and deserialize
    std::string recv_selected_samples_idx_str;
    deserialize_int_array(selected_samples_idx, recv_selected_samples_idx_str);
  }
  return selected_samples_idx;
}

void LimeExplainer::compute_dist_weights(const Party &party,
                                         EncodedNumber* weights,
                                         const std::vector<double>& origin_data,
                                         const std::vector<std::vector<double>>& sampled_data,
                                         const std::string& distance_metric,
                                         const std::string& kernel,
                                         double kernel_width) {
  // check distance metric and kernel method
  if (distance_metric != "euclidean") {
    log_error("The distance metric " + distance_metric + " is not supported.");
    exit(EXIT_FAILURE);
  }
  if (kernel != "exponential") {
    log_error("The kernel function " + kernel + " is not supported.");
    exit(EXIT_FAILURE);
  }

  // required by spdz connector and mpc computation
  bigint::init_thread();

  // do the following steps to compute the distance and sample weights
  //  1. parties locally compute \sum_{i=1}^{d_i} (x[i] - s[i])^2
  int feature_size = (int) origin_data.size();
  std::vector<int> feature_sizes = party.sync_up_int_arr(feature_size);
  int total_feature_size = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0);

  //  2. parties aggregate and convert to secret shares
  log_info("[compute_dist_weights]: begin to compute square dist.");
  int sample_size = (int) sampled_data.size();
  auto* squared_dist = new EncodedNumber[sample_size];
  compute_squared_dist(party, origin_data, sampled_data, squared_dist);
  log_info("[compute_dist_weights]: finish to compute square dist.");
  std::vector<double> squared_dist_shares;
  party.ciphers_to_secret_shares(
      squared_dist,
      squared_dist_shares,
      sample_size,
      ACTIVE_PARTY_ID,
      PHE_FIXED_POINT_PRECISION);
  log_info("[compute_dist_weights]: compute encrypted distance finished");

  //  3. parties compute kernel width (replace the input one by default now -- let spdz compute)
  // kernel_width = std::sqrt(total_feature_size) * 0.75;

  //  4. parties connect to spdz and compute weights (both sqrt and exponential kernel)
  //    the spdz program compute sqrt(dist) and compute exponential kernel weights
  std::vector<int> public_values;
  public_values.push_back(sample_size);
  public_values.push_back(total_feature_size);

  falcon::SpdzLimeCompType comp_type = falcon::DIST_WEIGHT;
  std::promise<std::vector<double>> promise_values;
  std::future<std::vector<double>> future_values = promise_values.get_future();
  std::thread spdz_dist_weights(spdz_lime_computation,
                                party.party_num,
                                party.party_id,
                                party.executor_mpc_ports,
                                party.host_names,
                                public_values.size(),
                                public_values,
                                squared_dist_shares.size(),
                                squared_dist_shares,
                                comp_type,
                                &promise_values);
  std::vector<double> res = future_values.get();
  spdz_dist_weights.join();
  log_info("[compute_dist_weights]: communicate with spdz finished");
  log_info("[compute_dist_weights]: res.size = " + std::to_string(res.size()));

  // 5. convert to cipher and return
  party.secret_shares_to_ciphers(weights, res, sample_size,
                                 ACTIVE_PARTY_ID, PHE_FIXED_POINT_PRECISION);
  log_info("[compute_dist_weights]: shares to ciphers finished");
  delete [] squared_dist;
}

void LimeExplainer::compute_squared_dist(const Party &party,
                                         const std::vector<double> &origin_data,
                                         const std::vector<std::vector<double>> &sampled_data,
                                         EncodedNumber *squared_dist) {
  int sample_size = (int) sampled_data.size();
  log_info("[compute_squared_dist]: sample_size = " + std::to_string(sample_size));
  std::vector<double> local_squared_sum;
  for (int i = 0; i < sample_size; i++) {
    double ss = square_sum(origin_data, sampled_data[i]);
    local_squared_sum.push_back(ss);
  }

  djcs_t_public_key* phe_pub_key = djcs_t_init_public_key();
  party.getter_phe_pub_key(phe_pub_key);

  auto* local_squared_dist = new EncodedNumber[sample_size];
  for (int i = 0; i < sample_size; i++) {
    local_squared_dist[i].set_double(phe_pub_key->n[0],
                                     local_squared_sum[i],
                                     PHE_FIXED_POINT_PRECISION);
    djcs_t_aux_encrypt(phe_pub_key, party.phe_random,
                       local_squared_dist[i], local_squared_dist[i]);
  }

  // active party aggregate the local_squared_dist arr and broadcast
  if (party.party_type == falcon::ACTIVE_PARTY) {
    // copy local_squared_dist to squared_dist
    for (int i = 0; i < sample_size; i++) {
      squared_dist[i] = local_squared_dist[i];
    }
    // recv other parties info and aggregate
    for (int id = 0; id < party.party_num; id++) {
      if (id != party.party_id) {
        std::string recv_local_squared_dist_str;
        party.recv_long_message(id, recv_local_squared_dist_str);
        auto* recv_local_squared_dist = new EncodedNumber[sample_size];
        deserialize_encoded_number_array(recv_local_squared_dist,
                                         sample_size,
                                         recv_local_squared_dist_str);
        // phe aggregate
        for (int i = 0; i < sample_size; i++) {
          djcs_t_aux_ee_add(phe_pub_key,
                            squared_dist[i],
                            squared_dist[i],
                            recv_local_squared_dist[i]);
        }
        delete [] recv_local_squared_dist;
      }
    }
  } else {
    // send local_squared_dist to active party for aggregation
    std::string local_squared_dist_str;
    serialize_encoded_number_array(local_squared_dist, sample_size, local_squared_dist_str);
    party.send_long_message(ACTIVE_PARTY_ID, local_squared_dist_str);
  }
  // broadcast squared_dist array
  party.broadcast_encoded_number_array(squared_dist, sample_size, ACTIVE_PARTY_ID);

  delete [] local_squared_dist;
  djcs_t_free_public_key(phe_pub_key);
}

void LimeExplainer::select_features(Party party,
                                    const std::string &selected_samples_file,
                                    const std::string &selected_predictions_file,
                                    const std::string &sample_weights_file,
                                    int num_samples,
                                    int class_num,
                                    int class_id,
                                    const std::string &feature_selection,
                                    int num_explained_features,
                                    const std::string &selected_features_file) {
  // currently, only support pearson-based feature selection
  if (feature_selection != "pearson") {
    log_error("The feature selection method " + feature_selection + " not supported");
    exit(EXIT_FAILURE);
  }
  // do the following steps to selected feature
  // 1. read the selected sample file
  char delimiter = ',';
  std::vector<std::vector<double>> selected_samples =
      read_dataset(selected_samples_file, delimiter);
  std::vector<double> origin_data = selected_samples[0];
  int selected_sample_size = (int) selected_samples.size();
  log_info("Read the generated samples finished");

  // 2. read the computed_prediction file (note that
  //     we assume all parties have the same prediction file now)
  auto** selected_predictions = new EncodedNumber*[selected_sample_size];
  for (int i = 0; i < selected_sample_size; i++) {
    selected_predictions[i] = new EncodedNumber[class_num];
  }
  read_encoded_number_matrix_file(selected_predictions,
                                  selected_sample_size,
                                  class_num,
                                  selected_predictions_file);
  // extract the class_id computed predictions
  auto* selected_pred_class_id = new EncodedNumber[selected_sample_size];
  for (int i = 0; i < selected_sample_size; i++) {
    selected_pred_class_id[i] = selected_predictions[i][class_id];
  }
  log_info("Read the computed predictions finished");

  // 3. read the sample weights file (optional at the moment)
  // TODO: check feature selection with sample weights
  // TODO: check what secret shares need to be sent to spdz program

  // 4. connect to spdz program and return selected_feature_idx (which is public)
  std::vector<double> selected_pred_shares;
  party.ciphers_to_secret_shares(selected_pred_class_id,
                                 selected_pred_shares,
                                 selected_sample_size,
                                 ACTIVE_PARTY_ID,
                                 PHE_FIXED_POINT_PRECISION);
  std::vector<int> public_values;
  public_values.push_back(selected_sample_size);
  public_values.push_back(num_explained_features);
  falcon::SpdzLimeCompType comp_type = falcon::PEARSON;
  std::promise<std::vector<double>> promise_values;
  std::future<std::vector<double>> future_values = promise_values.get_future();
  std::thread spdz_pearson_comp(spdz_lime_computation,
                                party.party_num,
                                party.party_id,
                                party.executor_mpc_ports,
                                party.host_names,
                                public_values.size(),
                                public_values,
                                selected_pred_shares.size(),
                                selected_pred_shares,
                                comp_type,
                                &promise_values);
  std::vector<double> res = future_values.get();
  spdz_pearson_comp.join();
  log_error("[select_features]: communicate with spdz finished");

  // 5. convert the returned res into int vector, and re-organize the selected_features_file
  std::vector<int> selected_feat_idx;
  for (int i = 0; i < res.size(); i++) {
    selected_feat_idx[i] = (int) res[i];
  }
  std::vector<std::vector<double>> selected_feat_samples;
  for (int i = 0; i < selected_sample_size; i++) {
    std::vector<double> sample;
    for (int j = 0; j < num_explained_features; j++) {
      sample.push_back(selected_samples[i][selected_feat_idx[j]]);
    }
    selected_feat_samples.push_back(sample);
  }
  write_dataset_to_file(selected_feat_samples, delimiter, selected_features_file);

  for (int i = 0; i < selected_sample_size; i++) {
    delete [] selected_predictions[i];
  }
  delete [] selected_predictions;
  delete [] selected_pred_class_id;
}

std::vector<double> LimeExplainer::interpret(const Party &party,
                                             const std::string &selected_data_file,
                                             const std::string &selected_predictions_file,
                                             const std::string &sample_weights_file,
                                             int num_samples,
                                             int class_num,
                                             int class_id,
                                             const std::string &interpret_model_name,
                                             const std::string &interpret_model_param,
                                             const std::string &explanation_report,
                                             const std::string& ps_network_str,
                                             int is_distributed,
                                             int distributed_role,
                                             int worker_id) {
  // for interpret model training, do the following steps:
  //  1. read the selected_features_file to get the training data (the first row should be
  //      the origin data to be explained)
  //  2. read the selected_predictions_file to get the encrypted labels (with probabilities)
  //  3. read the sample_weights_file to get the encrypted sample weights for each data
  //  4. loop for class_num (currently num_top_labels is reserved for selecting labels to explain)
  //    4.1. prepare the information for invoking the corresponding training API
  //  5. return the result for display and save the explanations to the report

  //  1. read the selected_features_file to get the training data (the first row should be
  //      the origin data to be explained)
  char delimiter = ',';
  std::vector<std::vector<double>> selected_data = read_dataset(selected_data_file, delimiter);
  int cur_sample_size = (int) selected_data.size();

  //  2. read the selected_predictions_file to get the encrypted labels (with probabilities)
  auto** encrypted_predictions = new EncodedNumber*[cur_sample_size];
  for (int i = 0; i < cur_sample_size; i++) {
    encrypted_predictions[i] = new EncodedNumber[class_num];
  }
  read_encoded_number_matrix_file(encrypted_predictions,
                                  cur_sample_size,
                                  class_num,
                                  selected_predictions_file);

  //  3. read the sample_weights_file to get the encrypted sample weights for each data
  auto* encrypted_weights = new EncodedNumber[cur_sample_size];
  read_encoded_number_array_file(encrypted_weights,
                                 cur_sample_size,
                                 sample_weights_file);

  //  4. prepare the information for invoking the corresponding training API
  // extract the k-th model predictions
  std::vector<std::vector<double>> wrap_explanations;
  auto* kth_predictions = new EncodedNumber[cur_sample_size];
  for (int i = 0; i < cur_sample_size; i++) {
    kth_predictions[i] = encrypted_predictions[i][class_id];
  }
  std::vector<double> explanations = explain_one_label(party,
                                                       selected_data,
                                                       kth_predictions,
                                                       encrypted_weights,
                                                       cur_sample_size,
                                                       interpret_model_name,
                                                       interpret_model_param,
                                                       ps_network_str,
                                                       is_distributed,
                                                       distributed_role,
                                                       worker_id);

  //  5. return the result for display and save the explanations to the report
  wrap_explanations.push_back(explanations);
  write_dataset_to_file(wrap_explanations, delimiter, explanation_report);

  for (int i = 0; i < cur_sample_size; i++) {
    delete [] encrypted_predictions[i];
  }
  delete [] encrypted_predictions;
  delete [] encrypted_weights;
  delete [] kth_predictions;

  return explanations;
}

std::vector<double> LimeExplainer::explain_one_label(
    const Party &party,
    std::vector<std::vector<double>> train_data,
    EncodedNumber* predictions,
    EncodedNumber* sample_weights,
    int num_samples,
    const std::string &interpret_model_name,
    const std::string &interpret_model_param,
    const std::string& ps_network_str,
    int is_distributed,
    int distributed_role,
    int worker_id) {
  // for explaining one label, we do the following steps:
  //  1. given the interpret_model_name and interpret_model_param, init the corresponding builder
  //  2. invoke the train function directly (need to make sure that it supports sample_weights)
  //  3. after training finished, decrypt the model weights and return
  //    TODO: what if the model is decision tree?

  std::vector<double> explanations;
  falcon::AlgorithmName algorithm_name = parse_algorithm_name(interpret_model_name);
  switch (algorithm_name) {
    case falcon::LINEAR_REG: {
      // call linear regression training
      explanations = lime_linear_reg_train(
          party,
          interpret_model_param,
          train_data,
          predictions,
          sample_weights,
          ps_network_str,
          is_distributed,
          distributed_role,
          worker_id);
      break;
    }
    case falcon::DT: {
      // call decision tree training
      break;
    }
    default:
      log_error("The interpret algorithm " + interpret_model_name + " is not supported.");
      break;
  }

  return explanations;
}

std::vector<double> LimeExplainer::lime_linear_reg_train(
    Party party,
    const std::string &linear_reg_param_str,
    const std::vector<std::vector<double>>& train_data,
    EncodedNumber *predictions,
    EncodedNumber *sample_weights,
    const std::string& ps_network_str,
    int is_distributed,
    int distributed_role,
    int worker_id) {
  std::vector<double> local_explanations;
  // deserialize linear regression params
  LinearRegressionParams linear_reg_params;
  //  // create for debug
  //  linear_reg_params.batch_size = 8;
  //  linear_reg_params.max_iteration = 100;
  //  linear_reg_params.converge_threshold = 0.0001;
  //  linear_reg_params.with_regularization = true;
  //  linear_reg_params.alpha = 0.1;
  //  linear_reg_params.learning_rate = 0.1;
  //  linear_reg_params.decay = 0.1;
  //  linear_reg_params.penalty = "l2";
  //  linear_reg_params.optimizer = "sgd";
  //  linear_reg_params.metric = "mse";
  //  linear_reg_params.dp_budget = 0.1;
  //  linear_reg_params.fit_bias = true;

  std::string linear_reg_param_pb_str = base64_decode_to_pb_string(linear_reg_param_str);
  deserialize_lir_params(linear_reg_params, linear_reg_param_pb_str);
  log_linear_regression_params(linear_reg_params);

  // train a lime linear regression model
  std::vector<std::vector<double>> dummy_test_data;
  std::vector<double> dummy_train_labels, dummy_test_labels;
  double dummy_train_accuracy, dummy_test_accuracy;
  std::vector<std::vector<double>> used_train_data = train_data;
  int weight_size = party.getter_feature_num();
  if (party.party_type == falcon::ACTIVE_PARTY && linear_reg_params.fit_bias) {
    log_info("fit_bias = TRUE, will insert x1=1 to features");
    // insert x1=1 to front of the features
    double x1 = BIAS_VALUE;
    for (int i = 0; i < used_train_data.size(); i++) {
      used_train_data[i].insert(used_train_data[i].begin(), x1);
    }
    party.setter_feature_num(++weight_size);
  }

  if (is_distributed == 0) {
    LinearRegressionBuilder linear_reg_builder(linear_reg_params,
                                               weight_size,
                                               used_train_data,
                                               dummy_test_data,
                                               dummy_train_labels,
                                               dummy_test_labels,
                                               dummy_train_accuracy,
                                               dummy_test_accuracy);
    log_info("Init linear regression builder finished");
    linear_reg_builder.lime_train(party,
                                  true, predictions,
                                  true, sample_weights);
    log_info("Train lime linear regression model finished.");
    // decrypt the model weights and return
    local_explanations =
        linear_reg_builder.linear_reg_model.display_weights(party);
    log_info("Decrypt and display local explanations finished");
  }

  // is_distributed == 1 and distributed_role = 0 (parameter server)
  if (is_distributed == 1 && distributed_role == 0) {
    log_info("************* Distributed LIME Interpret *************");
    // init linear_reg instance
    auto linear_reg_model_builder = new LinearRegressionBuilder (
        linear_reg_params,
        weight_size,
        used_train_data,
        dummy_test_data,
        dummy_train_labels,
        dummy_test_labels,
        dummy_train_accuracy,
        dummy_test_accuracy);
    log_info("[lime_linear_reg_train ps]: init linear regression model");
    auto ps = new LinearRegParameterServer(*linear_reg_model_builder,
                                           party,
                                           ps_network_str);
    log_info("[lime_linear_reg_train ps]: Init ps finished.");
    // start to train the task in a distributed way
    ps->distributed_lime_train(true, predictions,
                               true, sample_weights);
    log_info("[lime_linear_reg_train ps]: distributed train finished.");
  }

  // is_distributed == 1 and distributed_role = 1 (worker)
  if (is_distributed == 1 && distributed_role == 1) {
    log_info("************* Distributed LIME Interpret *************");
    // init linear_reg instance
    auto linear_reg_model_builder = new LinearRegressionBuilder (
        linear_reg_params,
        weight_size,
        used_train_data,
        dummy_test_data,
        dummy_train_labels,
        dummy_test_labels,
        dummy_train_accuracy,
        dummy_test_accuracy);
    // worker is created to communicate with parameter server
    auto worker = new Worker(ps_network_str, worker_id);

    log_info("[lime_linear_reg_train worker]: init linear regression model");
    linear_reg_model_builder->distributed_lime_train(party, *worker,
                                                     true, predictions,
                                                     true, sample_weights);
  }

  return local_explanations;
}

void lime_comp_pred(Party party, const std::string& params_str) {
  log_info("Begin to compute the lime required samples and predictions");
  // 1. deserialize the LimePreComputeParams
  LimeCompPredictionParams comp_prediction_params;
  // set the values for local debug
  std::string path_prefix = "/opt/falcon/exps/breast_cancer/client" + std::to_string(party.party_id);
//  comp_prediction_params.original_model_name = "logistic_regression";
//  comp_prediction_params.original_model_saved_file = path_prefix + "/log_reg/saved_model.pb";
//  comp_prediction_params.model_type = "classification";
//  comp_prediction_params.class_num = 2;
//  comp_prediction_params.explain_instance_idx = 0;
//  comp_prediction_params.sample_around_instance = true;
//  comp_prediction_params.num_total_samples = 1000;
//  comp_prediction_params.sampling_method = "gaussian";
//  comp_prediction_params.generated_sample_file = path_prefix + "/log_reg/sampled_data.txt";
//  comp_prediction_params.computed_prediction_file = path_prefix + "/log_reg/predictions.txt";

  std::string comp_pred_params_str = base64_decode_to_pb_string(params_str);
  deserialize_lime_comp_pred_params(comp_prediction_params, comp_pred_params_str);
  comp_prediction_params.original_model_saved_file = path_prefix + comp_prediction_params.original_model_saved_file;
  comp_prediction_params.generated_sample_file = path_prefix + comp_prediction_params.generated_sample_file;
  comp_prediction_params.computed_prediction_file = path_prefix + comp_prediction_params.computed_prediction_file;
  log_info("Deserialize the lime comp_prediction params");

  // 2. generate random samples
  LimeExplainer lime_explainer;
  int num_total_samples = comp_prediction_params.num_total_samples;
  int class_num = comp_prediction_params.class_num;
  std::vector<std::vector<double>> train_data, test_data;
  std::vector<double> train_labels, test_labels;
  split_dataset(&party, false,train_data, test_data,
                train_labels, test_labels, SPLIT_TRAIN_TEST_RATIO);
  auto* scaler = new StandardScaler(true, true);
  scaler->fit(train_data, train_labels);
  std::vector<double> data_row = train_data[comp_prediction_params.explain_instance_idx];
  std::vector<std::vector<double>> generated_samples = lime_explainer.generate_random_samples(
      party,
      scaler,
      comp_prediction_params.sample_around_instance,
      data_row,
      num_total_samples,
      comp_prediction_params.sampling_method
      );
  log_info("Finish generating random samples");

  // 3. load model and compute model predictions
  int cur_sample_size = (int) generated_samples.size();
  auto** predictions = new EncodedNumber*[cur_sample_size];
  for (int i = 0; i < cur_sample_size; i++) {
    predictions[i] = new EncodedNumber[class_num];
  }
  lime_explainer.load_predict_origin_model(party,
                                           comp_prediction_params.original_model_name,
                                           comp_prediction_params.original_model_saved_file,
                                           generated_samples,
                                           comp_prediction_params.model_type,
                                           comp_prediction_params.class_num,
                                           predictions);
  log_info("Load the model and compute model predictions");

  // 4. save the generated samples and model predictions to the corresponding file
  char delimiter = ',';
  write_dataset_to_file(generated_samples,
                        delimiter,
                        comp_prediction_params.generated_sample_file);
  write_encoded_number_matrix_to_file(predictions,
                                      cur_sample_size,
                                      class_num,
                                      comp_prediction_params.computed_prediction_file);
  log_info("Save the generated samples and predictions to file");

  // free information
  for (int i = 0; i < cur_sample_size; i++) {
    delete [] predictions[i];
  }
  delete [] predictions;
  delete scaler;
}

void lime_comp_weight(Party party, const std::string& params_str) {
  log_info("Begin to compute sample weights");
  // deserialize the LimeCompWeightsParams
  LimeCompWeightsParams comp_weights_params;
  // set the values for local debug
  std::string path_prefix = "/opt/falcon/exps/breast_cancer/client" + std::to_string(party.party_id);
//  comp_weights_params.explain_instance_idx = 0;
//  comp_weights_params.generated_sample_file = path_prefix + "/log_reg/sampled_data.txt";
//  comp_weights_params.computed_prediction_file = path_prefix + "/log_reg/predictions.txt";
//  comp_weights_params.is_precompute = false;
//  comp_weights_params.num_samples = 1000;
//  comp_weights_params.class_num = 2;
//  comp_weights_params.distance_metric = "euclidean";
//  comp_weights_params.kernel = "exponential";
//  comp_weights_params.kernel_width = 0.75;
//  comp_weights_params.sample_weights_file = path_prefix + "/log_reg/sample_weights.txt";
//  comp_weights_params.selected_samples_file = path_prefix + "/log_reg/selected_sampled_data.txt";
//  comp_weights_params.selected_predictions_file = path_prefix + "/log_reg/selected_predictions.txt";

  std::string comp_weight_params_str = base64_decode_to_pb_string(params_str);
  deserialize_lime_comp_weights_params(comp_weights_params, comp_weight_params_str);
  comp_weights_params.generated_sample_file = path_prefix + comp_weights_params.generated_sample_file;
  comp_weights_params.computed_prediction_file = path_prefix + comp_weights_params.computed_prediction_file;
  comp_weights_params.sample_weights_file = path_prefix + comp_weights_params.sample_weights_file;
  comp_weights_params.selected_samples_file = path_prefix + comp_weights_params.selected_samples_file;
  comp_weights_params.selected_predictions_file = path_prefix + comp_weights_params.selected_predictions_file;
  log_info("Deserialize the lime comp_weights params");

  // call lime_explainer compute_sample_weights function
  LimeExplainer lime_explainer;
  lime_explainer.compute_sample_weights(
      party,
      comp_weights_params.generated_sample_file,
      comp_weights_params.computed_prediction_file,
      comp_weights_params.is_precompute,
      comp_weights_params.num_samples,
      comp_weights_params.class_num,
      comp_weights_params.distance_metric,
      comp_weights_params.kernel,
      comp_weights_params.kernel_width,
      comp_weights_params.sample_weights_file,
      comp_weights_params.selected_samples_file,
      comp_weights_params.selected_predictions_file);
  log_info("Finish computing sample weights");
}

void lime_feat_sel(Party party, const std::string& params_str) {
  log_info("Begin to select features");
  // deserialize the LimeCompWeightsParams
  LimeFeatSelParams feat_sel_params;
  deserialize_lime_feat_sel_params(feat_sel_params, params_str);
  log_info("Deserialize the lime feature_selection params");

  LimeExplainer lime_explainer;
  lime_explainer.select_features(party,
                                 feat_sel_params.selected_samples_file,
                                 feat_sel_params.selected_predictions_file,
                                 feat_sel_params.sample_weights_file,
                                 feat_sel_params.num_samples,
                                 feat_sel_params.class_num,
                                 feat_sel_params.class_id,
                                 feat_sel_params.feature_selection,
                                 feat_sel_params.num_explained_features,
                                 feat_sel_params.selected_features_file);
  log_info("Finish feature selection");
}

void lime_interpret(Party party, const std::string& params_str,
                    const std::string& ps_network_str,
                    int is_distributed,
                    int distributed_role,
                    int worker_id) {
  log_info("Begin to interpret using lime");
  // 1. deserialize the LimeInterpretParams
  LimeInterpretParams interpret_params;
  // set the values for local debug
  std::string path_prefix = "/opt/falcon/exps/breast_cancer/client" + std::to_string(party.party_id);
//  interpret_params.selected_data_file = path_prefix + "/log_reg/selected_sampled_data.txt";
//  interpret_params.selected_predictions_file = path_prefix + "/log_reg/selected_predictions.txt";
//  interpret_params.sample_weights_file = path_prefix + "/log_reg/sample_weights.txt";
//  interpret_params.num_samples = 1000;
//  interpret_params.class_num = 2;
//  interpret_params.class_id = 0;
//  interpret_params.interpret_model_name = "linear_regression";
//  interpret_params.interpret_model_param = "reserved";
//  interpret_params.explanation_report = path_prefix + "/log_reg/exp_report.txt";

  std::string interpret_params_str = base64_decode_to_pb_string(params_str);
  deserialize_lime_interpret_params(interpret_params, interpret_params_str);
  interpret_params.selected_data_file = path_prefix + interpret_params.selected_data_file;
  interpret_params.selected_predictions_file = path_prefix + interpret_params.selected_predictions_file;
  interpret_params.sample_weights_file = path_prefix + interpret_params.sample_weights_file;
  interpret_params.explanation_report = path_prefix + interpret_params.explanation_report;
  log_info("Deserialize the lime interpret params");
  log_info("[lime_interpret]: interpret_params[class_id] = " + std::to_string(interpret_params.class_id));

  // 2. create the LimeExplainer instance and call explain instance
  LimeExplainer lime_explainer;
  std::vector<double> explanation = lime_explainer.interpret(party,
                           interpret_params.selected_data_file,
                           interpret_params.selected_predictions_file,
                           interpret_params.sample_weights_file,
                           interpret_params.num_samples,
                           interpret_params.class_num,
                           interpret_params.class_id,
                           interpret_params.interpret_model_name,
                           interpret_params.interpret_model_param,
                           interpret_params.explanation_report,
                           ps_network_str,
                           is_distributed,
                           distributed_role,
                           worker_id);

  // 3. print the explanation
  log_info("The explanation for class" + std::to_string(interpret_params.class_id) + " is: ");
  for (int i = 0; i < explanation.size(); i++) {
    log_info("explanation " + std::to_string(i) + " = " + std::to_string(explanation[i]));
  }
}

void spdz_lime_computation(int party_num,
                           int party_id,
                           std::vector<int> mpc_port_bases,
                           std::vector<std::string> party_host_names,
                           int public_value_size,
                           const std::vector<int>& public_values,
                           int private_value_size,
                           const std::vector<double>& private_values,
                           falcon::SpdzLimeCompType lime_comp_type,
                           std::promise<std::vector<double>> *res) {
  // Here put the whole setup socket code together, as using a function call
  // would result in a problem when deleting the created sockets
  // setup connections from this party to each spdz party socket
  std::vector<ssl_socket*> mpc_sockets(party_num);
  vector<int> plain_sockets(party_num);
  // ssl_ctx ctx(mpc_player_path, "C" + to_string(party_id));
  ssl_ctx ctx("C" + to_string(party_id));
  // std::cout << "correct init ctx" << std::endl;
  ssl_service io_service;
  octetStream specification;
  std::cout << "begin connect to spdz parties" << std::endl;
  std::cout << "party_num = " << party_num << std::endl;
  for (int i = 0; i < party_num; i++)
  {
    set_up_client_socket(plain_sockets[i], party_host_names[i].c_str(), mpc_port_bases[i] + i);
    send(plain_sockets[i], (octet*) &party_id, sizeof(int));
    mpc_sockets[i] = new ssl_socket(io_service, ctx, plain_sockets[i],
                                    "P" + to_string(i), "C" + to_string(party_id), true);
    if (i == 0){
      // receive gfp prime
      specification.Receive(mpc_sockets[0]);
    }
    LOG(INFO) << "Set up socket connections for " << i << "-th spdz party succeed,"
                                                          " sockets = " << mpc_sockets[i] << ", port_num = " << mpc_port_bases[i] + i << ".";
  }
  LOG(INFO) << "Finish setup socket connections to spdz engines.";
  std::cout << "Finish setup socket connections to spdz engines." << std::endl;
  int type = specification.get<int>();
  switch (type)
  {
    case 'p':
    {
      gfp::init_field(specification.get<bigint>());
      LOG(INFO) << "Using prime " << gfp::pr();
      break;
    }
    default:
      LOG(ERROR) << "Type " << type << " not implemented";
      exit(1);
  }
  LOG(INFO) << "Finish initializing gfp field.";
  // std::cout << "Finish initializing gfp field." << std::endl;
  // std::cout << "batch aggregation size = " << batch_aggregation_shares.size() << std::endl;
  google::FlushLogFiles(google::INFO);

  // send data to spdz parties
  std::cout << "party_id = " << party_id << std::endl;
  LOG(INFO) << "party_id = " << party_id;
  if (party_id == ACTIVE_PARTY_ID) {
    // the active party sends computation id for spdz computation
    std::vector<int> computation_id;
    computation_id.push_back(lime_comp_type);
    std::cout << "lime_comp_type = " << lime_comp_type << std::endl;
    LOG(INFO) << "lime_comp_type = " << lime_comp_type;
    google::FlushLogFiles(google::INFO);
    send_public_values(computation_id, mpc_sockets, party_num);
    // the active party sends public values to spdz parties
    for (int i = 0; i < public_value_size; i++) {
      std::vector<int> x;
      x.push_back(public_values[i]);
      send_public_values(x, mpc_sockets, party_num);
    }
  }
  // all the parties send private shares
  std::cout << "private value size = " << private_value_size << std::endl;
  LOG(INFO) << "private value size = " << private_value_size;
  for (int i = 0; i < private_value_size; i++) {
    vector<double> x;
    x.push_back(private_values[i]);
    send_private_inputs(x, mpc_sockets, party_num);
  }

  // receive result from spdz parties according to the computation type
  switch (lime_comp_type) {
    case falcon::DIST_WEIGHT: {
      LOG(INFO) << "SPDZ lime computation dist weights computation";
      std::vector<double> return_values = receive_result(mpc_sockets, party_num, private_value_size);
      res->set_value(return_values);
      break;
    }
    case falcon::PEARSON: {
      LOG(INFO) << "SPDZ lime computation type find best split returned";
      std::vector<double> return_values;
      std::vector<double> best_split_index = receive_result(mpc_sockets, party_num, 1);
      std::vector<double> best_left_impurity = receive_result(mpc_sockets, party_num, 1);
      std::vector<double> best_right_impurity = receive_result(mpc_sockets, party_num, 1);
      return_values.push_back(best_split_index[0]);
      return_values.push_back(best_left_impurity[0]);
      return_values.push_back(best_right_impurity[0]);
      res->set_value(return_values);
      break;
    }
    default:
      LOG(INFO) << "SPDZ lime computation type is not found.";
      exit(1);
  }

  for (int i = 0; i < party_num; i++) {
    close_client_socket(plain_sockets[i]);
  }

  // free memory and close mpc_sockets
  for (int i = 0; i < party_num; i++) {
    delete mpc_sockets[i];
    mpc_sockets[i] = nullptr;
  }
}