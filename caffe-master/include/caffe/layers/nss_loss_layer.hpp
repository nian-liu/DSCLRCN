#ifndef CAFFE_NSS_LOSS_LAYER_HPP_
#define CAFFE_NSS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Normalized Scanpath Saliency (NSS) loss for eye fixation prediction (A MVN layer should be used first to normalize the saliency map)
 */
template <typename Dtype>
class NSSLossLayer : public LossLayer<Dtype> {
 public:
  explicit NSSLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NSSLoss"; }

 protected:
  /// @copydoc NSSLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> summed_fm_;
  /// sum_multiplier is used to carry out sum using BLAS
  //Blob<Dtype> sum_multiplier_;
  int num_;
};

}  // namespace caffe

#endif  // CAFFE_NSS_LOSS_LAYER_HPP_
