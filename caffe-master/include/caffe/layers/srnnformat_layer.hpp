#ifndef CAFFE_SRNNFORMAT_LAYER_HPP_
#define CAFFE_SRNNFORMAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief When building Spatial RNN layers, format spatial feature maps 
 *        (with shape num*channel*height*width) to the format requested by 
 *        the RNN(LSTM) layers (with shape T*N*..., where the shape is 
 *        width*(num*height)*channel with using one row as one sequence);
 *
 *Unformat does the reverse operation; 
 */
template <typename Dtype>
class SRNNFormatLayer : public Layer<Dtype> {
 public:
  explicit SRNNFormatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
	  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SRNNFormat"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  
  int num_;
  int channels_;
  int height_;
  int width_;
  int T_;
  int N_;
  SRNNFormatParameter_Operation operation_;
};

}  // namespace caffe

#endif  // CAFFE_SRNNFORMAT_LAYER_HPP_
