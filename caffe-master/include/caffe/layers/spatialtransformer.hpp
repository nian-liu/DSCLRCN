#ifndef CAFFE_SPATIALTRANSFORMER_LAYER_HPP_
#define CAFFE_SPATIALTRANSFORMER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Transform spatial feature maps (with shape num*channel*height*width) 
 *        with rotation (clockwise) with an integer multiple of 90 degrees and 
 *        mirror (horizontally) (first, mirror, then rotate)
 *
 *Untransform does the reverse operation; 
 */
template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
 public:
  explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
	  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialTransformer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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
  bool mirror_;
  SpatialTransformerParameter_Rotate rotate_;
  SpatialTransformerParameter_Operation operation_;
};

}  // namespace caffe

#endif  // CAFFE_SPATIALTRANSFORMER_LAYER_HPP_
