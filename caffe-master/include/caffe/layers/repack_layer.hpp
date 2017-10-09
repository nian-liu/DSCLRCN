#ifndef CAFFE_REPACK_LAYER_HPP_
#define CAFFE_REPACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Rearranges the input blobs, by samping every n-th pixel, and 
 *        stacking them along the 'num' dimension.
 *
 * Use this layer if you want to compute use fully convolutional network and
 * overlapping pooling regions. For example change a pooling region of
 * stride:2 to stride:1 and add a repack layer with stride:2. Then add an
 * unpack layer to the end of the network. Always repack the topmost pooling
 * operation first, then lower ones. This allows you to run the VGG network with
 * a 4 x 4 top level stride, in only 2.5-3 times the computational time.
*/
template <typename Dtype>
class RepackLayer : public Layer<Dtype> {
 public:
  explicit RepackLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Repack"; }
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

  int stride_w_, stride_h_;
  RepackParameter_Operation operation_;
};

}  // namespace caffe

#endif  // CAFFE_REPACK_LAYER_HPP_
