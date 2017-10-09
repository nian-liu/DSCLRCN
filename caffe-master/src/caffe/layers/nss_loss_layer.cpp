#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nss_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NSSLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void NSSLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  num_ = bottom[0]->num();
  summed_fm_.Reshape(bottom[1]->num(), 1, 1, 1);
  //sum_multiplier_.Reshape(num_, 1, bottom[0]->height(), bottom[0]->width());
  //Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  //caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void NSSLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int map_size_ = bottom[0]->height() * bottom[0]->width();
  const int count_ = num_ * map_size_;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype* summed_fm = summed_fm_.mutable_cpu_data();
  Dtype* summed_sm = summed_fm_.mutable_cpu_diff();
  Dtype* normalized_fm = bottom[0]->mutable_cpu_diff();
  //Dtype* tmp = bottom[1]->mutable_cpu_diff();
  Dtype loss = 0;
  
  //caffe_cpu_gemv<Dtype>(CblasNoTrans, num_, map_size_, 1., input_data,
  //    sum_multiplier_.cpu_data(), 0., summed_sm);//summed_sm
  for (int i = 0; i < num_; ++i) {
	//CHECK_LE(abs(summed_sm [i]/map_size_), 1e-5)<<'Saliency map must be normalized';
	summed_fm [i] = caffe_cpu_asum(map_size_, target + i * map_size_);
	CHECK_GE(summed_fm [i], 0)<<'Fixation map must be non-zero';
	caffe_cpu_scale(map_size_, Dtype(-1.)/summed_fm [i], target + i * map_size_, normalized_fm + i * map_size_);  //normalized FM
  }
  
  for (int i = 0; i < count_; ++i) {
    loss += input_data[i] * normalized_fm[i];
  }
  top[0]->mutable_cpu_data()[0] = loss / num_;
}

template <typename Dtype>
void NSSLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count_ = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count_, loss_weight / num_, bottom_diff);
  }
}

INSTANTIATE_CLASS(NSSLossLayer);
REGISTER_LAYER_CLASS(NSSLoss);

}  // namespace caffe
