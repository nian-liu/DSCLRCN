#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kldiv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLDivLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void KLDivLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  num_ = bottom[0]->num();
  summed_sm_.Reshape(bottom[0]->num(), 1, 1, 1);
  normalized_fm_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void KLDivLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int map_size_ = bottom[0]->height() * bottom[0]->width();
  const int count_ = num_ * map_size_;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype* summed_sm = summed_sm_.mutable_cpu_data();
  Dtype* normalized_fm = normalized_fm_.mutable_cpu_data();
  Dtype* tmp = normalized_fm_.mutable_cpu_diff();
  Dtype summed_fm;
  Dtype loss = 0;
  for (int i = 0; i < num_; ++i) {
    summed_sm [i] = caffe_cpu_asum(map_size_, input_data + i * map_size_);
	summed_fm = caffe_cpu_asum(map_size_, target + i * map_size_);
	caffe_cpu_scale(map_size_, Dtype(1)/(summed_fm + epsilon_), target + i * map_size_, normalized_fm + i * map_size_);  //normalized FM
	caffe_cpu_scale(map_size_, Dtype(1)/(summed_sm [i] + epsilon_), input_data + i * map_size_, tmp + i * map_size_);   //normalized SM
  }
  for (int i = 0; i < count_; ++i) {
    loss += normalized_fm [i] * log (normalized_fm [i] / (tmp [i] + epsilon_) + epsilon_);
  }
  /*
  caffe_add_scalar(count_, epsilon_, tmp); //SM(x)+eps
  caffe_div(count_, normalized_fm, tmp, tmp);  //FM(x)/(SM(x)+eps)
  caffe_add_scalar(count_, epsilon_, tmp); //FM(x)/(SM(x)+eps)+eps
  caffe_log(count_, tmp, tmp);  //log(FM(x)/(SM(x)+eps)+eps)
  caffe_mul(count_, normalized_fm, tmp, tmp);  //FM(x) * log(FM(x)/(SM(x)+eps)+eps)
  loss = caffe_cpu_asum(count_, tmp);  //accumulation
  */
  top[0]->mutable_cpu_data()[0] = loss / num_;
}

template <typename Dtype>
void KLDivLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff = -FM(x)*(summed_fm_-sm(X))/(summed_fm_*sm(X))
	const int map_size_ = bottom[0]->height() * bottom[0]->width();
    const int count_ = num_ * map_size_;
	const Dtype* summed_sm = summed_sm_.cpu_data();
    const Dtype* normalized_fm = normalized_fm_.cpu_data();
	const Dtype* input_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int num_idx;
	for (int i = 0; i < count_; ++i) {
	  num_idx = i / map_size_;
	  bottom_diff [i] = Dtype( -1) * normalized_fm [i] * (summed_sm [num_idx] - input_data [i]) /
	                       (summed_sm [num_idx] * input_data [i] + epsilon_);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count_, loss_weight / num_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(KLDivLossLayer);
#endif

INSTANTIATE_CLASS(KLDivLossLayer);
REGISTER_LAYER_CLASS(KLDivLoss);

}  // namespace caffe
