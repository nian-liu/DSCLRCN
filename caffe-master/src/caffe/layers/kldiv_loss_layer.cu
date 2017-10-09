#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kldiv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KLDivLossForwardGPU(const int n, Dtype* FM, Dtype* SM, const Dtype eps, Dtype* output) {
	CUDA_KERNEL_LOOP(index, n) {
		output[index] = FM [index] * log (FM [index] / (SM [index] + eps) + eps);
	}
}

template <typename Dtype>
__global__ void KLDivLossBackwardGPU(const int n, const int map_size, const Dtype* FM, const Dtype* sum, const Dtype* SM, const Dtype eps, Dtype* output) {
	CUDA_KERNEL_LOOP(index, n) {
	    int num_idx = index / map_size;
	    output [index] = Dtype( -1) * FM [index] * (sum [num_idx] - SM [index]) /
	                       (sum [num_idx] * SM [index] + eps);
	}
}

template <typename Dtype>
void KLDivLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int map_size_ = bottom[0]->height() * bottom[0]->width();
  const int count_ = num_ * map_size_;
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* summed_sm = summed_sm_.mutable_cpu_data();
  Dtype* normalized_fm = normalized_fm_.mutable_gpu_data();
  Dtype* tmp = normalized_fm_.mutable_gpu_diff();
  Dtype summed_fm;
  Dtype loss;
  for (int i = 0; i < num_; ++i) {
	caffe_gpu_asum(map_size_, target + i * map_size_, &summed_fm);  //summed_fm
    caffe_gpu_asum(map_size_, input_data + i * map_size_, summed_sm + i);
	caffe_gpu_scale(map_size_, Dtype(1) / (summed_fm + epsilon_), target + i * map_size_, normalized_fm + i * map_size_);  //normalized FM
	caffe_gpu_scale(map_size_, Dtype(1)/(summed_sm [i] + epsilon_), input_data + i * map_size_, tmp + i * map_size_);   //normalized SM
  }
  KLDivLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS >>>(
			count_, normalized_fm, tmp, epsilon_, tmp);
  caffe_gpu_asum(count_, tmp, &loss);  //accumulation
  top[0]->mutable_cpu_data()[0] = loss / num_;
}

template <typename Dtype>
void KLDivLossLayer<Dtype>::Backward_gpu(
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
	const Dtype* summed_sm = summed_sm_.gpu_data();
    const Dtype* normalized_fm = normalized_fm_.gpu_data();
	const Dtype* input_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	
	KLDivLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS >>>(
			count_, map_size_, normalized_fm, summed_sm, input_data, epsilon_, bottom_diff);
	
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count_, loss_weight / num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KLDivLossLayer);

}  // namespace caffe
