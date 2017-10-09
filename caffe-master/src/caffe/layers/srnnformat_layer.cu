#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/srnnformat_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void format_gpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int height, int width, int N, bool gen_indicator, Dtype* indicator) {
  CUDA_KERNEL_LOOP(index, num * channels * height * width) {
	  int n = index / (channels * height * width);
	  int c = (index % (channels * height * width)) / (height * width);
	  int h = (index % (height * width)) / width;
	  int w = index % width;
	  
	  int top_index, bottom_index, T_idx = w, N_idx = n * height + h;
	  if (gen_indicator && c == 0) {
		indicator [T_idx * N + N_idx] = int (T_idx != 0);
	  }
	  bottom_index = n * channels * height * width + c * height * width + h * width + w;
	  top_index = T_idx * N * channels + N_idx * channels + c;
	  output[top_index] = input[bottom_index];
		
  }
}

template<typename Dtype>
__global__ void unformat_gpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int height, int width, int N) {
  CUDA_KERNEL_LOOP(index, num * channels * height * width) {
	  int n = index / (channels * height * width);
	  int c = (index % (channels * height * width)) / (height * width);
	  int h = (index % (height * width)) / width;
	  int w = index % width;
	  
	  int top_index, bottom_index, T_idx = w, N_idx = n * height + h;
	  
	  top_index = n * channels * height * width + c * height * width + h * width + w;
	  bottom_index = T_idx * N * channels + N_idx * channels + c;
	  output[top_index] = input[bottom_index];
  }
}

template<typename Dtype>
void SRNNFormatLayer< Dtype >::Forward_gpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  const int count = bottom[0]->count();
  if ( operation_ == SRNNFormatParameter_Operation_FORMAT ) {
    /* NOLINT_NEXT_LINE(whitespace/operators) */
    format_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	         bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), num_, channels_,
             height_, width_, N_, true, top[1]->mutable_gpu_data());
  } else {
    /* NOLINT_NEXT_LINE(whitespace/operators) */
    unformat_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	         bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), num_, channels_,
             height_, width_, N_);
  }
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void SRNNFormatLayer< Dtype >::Backward_gpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  if ( operation_ == SRNNFormatParameter_Operation_UNFORMAT ) {
    if (propagate_down[1])
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to the stream size input.";
	Dtype* nullindicator = NULL;
	/* NOLINT_NEXT_LINE(whitespace/operators) */
    format_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	         top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), num_, channels_,
			 height_, width_, N_, false, nullindicator);
  } else {
    /* NOLINT_NEXT_LINE(whitespace/operators) */
    unformat_gpu<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	         top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), num_, channels_,
             height_, width_, N_);
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SRNNFormatLayer);

}  // namespace caffe