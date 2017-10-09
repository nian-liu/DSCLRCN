#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/spatialtransformer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void transform_gpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int in_h, int in_w, int out_h, int out_w, bool do_mirror, 
					  SpatialTransformerParameter_Rotate rotate, bool undo) {
	if (undo) {
		if (rotate == SpatialTransformerParameter_Rotate_ROTATE90) {
			rotate = SpatialTransformerParameter_Rotate_ROTATE270;
		}
		else if (rotate == SpatialTransformerParameter_Rotate_ROTATE270) {
			rotate = SpatialTransformerParameter_Rotate_ROTATE90;
		}
	}
    CUDA_KERNEL_LOOP(index, num * channels * in_h * in_w) {
	  int n = index / (channels * in_h * in_w);
	  int c = (index % (channels * in_h * in_w)) / (in_h * in_w);
	  int h = (index % (in_h * in_w)) / in_w;
	  int w = index % in_w;
	  int top_index, bottom_index, h_idx, w_idx, temp;
	  
	  bottom_index = n * channels * in_h * in_w + c * in_h * in_w + h * in_w + w;
	  h_idx = h;
	  w_idx = w;
	  if (!undo && do_mirror) {
	   w_idx = in_w - 1 - w;
	  }
	  if (rotate == SpatialTransformerParameter_Rotate_ROTATE90) {
	   temp = w_idx;
	   w_idx = in_h - 1 - h_idx;
	   h_idx = temp;
	  } else if (rotate == SpatialTransformerParameter_Rotate_ROTATE180) {
	   w_idx = in_w - 1 - w_idx;
	   h_idx = in_h - 1 - h_idx;
	  } else if (rotate == SpatialTransformerParameter_Rotate_ROTATE270) {
	   temp = h_idx;
	   h_idx = in_w - 1 - w_idx;
	   w_idx = temp;
	  }
	  if (undo && do_mirror) {
	   w_idx = out_w - 1 - w_idx;
	  }
	  top_index = n * channels * out_h * out_w + c * out_h * out_w + h_idx * out_w + w_idx;
	  output[top_index] = input[bottom_index];
    }
}

template<typename Dtype>
void SpatialTransformerLayer< Dtype >::Forward_gpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  const int count = bottom[0]->count();
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  transform_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
           bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), num_, channels_,
           height_, width_, top[0]->height(), top[0]->width(), mirror_, rotate_, 
		   operation_ == SpatialTransformerParameter_Operation_UNTRANSFORM);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void SpatialTransformerLayer< Dtype >::Backward_gpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  transform_gpu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		   top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), num_, channels_,
           top[0]->height(), top[0]->width(), height_, width_, mirror_, rotate_, 
            operation_ == SpatialTransformerParameter_Operation_TRANSFORM);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

}  // namespace caffe