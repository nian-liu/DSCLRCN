#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/spatialtransformer.hpp"

namespace caffe {
template<typename Dtype>
void SpatialTransformerLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                      const vector< Blob< Dtype >* >& top) {
  SpatialTransformerParameter spatialtransformer_param = this->layer_param_.spatialtransformer_param();

  mirror_ = spatialtransformer_param.mirror();
  rotate_ = spatialtransformer_param.rotate();
  operation_ = spatialtransformer_param.operation();
}

template<typename Dtype>
void SpatialTransformerLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                   const vector< Blob< Dtype >* >& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if ( rotate_ == SpatialTransformerParameter_Rotate_NOROTATE || 
       rotate_ == SpatialTransformerParameter_Rotate_ROTATE180  ) {
	top[0]->ReshapeLike(*bottom[0]);
  } else if ( rotate_ == SpatialTransformerParameter_Rotate_ROTATE90 || 
       rotate_ == SpatialTransformerParameter_Rotate_ROTATE270  ) {
    top[0]->Reshape(num_, channels_, width_, height_);
  } else { LOG(ERROR) << "Unknown rotation style!"; }
}

template<typename Dtype>
inline void transform_cpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int in_h, int in_w, int out_h, int out_w, bool do_mirror, 
					  SpatialTransformerParameter_Rotate rotate, bool undo) {
  int top_index, bottom_index, h_idx, w_idx, temp;
  if (undo) {
    if (rotate == SpatialTransformerParameter_Rotate_ROTATE90) {
	  rotate = SpatialTransformerParameter_Rotate_ROTATE270;
	} else if (rotate == SpatialTransformerParameter_Rotate_ROTATE270) {
	  rotate = SpatialTransformerParameter_Rotate_ROTATE90;
	}
  }
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < in_h; ++h) {
	    for (int w = 0; w < in_w; ++w) {
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
    }
  }
}

template<typename Dtype>
void SpatialTransformerLayer< Dtype >::Forward_cpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  transform_cpu(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), num_, channels_,
           height_, width_, top[0]->height(), top[0]->width(), mirror_, rotate_, 
		   operation_ == SpatialTransformerParameter_Operation_UNTRANSFORM);
}

template<typename Dtype>
void SpatialTransformerLayer< Dtype >::Backward_cpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  transform_cpu(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(), num_, channels_,
           top[0]->height(), top[0]->width(), height_, width_, mirror_, rotate_, 
            operation_ == SpatialTransformerParameter_Operation_TRANSFORM);
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif
INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

}  // namespace caffe