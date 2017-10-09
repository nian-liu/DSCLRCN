#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/srnnformat_layer.hpp"

namespace caffe {
template<typename Dtype>
void SRNNFormatLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                      const vector< Blob< Dtype >* >& top) {
  SRNNFormatParameter srnnformat_param = this->layer_param_.srnnformat_param();

  operation_ = srnnformat_param.operation();
}

template<typename Dtype>
void SRNNFormatLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                   const vector< Blob< Dtype >* >& top) {
  if ( operation_ == SRNNFormatParameter_Operation_FORMAT ) {
    CHECK_EQ(1, bottom.size()) << "Only one input blob is needed!";
	CHECK_EQ(3, top.size()) << "Three output blobs are requested!";
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
        << "corresponding to (num, channels, height, width)";
	num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
	T_ = width_;
	N_ = num_ * height_;
	
	vector<int> RNNDataShape(3);
	RNNDataShape[0] = T_;
	RNNDataShape[1] = N_;
	RNNDataShape[2] = channels_;
	top[0]->Reshape(RNNDataShape);
	
	RNNDataShape.pop_back();
	top[1]->Reshape(RNNDataShape);
	
	vector<int> StreamSizeShape(0); //scalar, how many streams in an image (equal to height_)
	top[2]->Reshape(StreamSizeShape); //stream size
	top[2]->mutable_cpu_data()[0] = height_;
  } else if ( operation_ == SRNNFormatParameter_Operation_UNFORMAT ) {
    CHECK_EQ(2, bottom.size()) << "Two input blobs are needed!";
	CHECK_EQ(1, top.size()) << "Only one output blob is requested!";
    CHECK_EQ(3, bottom[0]->num_axes()) << "Input must have 3 axes, "
        << "corresponding to (timesteps, number of streams, channels)";
	CHECK_EQ(0, bottom[1]->num_axes()) << "Input must have 0 axes, "
        << "namely, it is a scalar";
	T_ = bottom[0]->shape(0);
    N_ = bottom[0]->shape(1);
	channels_ = bottom[0]->shape(2);
	height_ = bottom[1]->cpu_data()[0];
	width_ = T_;
	num_ = N_ / height_;
	top[0]->Reshape(num_, channels_, height_, width_);
  } else { LOG(ERROR) << "Unknown SRNNFormat operation!"; }
}

template<typename Dtype>
inline void format_cpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int height, int width, int N, bool gen_indicator, Dtype* indicator) {
  int top_index, bottom_index, T_idx, N_idx;
  for (int w = 0; w < width; ++w) {
    T_idx = w;
    for (int n = 0; n < num; ++n) {
      for (int h = 0; h < height; ++h) {
	    N_idx = n * height + h;
		if (gen_indicator) {
		  indicator [T_idx * N + N_idx] = int (T_idx != 0);
		}
	    for (int c = 0; c < channels; ++c){
	      bottom_index = n * channels * height * width + c * height * width + h * width + w;
		  top_index = T_idx * N * channels + N_idx * channels + c;
		  output[top_index] = input[bottom_index];
		}
	  }
	}
  }
}

template<typename Dtype>
inline void unformat_cpu(const Dtype* input, Dtype* output, int num, int channels, 
                      int height, int width, int N) {
  int top_index, bottom_index, T_idx, N_idx;
  for (int w = 0; w < width; ++w) {
    T_idx = w;
    for (int n = 0; n < num; ++n) {
      for (int h = 0; h < height; ++h) {
	    N_idx = n * height + h;
	    for (int c = 0; c < channels; ++c){
	      top_index = n * channels * height * width + c * height * width + h * width + w;
		  bottom_index = T_idx * N * channels + N_idx * channels + c;
		  output[top_index] = input[bottom_index];
		}
	  }
	}
  }
}

template<typename Dtype>
void SRNNFormatLayer< Dtype >::Forward_cpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  if ( operation_ == SRNNFormatParameter_Operation_FORMAT )
    format_cpu(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), num_, channels_,
             height_, width_, N_, true, top[1]->mutable_cpu_data());
  else
    unformat_cpu(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), num_, channels_,
             height_, width_, N_);
}

template<typename Dtype>
void SRNNFormatLayer< Dtype >::Backward_cpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  if ( operation_ == SRNNFormatParameter_Operation_UNFORMAT ) {
    if (propagate_down[1])
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to the stream size input.";
	Dtype* nullindicator = NULL;
    format_cpu(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(), num_, channels_,
		height_, width_, N_, false, nullindicator);
  } else
    unformat_cpu(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(), num_, channels_,
             height_, width_, N_);
}

#ifdef CPU_ONLY
STUB_GPU(SRNNFormatLayer);
#endif
INSTANTIATE_CLASS(SRNNFormatLayer);
REGISTER_LAYER_CLASS(SRNNFormat);

}  // namespace caffe