#!/usr/bin/env sh
CAFFE_DIR=/home/nianliu/Deeplearning_codes/caffe/caffe-master
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN5_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN7_SLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN7_DSLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN7_DSCLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel,$CAFFE_DIR/models/placesCNN/places205CNN_s.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_solver.prototxt --weights ../caffe2/models/FCN6_2_DSLSTM/EF_FCN6_DSLSTM_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_multilayer_solver.prototxt --weights ./models/FCN_conv6_1_ker5/EF_FCN_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_SLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_SLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_DSLSTM_MIT_finetune_solver.prototxt --weights ./models/FCN_SLSTM/SALICON/EF_FCN_DSLSTM_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_attention_DSLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_DSLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_SLSTM_MIT_finetune_solver.prototxt --weights ./models/FCN6_SLSTM/SALICON/SALICON_EF_SLSTM_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_DSLSTM_MIT_finetune_solver.prototxt --weights ./models/FCN_SLSTM/SALICON/480DSLSTM/EF_FCN6_DSLSTM_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN6_DSCLSTM_solver.prototxt --weights ./models/FCN_SLSTM/SALICON/EF_FCN6_DSLSTM_iter_5000.caffemodel,$CAFFE_DIR/models/placesCNN/places205CNN_iter_300000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_ResNet50_DSCLSTM_solver.prototxt --weights $CAFFE_DIR/models/ResNet/ResNet-50-model.caffemodel,$CAFFE_DIR/models/placesCNN/places205CNN_s.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_ResNet50_DSCLSTM_MIT_finetune_solver.prototxt --weights ./models/ResNet50_DSCLSTM/EF_ResNet50_DSCLSTM_iter_3000.caffemodel
$CAFFE_DIR/build/tools/caffe train --solver=EF_ResNet50_ML_DSCLSTM_solver.prototxt --weights $CAFFE_DIR/models/ResNet/ResNet-50-model.caffemodel,$CAFFE_DIR/models/placesCNN/places205CNN_s.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_ResNet50_ML_DSCLSTM_MIT_finetune_solver.prototxt --weights ./models/ResNet50_ML_DSCLSTM/EF_ResNet50_ML_DSCLSTM_iter_3000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN7_DSLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN7_DSLSTM_MIT_finetune_solver.prototxt --weights ./models/FCN_SLSTM/SALICON/480DSLSTM/EF_FCN7_DSLSTM_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_MIT_finetune_solver.prototxt --weights ./models/FCN_conv6_1_ker5/SALICON/EF_FCN_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_multilayer_MIT_finetune_solver.prototxt --weights ./models/FCN_multilayer/SALICON/EF_FCN_multilayer_iter_5000.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_ML_DSLSTM_solver.prototxt --weights $CAFFE_DIR/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel
#$CAFFE_DIR/build/tools/caffe train --solver=EF_FCN_ML_DSLSTM_MIT_finetune_solver.prototxt --weights ./models/FCN_multilayer_SLSTM/SALICON/EF_FCN_ML_DSLSTM_iter_5000.caffemodel
