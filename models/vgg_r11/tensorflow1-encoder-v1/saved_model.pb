�
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8�
�
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
�
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
�
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
�
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

0
1
2
3
 
 
�

layers
 layer_metrics
	variables
!layer_regularization_losses
regularization_losses
"non_trainable_variables
trainable_variables
#metrics
 
 
 
 
�

$layers
%layer_metrics
	variables
&layer_regularization_losses
regularization_losses
'non_trainable_variables
trainable_variables
(metrics
 
 
 
�

)layers
*layer_metrics
	variables
+layer_regularization_losses
regularization_losses
,non_trainable_variables
trainable_variables
-metrics
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
�

.layers
/layer_metrics
	variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
trainable_variables
2metrics
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
�

3layers
4layer_metrics
	variables
5layer_regularization_losses
regularization_losses
6non_trainable_variables
trainable_variables
7metrics
#
0
1
2
3
4
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 

0
1
 
�
serving_default_input_2Placeholder*A
_output_shapes/
-:+���������������������������*
dtype0*6
shape-:+���������������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/bias*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_signature_wrapper_5703
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*&
f!R
__inference__traced_save_5840
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/bias*
Tin	
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_restore_5864��
�
�
$__inference_model_layer_call_fn_5688
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_56772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_model_layer_call_and_return_conditional_losses_5677

inputs
block1_conv1_5666
block1_conv1_5668
block1_conv2_5671
block1_conv2_5673
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_55802+
)tf_op_layer_strided_slice/PartitionedCall�
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_55942%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_5666block1_conv1_5668*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_55362&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5671block1_conv2_5673*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_55582&
$block1_conv2/StatefulPartitionedCall�
IdentityIdentity-block1_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_block1_conv2_layer_call_fn_5568

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_55582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
� 
�
__inference__traced_save_5840
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_478d077d0ed443609275eb7d7ffc1216/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: :@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:

_output_shapes
: 
� 
�
?__inference_model_layer_call_and_return_conditional_losses_5751

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource
identity��
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tf_op_layer_strided_slice/strided_slice/begin�
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2-
+tf_op_layer_strided_slice/strided_slice/end�
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   ����21
/tf_op_layer_strided_slice/strided_slice/strides�
'tf_op_layer_strided_slice/strided_sliceStridedSliceinputs6tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������*

begin_mask*
ellipsis_mask*
end_mask2)
'tf_op_layer_strided_slice/strided_slice�
 tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"����َ��)\��2"
 tf_op_layer_BiasAdd/BiasAdd/bias�
tf_op_layer_BiasAdd/BiasAddBiasAdd0tf_op_layer_strided_slice/strided_slice:output:0)tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_BiasAdd/BiasAdd�
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp�
block1_conv1/Conv2DConv2D$tf_op_layer_BiasAdd/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
block1_conv1/Conv2D�
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp�
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv1/BiasAdd�
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv1/Relu�
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp�
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
block1_conv2/Conv2D�
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp�
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv2/BiasAdd�
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv2/Relu�
IdentityIdentityblock1_conv2/Relu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_model_layer_call_and_return_conditional_losses_5648

inputs
block1_conv1_5637
block1_conv1_5639
block1_conv2_5642
block1_conv2_5644
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_55802+
)tf_op_layer_strided_slice/PartitionedCall�
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_55942%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_5637block1_conv1_5639*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_55362&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5642block1_conv2_5644*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_55582&
$block1_conv2/StatefulPartitionedCall�
IdentityIdentity-block1_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_model_layer_call_and_return_conditional_losses_5613
input_2
block1_conv1_5602
block1_conv1_5604
block1_conv2_5607
block1_conv2_5609
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_55802+
)tf_op_layer_strided_slice/PartitionedCall�
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_55942%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_5602block1_conv1_5604*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_55362&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5607block1_conv2_5609*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_55582&
$block1_conv2/StatefulPartitionedCall�
IdentityIdentity-block1_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5536

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_5580

inputs
identity{
strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/beginw
strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/end
strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   ����2
strided_slice/strides�
strided_sliceStridedSliceinputsstrided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice�
IdentityIdentitystrided_slice:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_5796

inputs
identityq
BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"����َ��)\��2
BiasAdd/bias�
BiasAddBiasAddinputsBiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_5777

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_56772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�"
�
__inference__wrapped_model_5524
input_25
1model_block1_conv1_conv2d_readvariableop_resource6
2model_block1_conv1_biasadd_readvariableop_resource5
1model_block1_conv2_conv2d_readvariableop_resource6
2model_block1_conv2_biasadd_readvariableop_resource
identity��
3model/tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/tf_op_layer_strided_slice/strided_slice/begin�
1model/tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        23
1model/tf_op_layer_strided_slice/strided_slice/end�
5model/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   ����27
5model/tf_op_layer_strided_slice/strided_slice/strides�
-model/tf_op_layer_strided_slice/strided_sliceStridedSliceinput_2<model/tf_op_layer_strided_slice/strided_slice/begin:output:0:model/tf_op_layer_strided_slice/strided_slice/end:output:0>model/tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������*

begin_mask*
ellipsis_mask*
end_mask2/
-model/tf_op_layer_strided_slice/strided_slice�
&model/tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"����َ��)\��2(
&model/tf_op_layer_BiasAdd/BiasAdd/bias�
!model/tf_op_layer_BiasAdd/BiasAddBiasAdd6model/tf_op_layer_strided_slice/strided_slice:output:0/model/tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2#
!model/tf_op_layer_BiasAdd/BiasAdd�
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model/block1_conv1/Conv2D/ReadVariableOp�
model/block1_conv1/Conv2DConv2D*model/tf_op_layer_BiasAdd/BiasAdd:output:00model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
model/block1_conv1/Conv2D�
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv1/BiasAdd/ReadVariableOp�
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
model/block1_conv1/BiasAdd�
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
model/block1_conv1/Relu�
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(model/block1_conv2/Conv2D/ReadVariableOp�
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
model/block1_conv2/Conv2D�
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv2/BiasAdd/ReadVariableOp�
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
model/block1_conv2/BiasAdd�
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
model/block1_conv2/Relu�
IdentityIdentity%model/block1_conv2/Relu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
N
2__inference_tf_op_layer_BiasAdd_layer_call_fn_5801

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_55942
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
?__inference_model_layer_call_and_return_conditional_losses_5629
input_2
block1_conv1_5618
block1_conv1_5620
block1_conv2_5623
block1_conv2_5625
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_55802+
)tf_op_layer_strided_slice/PartitionedCall�
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_55942%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_5618block1_conv1_5620*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_55362&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5623block1_conv2_5625*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_55582&
$block1_conv2/StatefulPartitionedCall�
IdentityIdentity-block1_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
 __inference__traced_restore_5864
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5558

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_block1_conv1_layer_call_fn_5546

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_55362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
"__inference_signature_wrapper_5703
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__wrapped_model_55242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_model_layer_call_fn_5764

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_56482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_5594

inputs
identityq
BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"����َ��)\��2
BiasAdd/bias�
BiasAddBiasAddinputsBiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
� 
�
?__inference_model_layer_call_and_return_conditional_losses_5727

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource
identity��
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tf_op_layer_strided_slice/strided_slice/begin�
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2-
+tf_op_layer_strided_slice/strided_slice/end�
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   ����21
/tf_op_layer_strided_slice/strided_slice/strides�
'tf_op_layer_strided_slice/strided_sliceStridedSliceinputs6tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������*

begin_mask*
ellipsis_mask*
end_mask2)
'tf_op_layer_strided_slice/strided_slice�
 tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"����َ��)\��2"
 tf_op_layer_BiasAdd/BiasAdd/bias�
tf_op_layer_BiasAdd/BiasAddBiasAdd0tf_op_layer_strided_slice/strided_slice:output:0)tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_BiasAdd/BiasAdd�
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp�
block1_conv1/Conv2DConv2D$tf_op_layer_BiasAdd/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
block1_conv1/Conv2D�
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp�
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv1/BiasAdd�
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv1/Relu�
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp�
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
block1_conv2/Conv2D�
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp�
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv2/BiasAdd�
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
block1_conv2/Relu�
IdentityIdentityblock1_conv2/Relu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_5785

inputs
identity{
strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/beginw
strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/end
strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   ����2
strided_slice/strides�
strided_sliceStridedSliceinputsstrided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice�
IdentityIdentitystrided_slice:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_5659
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_56482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
T
8__inference_tf_op_layer_strided_slice_layer_call_fn_5790

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_55802
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
input_2J
serving_default_input_2:0+���������������������������Z
block1_conv2J
StatefulPartitionedCall:0+���������������������������@tensorflow/serving/predict:��
�2
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
*8&call_and_return_all_conditional_losses
9_default_save_signature
:__call__"�0
_tf_keras_model�0{"class_name": "Model", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block1_conv2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block1_conv2", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�
	variables
regularization_losses
trainable_variables
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}}
�
	variables
regularization_losses
trainable_variables
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BiasAdd", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
�	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

layers
 layer_metrics
	variables
!layer_regularization_losses
regularization_losses
"non_trainable_variables
trainable_variables
#metrics
:__call__
9_default_save_signature
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

$layers
%layer_metrics
	variables
&layer_regularization_losses
regularization_losses
'non_trainable_variables
trainable_variables
(metrics
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

)layers
*layer_metrics
	variables
+layer_regularization_losses
regularization_losses
,non_trainable_variables
trainable_variables
-metrics
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

.layers
/layer_metrics
	variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
trainable_variables
2metrics
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

3layers
4layer_metrics
	variables
5layer_regularization_losses
regularization_losses
6non_trainable_variables
trainable_variables
7metrics
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
?__inference_model_layer_call_and_return_conditional_losses_5613
?__inference_model_layer_call_and_return_conditional_losses_5751
?__inference_model_layer_call_and_return_conditional_losses_5727
?__inference_model_layer_call_and_return_conditional_losses_5629�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_5524�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�8
input_2+���������������������������
�2�
$__inference_model_layer_call_fn_5659
$__inference_model_layer_call_fn_5777
$__inference_model_layer_call_fn_5764
$__inference_model_layer_call_fn_5688�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_5785�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_tf_op_layer_strided_slice_layer_call_fn_5790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_5796�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_tf_op_layer_BiasAdd_layer_call_fn_5801�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
+__inference_block1_conv1_layer_call_fn_5546�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5558�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
+__inference_block1_conv2_layer_call_fn_5568�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
1B/
"__inference_signature_wrapper_5703input_2�
__inference__wrapped_model_5524�J�G
@�=
;�8
input_2+���������������������������
� "U�R
P
block1_conv2@�=
block1_conv2+���������������������������@�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5536�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������@
� �
+__inference_block1_conv1_layer_call_fn_5546�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������@�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5558�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
+__inference_block1_conv2_layer_call_fn_5568�I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
?__inference_model_layer_call_and_return_conditional_losses_5613�R�O
H�E
;�8
input_2+���������������������������
p

 
� "?�<
5�2
0+���������������������������@
� �
?__inference_model_layer_call_and_return_conditional_losses_5629�R�O
H�E
;�8
input_2+���������������������������
p 

 
� "?�<
5�2
0+���������������������������@
� �
?__inference_model_layer_call_and_return_conditional_losses_5727�Q�N
G�D
:�7
inputs+���������������������������
p

 
� "?�<
5�2
0+���������������������������@
� �
?__inference_model_layer_call_and_return_conditional_losses_5751�Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "?�<
5�2
0+���������������������������@
� �
$__inference_model_layer_call_fn_5659�R�O
H�E
;�8
input_2+���������������������������
p

 
� "2�/+���������������������������@�
$__inference_model_layer_call_fn_5688�R�O
H�E
;�8
input_2+���������������������������
p 

 
� "2�/+���������������������������@�
$__inference_model_layer_call_fn_5764�Q�N
G�D
:�7
inputs+���������������������������
p

 
� "2�/+���������������������������@�
$__inference_model_layer_call_fn_5777�Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "2�/+���������������������������@�
"__inference_signature_wrapper_5703�U�R
� 
K�H
F
input_2;�8
input_2+���������������������������"U�R
P
block1_conv2@�=
block1_conv2+���������������������������@�
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_5796�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
2__inference_tf_op_layer_BiasAdd_layer_call_fn_5801I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_5785�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
8__inference_tf_op_layer_strided_slice_layer_call_fn_5790I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������