��
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
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8�
�
rev_block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_namerev_block2_conv2/kernel
�
+rev_block2_conv2/kernel/Read/ReadVariableOpReadVariableOprev_block2_conv2/kernel*(
_output_shapes
:��*
dtype0
�
rev_block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namerev_block2_conv2/bias
|
)rev_block2_conv2/bias/Read/ReadVariableOpReadVariableOprev_block2_conv2/bias*
_output_shapes	
:�*
dtype0
�
rev_block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*(
shared_namerev_block2_conv1/kernel
�
+rev_block2_conv1/kernel/Read/ReadVariableOpReadVariableOprev_block2_conv1/kernel*'
_output_shapes
:�@*
dtype0
�
rev_block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namerev_block2_conv1/bias
{
)rev_block2_conv1/bias/Read/ReadVariableOpReadVariableOprev_block2_conv1/bias*
_output_shapes
:@*
dtype0
�
rev_block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_namerev_block1_conv2/kernel
�
+rev_block1_conv2/kernel/Read/ReadVariableOpReadVariableOprev_block1_conv2/kernel*&
_output_shapes
:@@*
dtype0
�
rev_block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namerev_block1_conv2/bias
{
)rev_block1_conv2/bias/Read/ReadVariableOpReadVariableOprev_block1_conv2/bias*
_output_shapes
:@*
dtype0
�
rev_block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namerev_block1_conv1/kernel
�
+rev_block1_conv1/kernel/Read/ReadVariableOpReadVariableOprev_block1_conv1/kernel*&
_output_shapes
:@*
dtype0
�
rev_block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namerev_block1_conv1/bias
{
)rev_block1_conv1/bias/Read/ReadVariableOpReadVariableOprev_block1_conv1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�1
value�1B�1 B�1
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
R
<trainable_variables
=regularization_losses
>	variables
?	keras_api
 
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
R
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
h

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
R
\trainable_variables
]regularization_losses
^	variables
_	keras_api
8
0
1
2
3
L4
M5
R6
S7
 
8
0
1
2
3
L4
M5
R6
S7
�
trainable_variables

`layers
alayer_metrics
bmetrics
cnon_trainable_variables
regularization_losses
	variables
dlayer_regularization_losses
 
ca
VARIABLE_VALUErev_block2_conv2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUErev_block2_conv2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables

elayers
flayer_metrics
gmetrics
hnon_trainable_variables
regularization_losses
	variables
ilayer_regularization_losses
ca
VARIABLE_VALUErev_block2_conv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUErev_block2_conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
 trainable_variables

jlayers
klayer_metrics
lmetrics
mnon_trainable_variables
!regularization_losses
"	variables
nlayer_regularization_losses
 
 
 
�
$trainable_variables

olayers
player_metrics
qmetrics
rnon_trainable_variables
%regularization_losses
&	variables
slayer_regularization_losses
 
 
 
�
(trainable_variables

tlayers
ulayer_metrics
vmetrics
wnon_trainable_variables
)regularization_losses
*	variables
xlayer_regularization_losses
 
 
 
�
,trainable_variables

ylayers
zlayer_metrics
{metrics
|non_trainable_variables
-regularization_losses
.	variables
}layer_regularization_losses
 
 
 
�
0trainable_variables

~layers
layer_metrics
�metrics
�non_trainable_variables
1regularization_losses
2	variables
 �layer_regularization_losses
 
 
 
�
4trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
5regularization_losses
6	variables
 �layer_regularization_losses
 
 
 
�
8trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
9regularization_losses
:	variables
 �layer_regularization_losses
 
 
 
�
<trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
=regularization_losses
>	variables
 �layer_regularization_losses
 
 
 
�
@trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Aregularization_losses
B	variables
 �layer_regularization_losses
 
 
 
�
Dtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Eregularization_losses
F	variables
 �layer_regularization_losses
 
 
 
�
Htrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Iregularization_losses
J	variables
 �layer_regularization_losses
ca
VARIABLE_VALUErev_block1_conv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUErev_block1_conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
�
Ntrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Oregularization_losses
P	variables
 �layer_regularization_losses
ca
VARIABLE_VALUErev_block1_conv1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUErev_block1_conv1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
�
Ttrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Uregularization_losses
V	variables
 �layer_regularization_losses
 
 
 
�
Xtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Yregularization_losses
Z	variables
 �layer_regularization_losses
 
 
 
�
\trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
]regularization_losses
^	variables
 �layer_regularization_losses
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
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
�
serving_default_input_3Placeholder*B
_output_shapes0
.:,����������������������������*
dtype0*7
shape.:,����������������������������
z
serving_default_input_4Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4rev_block2_conv2/kernelrev_block2_conv2/biasrev_block2_conv1/kernelrev_block2_conv1/biasrev_block1_conv2/kernelrev_block1_conv2/biasrev_block1_conv1/kernelrev_block1_conv1/bias*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_842051
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+rev_block2_conv2/kernel/Read/ReadVariableOp)rev_block2_conv2/bias/Read/ReadVariableOp+rev_block2_conv1/kernel/Read/ReadVariableOp)rev_block2_conv1/bias/Read/ReadVariableOp+rev_block1_conv2/kernel/Read/ReadVariableOp)rev_block1_conv2/bias/Read/ReadVariableOp+rev_block1_conv1/kernel/Read/ReadVariableOp)rev_block1_conv1/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_842412
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerev_block2_conv2/kernelrev_block2_conv2/biasrev_block2_conv1/kernelrev_block2_conv1/biasrev_block1_conv2/kernelrev_block1_conv2/biasrev_block1_conv1/kernelrev_block1_conv1/bias*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_842448ʺ
�X
�
C__inference_model_1_layer_call_and_return_conditional_losses_842171
inputs_0
inputs_1	3
/rev_block2_conv2_conv2d_readvariableop_resource4
0rev_block2_conv2_biasadd_readvariableop_resource3
/rev_block2_conv1_conv2d_readvariableop_resource4
0rev_block2_conv1_biasadd_readvariableop_resource3
/rev_block1_conv2_conv2d_readvariableop_resource4
0rev_block1_conv2_biasadd_readvariableop_resource3
/rev_block1_conv1_conv2d_readvariableop_resource4
0rev_block1_conv1_biasadd_readvariableop_resource
identity��
&rev_block2_conv2/Conv2D/ReadVariableOpReadVariableOp/rev_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&rev_block2_conv2/Conv2D/ReadVariableOp�
rev_block2_conv2/Conv2DConv2Dinputs_0.rev_block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
rev_block2_conv2/Conv2D�
'rev_block2_conv2/BiasAdd/ReadVariableOpReadVariableOp0rev_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'rev_block2_conv2/BiasAdd/ReadVariableOp�
rev_block2_conv2/BiasAddBiasAdd rev_block2_conv2/Conv2D:output:0/rev_block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
rev_block2_conv2/BiasAdd�
rev_block2_conv2/ReluRelu!rev_block2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
rev_block2_conv2/Relu�
&rev_block2_conv1/Conv2D/ReadVariableOpReadVariableOp/rev_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02(
&rev_block2_conv1/Conv2D/ReadVariableOp�
rev_block2_conv1/Conv2DConv2D#rev_block2_conv2/Relu:activations:0.rev_block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
rev_block2_conv1/Conv2D�
'rev_block2_conv1/BiasAdd/ReadVariableOpReadVariableOp0rev_block2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rev_block2_conv1/BiasAdd/ReadVariableOp�
rev_block2_conv1/BiasAddBiasAdd rev_block2_conv1/Conv2D:output:0/rev_block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block2_conv1/BiasAdd�
rev_block2_conv1/ReluRelu!rev_block2_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block2_conv1/Relu�
tf_op_layer_Shape_2/Shape_2Shape#rev_block2_conv1/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_2/Shape_2�
1tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_6/strided_slice_6/begin�
/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_6/strided_slice_6/end�
3tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_6/strided_slice_6/strides�
+tf_op_layer_strided_slice_6/strided_slice_6StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_6/strided_slice_6/begin:output:08tf_op_layer_strided_slice_6/strided_slice_6/end:output:0<tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_6/strided_slice_6�
1tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_5/strided_slice_5/begin�
/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_5/strided_slice_5/end�
3tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_5/strided_slice_5/strides�
+tf_op_layer_strided_slice_5/strided_slice_5StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_5/strided_slice_5/begin:output:08tf_op_layer_strided_slice_5/strided_slice_5/end:output:0<tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_5/strided_slice_5�
1tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_4/strided_slice_4/begin�
/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_4/strided_slice_4/end�
3tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_4/strided_slice_4/strides�
+tf_op_layer_strided_slice_4/strided_slice_4StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_4/strided_slice_4/begin:output:08tf_op_layer_strided_slice_4/strided_slice_4/end:output:0<tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_4/strided_slice_4x
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Mul_1/Mul_1/y�
tf_op_layer_Mul_1/Mul_1Mul4tf_op_layer_strided_slice_5/strided_slice_5:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mul_1/Mul_1x
tf_op_layer_Mul_2/Mul_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Mul_2/Mul_2/y�
tf_op_layer_Mul_2/Mul_2Mul4tf_op_layer_strided_slice_6/strided_slice_6:output:0"tf_op_layer_Mul_2/Mul_2/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mul_2/Mul_2�
+tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_2/Prod_2/reduction_indices�
tf_op_layer_Prod_2/Prod_2Prod$tf_op_layer_Shape_2/Shape_2:output:04tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_2/Prod_2�
tf_op_layer_Reshape_1/Reshape_1Reshape#rev_block2_conv1/Relu:activations:0"tf_op_layer_Prod_2/Prod_2:output:0*
T0*
Tshape0	*
_cloned(*#
_output_shapes
:���������2!
tf_op_layer_Reshape_1/Reshape_1�
-tf_op_layer_ScatterNd/shape/ScatterNd/shape/3Const*
_output_shapes
: *
dtype0	*
value	B	 R@2/
-tf_op_layer_ScatterNd/shape/ScatterNd/shape/3�
+tf_op_layer_ScatterNd/shape/ScatterNd/shapePack4tf_op_layer_strided_slice_4/strided_slice_4:output:0tf_op_layer_Mul_1/Mul_1:z:0tf_op_layer_Mul_2/Mul_2:z:06tf_op_layer_ScatterNd/shape/ScatterNd/shape/3:output:0*
N*
T0	*
_cloned(*
_output_shapes
:2-
+tf_op_layer_ScatterNd/shape/ScatterNd/shape�
tf_op_layer_ScatterNd/ScatterNd	ScatterNdinputs_1(tf_op_layer_Reshape_1/Reshape_1:output:04tf_op_layer_ScatterNd/shape/ScatterNd/shape:output:0*
T0*
Tindices0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2!
tf_op_layer_ScatterNd/ScatterNd�
&rev_block1_conv2/Conv2D/ReadVariableOpReadVariableOp/rev_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&rev_block1_conv2/Conv2D/ReadVariableOp�
rev_block1_conv2/Conv2DConv2D(tf_op_layer_ScatterNd/ScatterNd:output:0.rev_block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
rev_block1_conv2/Conv2D�
'rev_block1_conv2/BiasAdd/ReadVariableOpReadVariableOp0rev_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rev_block1_conv2/BiasAdd/ReadVariableOp�
rev_block1_conv2/BiasAddBiasAdd rev_block1_conv2/Conv2D:output:0/rev_block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block1_conv2/BiasAdd�
rev_block1_conv2/ReluRelu!rev_block1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block1_conv2/Relu�
&rev_block1_conv1/Conv2D/ReadVariableOpReadVariableOp/rev_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&rev_block1_conv1/Conv2D/ReadVariableOp�
rev_block1_conv1/Conv2DConv2D#rev_block1_conv2/Relu:activations:0.rev_block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
rev_block1_conv1/Conv2D�
'rev_block1_conv1/BiasAdd/ReadVariableOpReadVariableOp0rev_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rev_block1_conv1/BiasAdd/ReadVariableOp�
rev_block1_conv1/BiasAddBiasAdd rev_block1_conv1/Conv2D:output:0/rev_block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2
rev_block1_conv1/BiasAdd�
rev_block1_conv1/ReluRelu!rev_block1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
rev_block1_conv1/Relu�
tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
tf_op_layer_Minimum/Minimum/y�
tf_op_layer_Minimum/MinimumMinimum#rev_block1_conv1/Relu:activations:0&tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_Minimum/Minimum�
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf_op_layer_Maximum/Maximum/y�
tf_op_layer_Maximum/MaximumMaximumtf_op_layer_Minimum/Minimum:z:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_Maximum/Maximum�
IdentityIdentitytf_op_layer_Maximum/Maximum:z:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������:::::::::l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
s
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_842246

inputs	
identity	x
strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/begint
strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/end|
strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/strides�
strided_slice_6StridedSliceinputsstrided_slice_6/begin:output:0strided_slice_6/end:output:0 strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6[
IdentityIdentitystrided_slice_6:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
s
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_842233

inputs	
identity	x
strided_slice_5/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/begint
strided_slice_5/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/end|
strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/strides�
strided_slice_5StridedSliceinputsstrided_slice_5/begin:output:0strided_slice_5/end:output:0 strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5[
IdentityIdentitystrided_slice_5:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
}
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_842303
inputs_0
inputs_1	
identity�
	Reshape_1Reshapeinputs_0inputs_1*
T0*
Tshape0	*
_cloned(*#
_output_shapes
:���������2
	Reshape_1b
IdentityIdentityReshape_1:output:0*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+���������������������������@::k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
N
2__inference_tf_op_layer_Mul_2_layer_call_fn_842297

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8417602
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
P
4__inference_tf_op_layer_Minimum_layer_call_fn_842349

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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_8418482
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
�B
�
C__inference_model_1_layer_call_and_return_conditional_losses_841949

inputs
inputs_1	
rev_block2_conv2_841916
rev_block2_conv2_841918
rev_block2_conv1_841921
rev_block2_conv1_841923
rev_block1_conv2_841936
rev_block1_conv2_841938
rev_block1_conv1_841941
rev_block1_conv1_841943
identity��(rev_block1_conv1/StatefulPartitionedCall�(rev_block1_conv2/StatefulPartitionedCall�(rev_block2_conv1/StatefulPartitionedCall�(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv2/StatefulPartitionedCallStatefulPartitionedCallinputsrev_block2_conv2_841916rev_block2_conv2_841918*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_8415882*
(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block2_conv2/StatefulPartitionedCall:output:0rev_block2_conv1_841921rev_block2_conv1_841923*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_8416102*
(rev_block2_conv1/StatefulPartitionedCall�
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_8416842%
#tf_op_layer_Shape_2/PartitionedCall�
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_8417002-
+tf_op_layer_strided_slice_6/PartitionedCall�
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_8417162-
+tf_op_layer_strided_slice_5/PartitionedCall�
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_8417322-
+tf_op_layer_strided_slice_4/PartitionedCall�
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8417462#
!tf_op_layer_Mul_1/PartitionedCall�
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8417602#
!tf_op_layer_Mul_2/PartitionedCall�
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_8417742$
"tf_op_layer_Prod_2/PartitionedCall�
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2	*
Tout
2*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8417882'
%tf_op_layer_Reshape_1/PartitionedCall�
+tf_op_layer_ScatterNd/shape/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*tf_op_layer_Mul_1/PartitionedCall:output:0*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2			*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_8418052-
+tf_op_layer_ScatterNd/shape/PartitionedCall�
%tf_op_layer_ScatterNd/PartitionedCallPartitionedCallinputs_1.tf_op_layer_Reshape_1/PartitionedCall:output:04tf_op_layer_ScatterNd/shape/PartitionedCall:output:0*
Tin
2		*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_8418222'
%tf_op_layer_ScatterNd/PartitionedCall�
(rev_block1_conv2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_ScatterNd/PartitionedCall:output:0rev_block1_conv2_841936rev_block1_conv2_841938*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_8416322*
(rev_block1_conv2/StatefulPartitionedCall�
(rev_block1_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block1_conv2/StatefulPartitionedCall:output:0rev_block1_conv1_841941rev_block1_conv1_841943*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_8416542*
(rev_block1_conv1/StatefulPartitionedCall�
#tf_op_layer_Minimum/PartitionedCallPartitionedCall1rev_block1_conv1/StatefulPartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_8418482%
#tf_op_layer_Minimum/PartitionedCall�
#tf_op_layer_Maximum/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_8418622%
#tf_op_layer_Maximum/PartitionedCall�
IdentityIdentity,tf_op_layer_Maximum/PartitionedCall:output:0)^rev_block1_conv1/StatefulPartitionedCall)^rev_block1_conv2/StatefulPartitionedCall)^rev_block2_conv1/StatefulPartitionedCall)^rev_block2_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::2T
(rev_block1_conv1/StatefulPartitionedCall(rev_block1_conv1/StatefulPartitionedCall2T
(rev_block1_conv2/StatefulPartitionedCall(rev_block1_conv2/StatefulPartitionedCall2T
(rev_block2_conv1/StatefulPartitionedCall(rev_block2_conv1/StatefulPartitionedCall2T
(rev_block2_conv2/StatefulPartitionedCall(rev_block2_conv2/StatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

�
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_841654

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

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
�
i
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_842292

inputs	
identity	T
Mul_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2	
Mul_2/y_
Mul_2MulinputsMul_2/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Mul_2L
IdentityIdentity	Mul_2:z:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_842193
inputs_0
inputs_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_8419492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
k
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_842344

inputs
identity[
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	Minimum/y�
MinimumMinimuminputsMinimum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
Minimumy
IdentityIdentityMinimum:z:0*
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

�
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_841588

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�B
�
C__inference_model_1_layer_call_and_return_conditional_losses_841908
input_3
input_4	
rev_block2_conv2_841875
rev_block2_conv2_841877
rev_block2_conv1_841880
rev_block2_conv1_841882
rev_block1_conv2_841895
rev_block1_conv2_841897
rev_block1_conv1_841900
rev_block1_conv1_841902
identity��(rev_block1_conv1/StatefulPartitionedCall�(rev_block1_conv2/StatefulPartitionedCall�(rev_block2_conv1/StatefulPartitionedCall�(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_3rev_block2_conv2_841875rev_block2_conv2_841877*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_8415882*
(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block2_conv2/StatefulPartitionedCall:output:0rev_block2_conv1_841880rev_block2_conv1_841882*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_8416102*
(rev_block2_conv1/StatefulPartitionedCall�
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_8416842%
#tf_op_layer_Shape_2/PartitionedCall�
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_8417002-
+tf_op_layer_strided_slice_6/PartitionedCall�
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_8417162-
+tf_op_layer_strided_slice_5/PartitionedCall�
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_8417322-
+tf_op_layer_strided_slice_4/PartitionedCall�
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8417462#
!tf_op_layer_Mul_1/PartitionedCall�
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8417602#
!tf_op_layer_Mul_2/PartitionedCall�
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_8417742$
"tf_op_layer_Prod_2/PartitionedCall�
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2	*
Tout
2*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8417882'
%tf_op_layer_Reshape_1/PartitionedCall�
+tf_op_layer_ScatterNd/shape/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*tf_op_layer_Mul_1/PartitionedCall:output:0*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2			*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_8418052-
+tf_op_layer_ScatterNd/shape/PartitionedCall�
%tf_op_layer_ScatterNd/PartitionedCallPartitionedCallinput_4.tf_op_layer_Reshape_1/PartitionedCall:output:04tf_op_layer_ScatterNd/shape/PartitionedCall:output:0*
Tin
2		*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_8418222'
%tf_op_layer_ScatterNd/PartitionedCall�
(rev_block1_conv2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_ScatterNd/PartitionedCall:output:0rev_block1_conv2_841895rev_block1_conv2_841897*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_8416322*
(rev_block1_conv2/StatefulPartitionedCall�
(rev_block1_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block1_conv2/StatefulPartitionedCall:output:0rev_block1_conv1_841900rev_block1_conv1_841902*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_8416542*
(rev_block1_conv1/StatefulPartitionedCall�
#tf_op_layer_Minimum/PartitionedCallPartitionedCall1rev_block1_conv1/StatefulPartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_8418482%
#tf_op_layer_Minimum/PartitionedCall�
#tf_op_layer_Maximum/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_8418622%
#tf_op_layer_Maximum/PartitionedCall�
IdentityIdentity,tf_op_layer_Maximum/PartitionedCall:output:0)^rev_block1_conv1/StatefulPartitionedCall)^rev_block1_conv2/StatefulPartitionedCall)^rev_block2_conv1/StatefulPartitionedCall)^rev_block2_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::2T
(rev_block1_conv1/StatefulPartitionedCall(rev_block1_conv1/StatefulPartitionedCall2T
(rev_block1_conv2/StatefulPartitionedCall(rev_block1_conv2/StatefulPartitionedCall2T
(rev_block2_conv1/StatefulPartitionedCall(rev_block2_conv1/StatefulPartitionedCall2T
(rev_block2_conv2/StatefulPartitionedCall(rev_block2_conv2/StatefulPartitionedCall:k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�a
�
!__inference__wrapped_model_841576
input_3
input_4	;
7model_1_rev_block2_conv2_conv2d_readvariableop_resource<
8model_1_rev_block2_conv2_biasadd_readvariableop_resource;
7model_1_rev_block2_conv1_conv2d_readvariableop_resource<
8model_1_rev_block2_conv1_biasadd_readvariableop_resource;
7model_1_rev_block1_conv2_conv2d_readvariableop_resource<
8model_1_rev_block1_conv2_biasadd_readvariableop_resource;
7model_1_rev_block1_conv1_conv2d_readvariableop_resource<
8model_1_rev_block1_conv1_biasadd_readvariableop_resource
identity��
.model_1/rev_block2_conv2/Conv2D/ReadVariableOpReadVariableOp7model_1_rev_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype020
.model_1/rev_block2_conv2/Conv2D/ReadVariableOp�
model_1/rev_block2_conv2/Conv2DConv2Dinput_36model_1/rev_block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2!
model_1/rev_block2_conv2/Conv2D�
/model_1/rev_block2_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_1_rev_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype021
/model_1/rev_block2_conv2/BiasAdd/ReadVariableOp�
 model_1/rev_block2_conv2/BiasAddBiasAdd(model_1/rev_block2_conv2/Conv2D:output:07model_1/rev_block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2"
 model_1/rev_block2_conv2/BiasAdd�
model_1/rev_block2_conv2/ReluRelu)model_1/rev_block2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
model_1/rev_block2_conv2/Relu�
.model_1/rev_block2_conv1/Conv2D/ReadVariableOpReadVariableOp7model_1_rev_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype020
.model_1/rev_block2_conv1/Conv2D/ReadVariableOp�
model_1/rev_block2_conv1/Conv2DConv2D+model_1/rev_block2_conv2/Relu:activations:06model_1/rev_block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2!
model_1/rev_block2_conv1/Conv2D�
/model_1/rev_block2_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_1_rev_block2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_1/rev_block2_conv1/BiasAdd/ReadVariableOp�
 model_1/rev_block2_conv1/BiasAddBiasAdd(model_1/rev_block2_conv1/Conv2D:output:07model_1/rev_block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2"
 model_1/rev_block2_conv1/BiasAdd�
model_1/rev_block2_conv1/ReluRelu)model_1/rev_block2_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
model_1/rev_block2_conv1/Relu�
#model_1/tf_op_layer_Shape_2/Shape_2Shape+model_1/rev_block2_conv1/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2%
#model_1/tf_op_layer_Shape_2/Shape_2�
9model_1/tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/tf_op_layer_strided_slice_6/strided_slice_6/begin�
7model_1/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB:29
7model_1/tf_op_layer_strided_slice_6/strided_slice_6/end�
;model_1/tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model_1/tf_op_layer_strided_slice_6/strided_slice_6/strides�
3model_1/tf_op_layer_strided_slice_6/strided_slice_6StridedSlice,model_1/tf_op_layer_Shape_2/Shape_2:output:0Bmodel_1/tf_op_layer_strided_slice_6/strided_slice_6/begin:output:0@model_1/tf_op_layer_strided_slice_6/strided_slice_6/end:output:0Dmodel_1/tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask25
3model_1/tf_op_layer_strided_slice_6/strided_slice_6�
9model_1/tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/tf_op_layer_strided_slice_5/strided_slice_5/begin�
7model_1/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*
valueB:29
7model_1/tf_op_layer_strided_slice_5/strided_slice_5/end�
;model_1/tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model_1/tf_op_layer_strided_slice_5/strided_slice_5/strides�
3model_1/tf_op_layer_strided_slice_5/strided_slice_5StridedSlice,model_1/tf_op_layer_Shape_2/Shape_2:output:0Bmodel_1/tf_op_layer_strided_slice_5/strided_slice_5/begin:output:0@model_1/tf_op_layer_strided_slice_5/strided_slice_5/end:output:0Dmodel_1/tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask25
3model_1/tf_op_layer_strided_slice_5/strided_slice_5�
9model_1/tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model_1/tf_op_layer_strided_slice_4/strided_slice_4/begin�
7model_1/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:29
7model_1/tf_op_layer_strided_slice_4/strided_slice_4/end�
;model_1/tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model_1/tf_op_layer_strided_slice_4/strided_slice_4/strides�
3model_1/tf_op_layer_strided_slice_4/strided_slice_4StridedSlice,model_1/tf_op_layer_Shape_2/Shape_2:output:0Bmodel_1/tf_op_layer_strided_slice_4/strided_slice_4/begin:output:0@model_1/tf_op_layer_strided_slice_4/strided_slice_4/end:output:0Dmodel_1/tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask25
3model_1/tf_op_layer_strided_slice_4/strided_slice_4�
!model_1/tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!model_1/tf_op_layer_Mul_1/Mul_1/y�
model_1/tf_op_layer_Mul_1/Mul_1Mul<model_1/tf_op_layer_strided_slice_5/strided_slice_5:output:0*model_1/tf_op_layer_Mul_1/Mul_1/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2!
model_1/tf_op_layer_Mul_1/Mul_1�
!model_1/tf_op_layer_Mul_2/Mul_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!model_1/tf_op_layer_Mul_2/Mul_2/y�
model_1/tf_op_layer_Mul_2/Mul_2Mul<model_1/tf_op_layer_strided_slice_6/strided_slice_6:output:0*model_1/tf_op_layer_Mul_2/Mul_2/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2!
model_1/tf_op_layer_Mul_2/Mul_2�
3model_1/tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 25
3model_1/tf_op_layer_Prod_2/Prod_2/reduction_indices�
!model_1/tf_op_layer_Prod_2/Prod_2Prod,model_1/tf_op_layer_Shape_2/Shape_2:output:0<model_1/tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2#
!model_1/tf_op_layer_Prod_2/Prod_2�
'model_1/tf_op_layer_Reshape_1/Reshape_1Reshape+model_1/rev_block2_conv1/Relu:activations:0*model_1/tf_op_layer_Prod_2/Prod_2:output:0*
T0*
Tshape0	*
_cloned(*#
_output_shapes
:���������2)
'model_1/tf_op_layer_Reshape_1/Reshape_1�
5model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shape/3Const*
_output_shapes
: *
dtype0	*
value	B	 R@27
5model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shape/3�
3model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shapePack<model_1/tf_op_layer_strided_slice_4/strided_slice_4:output:0#model_1/tf_op_layer_Mul_1/Mul_1:z:0#model_1/tf_op_layer_Mul_2/Mul_2:z:0>model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shape/3:output:0*
N*
T0	*
_cloned(*
_output_shapes
:25
3model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shape�
'model_1/tf_op_layer_ScatterNd/ScatterNd	ScatterNdinput_40model_1/tf_op_layer_Reshape_1/Reshape_1:output:0<model_1/tf_op_layer_ScatterNd/shape/ScatterNd/shape:output:0*
T0*
Tindices0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2)
'model_1/tf_op_layer_ScatterNd/ScatterNd�
.model_1/rev_block1_conv2/Conv2D/ReadVariableOpReadVariableOp7model_1_rev_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.model_1/rev_block1_conv2/Conv2D/ReadVariableOp�
model_1/rev_block1_conv2/Conv2DConv2D0model_1/tf_op_layer_ScatterNd/ScatterNd:output:06model_1/rev_block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2!
model_1/rev_block1_conv2/Conv2D�
/model_1/rev_block1_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_1_rev_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_1/rev_block1_conv2/BiasAdd/ReadVariableOp�
 model_1/rev_block1_conv2/BiasAddBiasAdd(model_1/rev_block1_conv2/Conv2D:output:07model_1/rev_block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2"
 model_1/rev_block1_conv2/BiasAdd�
model_1/rev_block1_conv2/ReluRelu)model_1/rev_block1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
model_1/rev_block1_conv2/Relu�
.model_1/rev_block1_conv1/Conv2D/ReadVariableOpReadVariableOp7model_1_rev_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.model_1/rev_block1_conv1/Conv2D/ReadVariableOp�
model_1/rev_block1_conv1/Conv2DConv2D+model_1/rev_block1_conv2/Relu:activations:06model_1/rev_block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2!
model_1/rev_block1_conv1/Conv2D�
/model_1/rev_block1_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_1_rev_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model_1/rev_block1_conv1/BiasAdd/ReadVariableOp�
 model_1/rev_block1_conv1/BiasAddBiasAdd(model_1/rev_block1_conv1/Conv2D:output:07model_1/rev_block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2"
 model_1/rev_block1_conv1/BiasAdd�
model_1/rev_block1_conv1/ReluRelu)model_1/rev_block1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
model_1/rev_block1_conv1/Relu�
%model_1/tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2'
%model_1/tf_op_layer_Minimum/Minimum/y�
#model_1/tf_op_layer_Minimum/MinimumMinimum+model_1/rev_block1_conv1/Relu:activations:0.model_1/tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2%
#model_1/tf_op_layer_Minimum/Minimum�
%model_1/tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%model_1/tf_op_layer_Maximum/Maximum/y�
#model_1/tf_op_layer_Maximum/MaximumMaximum'model_1/tf_op_layer_Minimum/Minimum:z:0.model_1/tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2%
#model_1/tf_op_layer_Maximum/Maximum�
IdentityIdentity'model_1/tf_op_layer_Maximum/Maximum:z:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������:::::::::k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�B
�
C__inference_model_1_layer_call_and_return_conditional_losses_842008

inputs
inputs_1	
rev_block2_conv2_841975
rev_block2_conv2_841977
rev_block2_conv1_841980
rev_block2_conv1_841982
rev_block1_conv2_841995
rev_block1_conv2_841997
rev_block1_conv1_842000
rev_block1_conv1_842002
identity��(rev_block1_conv1/StatefulPartitionedCall�(rev_block1_conv2/StatefulPartitionedCall�(rev_block2_conv1/StatefulPartitionedCall�(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv2/StatefulPartitionedCallStatefulPartitionedCallinputsrev_block2_conv2_841975rev_block2_conv2_841977*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_8415882*
(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block2_conv2/StatefulPartitionedCall:output:0rev_block2_conv1_841980rev_block2_conv1_841982*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_8416102*
(rev_block2_conv1/StatefulPartitionedCall�
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_8416842%
#tf_op_layer_Shape_2/PartitionedCall�
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_8417002-
+tf_op_layer_strided_slice_6/PartitionedCall�
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_8417162-
+tf_op_layer_strided_slice_5/PartitionedCall�
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_8417322-
+tf_op_layer_strided_slice_4/PartitionedCall�
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8417462#
!tf_op_layer_Mul_1/PartitionedCall�
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8417602#
!tf_op_layer_Mul_2/PartitionedCall�
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_8417742$
"tf_op_layer_Prod_2/PartitionedCall�
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2	*
Tout
2*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8417882'
%tf_op_layer_Reshape_1/PartitionedCall�
+tf_op_layer_ScatterNd/shape/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*tf_op_layer_Mul_1/PartitionedCall:output:0*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2			*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_8418052-
+tf_op_layer_ScatterNd/shape/PartitionedCall�
%tf_op_layer_ScatterNd/PartitionedCallPartitionedCallinputs_1.tf_op_layer_Reshape_1/PartitionedCall:output:04tf_op_layer_ScatterNd/shape/PartitionedCall:output:0*
Tin
2		*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_8418222'
%tf_op_layer_ScatterNd/PartitionedCall�
(rev_block1_conv2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_ScatterNd/PartitionedCall:output:0rev_block1_conv2_841995rev_block1_conv2_841997*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_8416322*
(rev_block1_conv2/StatefulPartitionedCall�
(rev_block1_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block1_conv2/StatefulPartitionedCall:output:0rev_block1_conv1_842000rev_block1_conv1_842002*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_8416542*
(rev_block1_conv1/StatefulPartitionedCall�
#tf_op_layer_Minimum/PartitionedCallPartitionedCall1rev_block1_conv1/StatefulPartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_8418482%
#tf_op_layer_Minimum/PartitionedCall�
#tf_op_layer_Maximum/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_8418622%
#tf_op_layer_Maximum/PartitionedCall�
IdentityIdentity,tf_op_layer_Maximum/PartitionedCall:output:0)^rev_block1_conv1/StatefulPartitionedCall)^rev_block1_conv2/StatefulPartitionedCall)^rev_block2_conv1/StatefulPartitionedCall)^rev_block2_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::2T
(rev_block1_conv1/StatefulPartitionedCall(rev_block1_conv1/StatefulPartitionedCall2T
(rev_block1_conv2/StatefulPartitionedCall(rev_block1_conv2/StatefulPartitionedCall2T
(rev_block2_conv1/StatefulPartitionedCall(rev_block2_conv1/StatefulPartitionedCall2T
(rev_block2_conv2/StatefulPartitionedCall(rev_block2_conv2/StatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

�
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_841632

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
�
P
4__inference_tf_op_layer_Shape_2_layer_call_fn_842225

inputs
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_8416842
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_842331
inputs_0	
inputs_1
inputs_2	
identity�
	ScatterNd	ScatterNdinputs_0inputs_1inputs_2*
T0*
Tindices0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
	ScatterNd�
IdentityIdentityScatterNd:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������:���������::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:D@

_output_shapes
:
"
_user_specified_name
inputs/2
�
O
3__inference_tf_op_layer_Prod_2_layer_call_fn_842262

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_8417742
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_841822

inputs	
inputs_1
inputs_2	
identity�
	ScatterNd	ScatterNdinputsinputs_1inputs_2*
T0*
Tindices0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
	ScatterNd�
IdentityIdentityScatterNd:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
P
4__inference_tf_op_layer_Maximum_layer_call_fn_842360

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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_8418622
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
�X
�
C__inference_model_1_layer_call_and_return_conditional_losses_842111
inputs_0
inputs_1	3
/rev_block2_conv2_conv2d_readvariableop_resource4
0rev_block2_conv2_biasadd_readvariableop_resource3
/rev_block2_conv1_conv2d_readvariableop_resource4
0rev_block2_conv1_biasadd_readvariableop_resource3
/rev_block1_conv2_conv2d_readvariableop_resource4
0rev_block1_conv2_biasadd_readvariableop_resource3
/rev_block1_conv1_conv2d_readvariableop_resource4
0rev_block1_conv1_biasadd_readvariableop_resource
identity��
&rev_block2_conv2/Conv2D/ReadVariableOpReadVariableOp/rev_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&rev_block2_conv2/Conv2D/ReadVariableOp�
rev_block2_conv2/Conv2DConv2Dinputs_0.rev_block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
rev_block2_conv2/Conv2D�
'rev_block2_conv2/BiasAdd/ReadVariableOpReadVariableOp0rev_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'rev_block2_conv2/BiasAdd/ReadVariableOp�
rev_block2_conv2/BiasAddBiasAdd rev_block2_conv2/Conv2D:output:0/rev_block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
rev_block2_conv2/BiasAdd�
rev_block2_conv2/ReluRelu!rev_block2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
rev_block2_conv2/Relu�
&rev_block2_conv1/Conv2D/ReadVariableOpReadVariableOp/rev_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02(
&rev_block2_conv1/Conv2D/ReadVariableOp�
rev_block2_conv1/Conv2DConv2D#rev_block2_conv2/Relu:activations:0.rev_block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
rev_block2_conv1/Conv2D�
'rev_block2_conv1/BiasAdd/ReadVariableOpReadVariableOp0rev_block2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rev_block2_conv1/BiasAdd/ReadVariableOp�
rev_block2_conv1/BiasAddBiasAdd rev_block2_conv1/Conv2D:output:0/rev_block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block2_conv1/BiasAdd�
rev_block2_conv1/ReluRelu!rev_block2_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block2_conv1/Relu�
tf_op_layer_Shape_2/Shape_2Shape#rev_block2_conv1/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_2/Shape_2�
1tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_6/strided_slice_6/begin�
/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_6/strided_slice_6/end�
3tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_6/strided_slice_6/strides�
+tf_op_layer_strided_slice_6/strided_slice_6StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_6/strided_slice_6/begin:output:08tf_op_layer_strided_slice_6/strided_slice_6/end:output:0<tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_6/strided_slice_6�
1tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_5/strided_slice_5/begin�
/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_5/strided_slice_5/end�
3tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_5/strided_slice_5/strides�
+tf_op_layer_strided_slice_5/strided_slice_5StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_5/strided_slice_5/begin:output:08tf_op_layer_strided_slice_5/strided_slice_5/end:output:0<tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_5/strided_slice_5�
1tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_4/strided_slice_4/begin�
/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_4/strided_slice_4/end�
3tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_4/strided_slice_4/strides�
+tf_op_layer_strided_slice_4/strided_slice_4StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_4/strided_slice_4/begin:output:08tf_op_layer_strided_slice_4/strided_slice_4/end:output:0<tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_4/strided_slice_4x
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Mul_1/Mul_1/y�
tf_op_layer_Mul_1/Mul_1Mul4tf_op_layer_strided_slice_5/strided_slice_5:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mul_1/Mul_1x
tf_op_layer_Mul_2/Mul_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Mul_2/Mul_2/y�
tf_op_layer_Mul_2/Mul_2Mul4tf_op_layer_strided_slice_6/strided_slice_6:output:0"tf_op_layer_Mul_2/Mul_2/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mul_2/Mul_2�
+tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_2/Prod_2/reduction_indices�
tf_op_layer_Prod_2/Prod_2Prod$tf_op_layer_Shape_2/Shape_2:output:04tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_2/Prod_2�
tf_op_layer_Reshape_1/Reshape_1Reshape#rev_block2_conv1/Relu:activations:0"tf_op_layer_Prod_2/Prod_2:output:0*
T0*
Tshape0	*
_cloned(*#
_output_shapes
:���������2!
tf_op_layer_Reshape_1/Reshape_1�
-tf_op_layer_ScatterNd/shape/ScatterNd/shape/3Const*
_output_shapes
: *
dtype0	*
value	B	 R@2/
-tf_op_layer_ScatterNd/shape/ScatterNd/shape/3�
+tf_op_layer_ScatterNd/shape/ScatterNd/shapePack4tf_op_layer_strided_slice_4/strided_slice_4:output:0tf_op_layer_Mul_1/Mul_1:z:0tf_op_layer_Mul_2/Mul_2:z:06tf_op_layer_ScatterNd/shape/ScatterNd/shape/3:output:0*
N*
T0	*
_cloned(*
_output_shapes
:2-
+tf_op_layer_ScatterNd/shape/ScatterNd/shape�
tf_op_layer_ScatterNd/ScatterNd	ScatterNdinputs_1(tf_op_layer_Reshape_1/Reshape_1:output:04tf_op_layer_ScatterNd/shape/ScatterNd/shape:output:0*
T0*
Tindices0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2!
tf_op_layer_ScatterNd/ScatterNd�
&rev_block1_conv2/Conv2D/ReadVariableOpReadVariableOp/rev_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&rev_block1_conv2/Conv2D/ReadVariableOp�
rev_block1_conv2/Conv2DConv2D(tf_op_layer_ScatterNd/ScatterNd:output:0.rev_block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
rev_block1_conv2/Conv2D�
'rev_block1_conv2/BiasAdd/ReadVariableOpReadVariableOp0rev_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rev_block1_conv2/BiasAdd/ReadVariableOp�
rev_block1_conv2/BiasAddBiasAdd rev_block1_conv2/Conv2D:output:0/rev_block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block1_conv2/BiasAdd�
rev_block1_conv2/ReluRelu!rev_block1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
rev_block1_conv2/Relu�
&rev_block1_conv1/Conv2D/ReadVariableOpReadVariableOp/rev_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&rev_block1_conv1/Conv2D/ReadVariableOp�
rev_block1_conv1/Conv2DConv2D#rev_block1_conv2/Relu:activations:0.rev_block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
rev_block1_conv1/Conv2D�
'rev_block1_conv1/BiasAdd/ReadVariableOpReadVariableOp0rev_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rev_block1_conv1/BiasAdd/ReadVariableOp�
rev_block1_conv1/BiasAddBiasAdd rev_block1_conv1/Conv2D:output:0/rev_block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2
rev_block1_conv1/BiasAdd�
rev_block1_conv1/ReluRelu!rev_block1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
rev_block1_conv1/Relu�
tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
tf_op_layer_Minimum/Minimum/y�
tf_op_layer_Minimum/MinimumMinimum#rev_block1_conv1/Relu:activations:0&tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_Minimum/Minimum�
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tf_op_layer_Maximum/Maximum/y�
tf_op_layer_Maximum/MaximumMaximumtf_op_layer_Minimum/Minimum:z:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2
tf_op_layer_Maximum/Maximum�
IdentityIdentitytf_op_layer_Maximum/Maximum:z:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������:::::::::l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
b
6__inference_tf_op_layer_Reshape_1_layer_call_fn_842309
inputs_0
inputs_1	
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8417882
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+���������������������������@::k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
i
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_841760

inputs	
identity	T
Mul_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2	
Mul_2/y_
Mul_2MulinputsMul_2/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Mul_2L
IdentityIdentity	Mul_2:z:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
X
<__inference_tf_op_layer_strided_slice_4_layer_call_fn_842275

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_8417322
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
k
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_841848

inputs
identity[
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	Minimum/y�
MinimumMinimuminputsMinimum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
Minimumy
IdentityIdentityMinimum:z:0*
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
�
v
<__inference_tf_op_layer_ScatterNd/shape_layer_call_fn_842324
inputs_0	
inputs_1	
inputs_2	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2			*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_8418052
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
: : : :@ <

_output_shapes
: 
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1:@<

_output_shapes
: 
"
_user_specified_name
inputs/2
�
�
1__inference_rev_block1_conv2_layer_call_fn_841642

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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_8416322
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
�B
�
C__inference_model_1_layer_call_and_return_conditional_losses_841871
input_3
input_4	
rev_block2_conv2_841669
rev_block2_conv2_841671
rev_block2_conv1_841674
rev_block2_conv1_841676
rev_block1_conv2_841832
rev_block1_conv2_841834
rev_block1_conv1_841837
rev_block1_conv1_841839
identity��(rev_block1_conv1/StatefulPartitionedCall�(rev_block1_conv2/StatefulPartitionedCall�(rev_block2_conv1/StatefulPartitionedCall�(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_3rev_block2_conv2_841669rev_block2_conv2_841671*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_8415882*
(rev_block2_conv2/StatefulPartitionedCall�
(rev_block2_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block2_conv2/StatefulPartitionedCall:output:0rev_block2_conv1_841674rev_block2_conv1_841676*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_8416102*
(rev_block2_conv1/StatefulPartitionedCall�
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_8416842%
#tf_op_layer_Shape_2/PartitionedCall�
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_8417002-
+tf_op_layer_strided_slice_6/PartitionedCall�
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_8417162-
+tf_op_layer_strided_slice_5/PartitionedCall�
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_8417322-
+tf_op_layer_strided_slice_4/PartitionedCall�
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8417462#
!tf_op_layer_Mul_1/PartitionedCall�
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8417602#
!tf_op_layer_Mul_2/PartitionedCall�
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_8417742$
"tf_op_layer_Prod_2/PartitionedCall�
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall1rev_block2_conv1/StatefulPartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2	*
Tout
2*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8417882'
%tf_op_layer_Reshape_1/PartitionedCall�
+tf_op_layer_ScatterNd/shape/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*tf_op_layer_Mul_1/PartitionedCall:output:0*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2			*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_8418052-
+tf_op_layer_ScatterNd/shape/PartitionedCall�
%tf_op_layer_ScatterNd/PartitionedCallPartitionedCallinput_4.tf_op_layer_Reshape_1/PartitionedCall:output:04tf_op_layer_ScatterNd/shape/PartitionedCall:output:0*
Tin
2		*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_8418222'
%tf_op_layer_ScatterNd/PartitionedCall�
(rev_block1_conv2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_ScatterNd/PartitionedCall:output:0rev_block1_conv2_841832rev_block1_conv2_841834*
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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_8416322*
(rev_block1_conv2/StatefulPartitionedCall�
(rev_block1_conv1/StatefulPartitionedCallStatefulPartitionedCall1rev_block1_conv2/StatefulPartitionedCall:output:0rev_block1_conv1_841837rev_block1_conv1_841839*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_8416542*
(rev_block1_conv1/StatefulPartitionedCall�
#tf_op_layer_Minimum/PartitionedCallPartitionedCall1rev_block1_conv1/StatefulPartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_8418482%
#tf_op_layer_Minimum/PartitionedCall�
#tf_op_layer_Maximum/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
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
GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_8418622%
#tf_op_layer_Maximum/PartitionedCall�
IdentityIdentity,tf_op_layer_Maximum/PartitionedCall:output:0)^rev_block1_conv1/StatefulPartitionedCall)^rev_block1_conv2/StatefulPartitionedCall)^rev_block2_conv1/StatefulPartitionedCall)^rev_block2_conv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::2T
(rev_block1_conv1/StatefulPartitionedCall(rev_block1_conv1/StatefulPartitionedCall2T
(rev_block1_conv2/StatefulPartitionedCall(rev_block1_conv2/StatefulPartitionedCall2T
(rev_block2_conv1/StatefulPartitionedCall(rev_block2_conv1/StatefulPartitionedCall2T
(rev_block2_conv2/StatefulPartitionedCall(rev_block2_conv2/StatefulPartitionedCall:k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
X
<__inference_tf_op_layer_strided_slice_6_layer_call_fn_842251

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_8417002
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
j
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_841774

inputs	
identity	~
Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_2/reduction_indices�
Prod_2Prodinputs!Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_2V
IdentityIdentityProd_2:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
X
<__inference_tf_op_layer_strided_slice_5_layer_call_fn_842238

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_8417162
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
k
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_842355

inputs
identity[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/y�
MaximumMaximuminputsMaximum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
Maximumy
IdentityIdentityMaximum:z:0*
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
�
�
1__inference_rev_block2_conv2_layer_call_fn_841598

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_8415882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
{
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_841788

inputs
inputs_1	
identity~
	Reshape_1Reshapeinputsinputs_1*
T0*
Tshape0	*
_cloned(*#
_output_shapes
:���������2
	Reshape_1b
IdentityIdentityReshape_1:output:0*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+���������������������������@::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
1__inference_rev_block2_conv1_layer_call_fn_841620

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
GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_8416102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�+
�
"__inference__traced_restore_842448
file_prefix,
(assignvariableop_rev_block2_conv2_kernel,
(assignvariableop_1_rev_block2_conv2_bias.
*assignvariableop_2_rev_block2_conv1_kernel,
(assignvariableop_3_rev_block2_conv1_bias.
*assignvariableop_4_rev_block1_conv2_kernel,
(assignvariableop_5_rev_block1_conv2_bias.
*assignvariableop_6_rev_block1_conv1_kernel,
(assignvariableop_7_rev_block1_conv1_bias

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp(assignvariableop_rev_block2_conv2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_rev_block2_conv2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_rev_block2_conv1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_rev_block2_conv1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_rev_block1_conv2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp(assignvariableop_5_rev_block1_conv2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp*assignvariableop_6_rev_block1_conv1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rev_block1_conv1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7�
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
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
k
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_841862

inputs
identity[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/y�
MaximumMaximuminputsMaximum/y:output:0*
T0*
_cloned(*A
_output_shapes/
-:+���������������������������2	
Maximumy
IdentityIdentityMaximum:z:0*
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
�
k
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_841684

inputs
identity	g
Shape_2Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_2W
IdentityIdentityShape_2:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
1__inference_rev_block1_conv1_layer_call_fn_841664

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
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_8416542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

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
�
p
6__inference_tf_op_layer_ScatterNd_layer_call_fn_842338
inputs_0	
inputs_1
inputs_2	
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2		*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_8418222
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������:���������::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:D@

_output_shapes
:
"
_user_specified_name
inputs/2
�
s
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_841732

inputs	
identity	x
strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/begint
strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/end|
strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/strides�
strided_slice_4StridedSliceinputsstrided_slice_4/begin:output:0strided_slice_4/end:output:0 strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4[
IdentityIdentitystrided_slice_4:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_842051
input_3
input_4	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_8415762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
s
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_841700

inputs	
identity	x
strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/begint
strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/end|
strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/strides�
strided_slice_6StridedSliceinputsstrided_slice_6/begin:output:0strided_slice_6/end:output:0 strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6[
IdentityIdentitystrided_slice_6:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
N
2__inference_tf_op_layer_Mul_1_layer_call_fn_842286

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8417462
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
i
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_842281

inputs	
identity	T
Mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2	
Mul_1/y_
Mul_1MulinputsMul_1/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Mul_1L
IdentityIdentity	Mul_1:z:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
j
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_842257

inputs	
identity	~
Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_2/reduction_indices�
Prod_2Prodinputs!Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_2V
IdentityIdentityProd_2:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
k
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_842220

inputs
identity	g
Shape_2Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_2W
IdentityIdentityShape_2:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_842027
input_3
input_4	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_8420082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�'
�
__inference__traced_save_842412
file_prefix6
2savev2_rev_block2_conv2_kernel_read_readvariableop4
0savev2_rev_block2_conv2_bias_read_readvariableop6
2savev2_rev_block2_conv1_kernel_read_readvariableop4
0savev2_rev_block2_conv1_bias_read_readvariableop6
2savev2_rev_block1_conv2_kernel_read_readvariableop4
0savev2_rev_block1_conv2_bias_read_readvariableop6
2savev2_rev_block1_conv1_kernel_read_readvariableop4
0savev2_rev_block1_conv1_bias_read_readvariableop
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
value3B1 B+_temp_da8856fa9f704c41b66ada4ad01edbc5/part2	
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_rev_block2_conv2_kernel_read_readvariableop0savev2_rev_block2_conv2_bias_read_readvariableop2savev2_rev_block2_conv1_kernel_read_readvariableop0savev2_rev_block2_conv1_bias_read_readvariableop2savev2_rev_block1_conv2_kernel_read_readvariableop0savev2_rev_block1_conv2_bias_read_readvariableop2savev2_rev_block1_conv1_kernel_read_readvariableop0savev2_rev_block1_conv1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
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

identity_1Identity_1:output:0*{
_input_shapesj
h: :��:�:�@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:�@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::	

_output_shapes
: 
�
�
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_841805

inputs	
inputs_1	
inputs_2	
identity	h
ScatterNd/shape/3Const*
_output_shapes
: *
dtype0	*
value	B	 R@2
ScatterNd/shape/3�
ScatterNd/shapePackinputsinputs_1inputs_2ScatterNd/shape/3:output:0*
N*
T0	*
_cloned(*
_output_shapes
:2
ScatterNd/shape_
IdentityIdentityScatterNd/shape:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
: : : :> :

_output_shapes
: 
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
�
s
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_842270

inputs	
identity	x
strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/begint
strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/end|
strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/strides�
strided_slice_4StridedSliceinputsstrided_slice_4/begin:output:0strided_slice_4/end:output:0 strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4[
IdentityIdentitystrided_slice_4:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
s
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_841716

inputs	
identity	x
strided_slice_5/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/begint
strided_slice_5/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/end|
strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/strides�
strided_slice_5StridedSliceinputsstrided_slice_5/begin:output:0strided_slice_5/end:output:0 strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5[
IdentityIdentitystrided_slice_5:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_842215
inputs_0
inputs_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_8420082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_841968
input_3
input_4	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_8419492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:,����������������������������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,����������������������������
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

�
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_841610

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
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
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
i
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_841746

inputs	
identity	T
Mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2	
Mul_1/y_
Mul_1MulinputsMul_1/y:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Mul_1L
IdentityIdentity	Mul_1:z:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_842317
inputs_0	
inputs_1	
inputs_2	
identity	h
ScatterNd/shape/3Const*
_output_shapes
: *
dtype0	*
value	B	 R@2
ScatterNd/shape/3�
ScatterNd/shapePackinputs_0inputs_1inputs_2ScatterNd/shape/3:output:0*
N*
T0	*
_cloned(*
_output_shapes
:2
ScatterNd/shape_
IdentityIdentityScatterNd/shape:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
: : : :@ <

_output_shapes
: 
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1:@<

_output_shapes
: 
"
_user_specified_name
inputs/2"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
V
input_3K
serving_default_input_3:0,����������������������������
;
input_40
serving_default_input_4:0	���������a
tf_op_layer_MaximumJ
StatefulPartitionedCall:0+���������������������������tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"��
_tf_keras_model��{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "rev_block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block2_conv2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block2_conv1", "inbound_nodes": [[["rev_block2_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["rev_block2_conv1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Shape_2", "inbound_nodes": [[["rev_block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_5", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [2], "3": [1]}}, "name": "tf_op_layer_strided_slice_5", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_6", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [2], "2": [3], "3": [1]}}, "name": "tf_op_layer_strided_slice_6", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["Shape_2", "Prod_2/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_2", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_4", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_4", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["strided_slice_5", "Mul_1/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_strided_slice_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["strided_slice_6", "Mul_2/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_Mul_2", "inbound_nodes": [[["tf_op_layer_strided_slice_6", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["rev_block2_conv1/Identity", "Prod_2"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["rev_block2_conv1", 0, 0, {}], ["tf_op_layer_Prod_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ScatterNd/shape", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd/shape", "op": "Pack", "input": ["strided_slice_4", "Mul_1", "Mul_2", "ScatterNd/shape/3"], "attr": {"N": {"i": "4"}, "axis": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"3": 64}}, "name": "tf_op_layer_ScatterNd/shape", "inbound_nodes": [[["tf_op_layer_strided_slice_4", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}], ["tf_op_layer_Mul_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ScatterNd", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd", "op": "ScatterNd", "input": ["input_4", "Reshape_1", "ScatterNd/shape"], "attr": {"Tindices": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_ScatterNd", "inbound_nodes": [[["input_4", 0, 0, {}], ["tf_op_layer_Reshape_1", 0, 0, {}], ["tf_op_layer_ScatterNd/shape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block1_conv2", "inbound_nodes": [[["tf_op_layer_ScatterNd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block1_conv1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block1_conv1", "inbound_nodes": [[["rev_block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["rev_block1_conv1/Identity", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_Minimum", "inbound_nodes": [[["rev_block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Minimum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["tf_op_layer_Minimum", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["tf_op_layer_Maximum", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 128]}, {"class_name": "TensorShape", "items": [null, 4]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "rev_block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block2_conv2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block2_conv1", "inbound_nodes": [[["rev_block2_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["rev_block2_conv1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Shape_2", "inbound_nodes": [[["rev_block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_5", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [2], "3": [1]}}, "name": "tf_op_layer_strided_slice_5", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_6", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [2], "2": [3], "3": [1]}}, "name": "tf_op_layer_strided_slice_6", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["Shape_2", "Prod_2/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_2", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_4", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_4", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["strided_slice_5", "Mul_1/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_strided_slice_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["strided_slice_6", "Mul_2/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_Mul_2", "inbound_nodes": [[["tf_op_layer_strided_slice_6", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["rev_block2_conv1/Identity", "Prod_2"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["rev_block2_conv1", 0, 0, {}], ["tf_op_layer_Prod_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ScatterNd/shape", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd/shape", "op": "Pack", "input": ["strided_slice_4", "Mul_1", "Mul_2", "ScatterNd/shape/3"], "attr": {"N": {"i": "4"}, "axis": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"3": 64}}, "name": "tf_op_layer_ScatterNd/shape", "inbound_nodes": [[["tf_op_layer_strided_slice_4", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}], ["tf_op_layer_Mul_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ScatterNd", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd", "op": "ScatterNd", "input": ["input_4", "Reshape_1", "ScatterNd/shape"], "attr": {"Tindices": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_ScatterNd", "inbound_nodes": [[["input_4", 0, 0, {}], ["tf_op_layer_Reshape_1", 0, 0, {}], ["tf_op_layer_ScatterNd/shape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block1_conv2", "inbound_nodes": [[["tf_op_layer_ScatterNd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "rev_block1_conv1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rev_block1_conv1", "inbound_nodes": [[["rev_block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["rev_block1_conv1/Identity", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_Minimum", "inbound_nodes": [[["rev_block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Minimum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["tf_op_layer_Minimum", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["tf_op_layer_Maximum", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
�	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "rev_block2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rev_block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
�	

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "rev_block2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rev_block2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
�
$trainable_variables
%regularization_losses
&	variables
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["rev_block2_conv1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�
(trainable_variables
)regularization_losses
*	variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_5", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [2], "3": [1]}}}
�
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_6", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [2], "2": [3], "3": [1]}}}
�
0trainable_variables
1regularization_losses
2	variables
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["Shape_2", "Prod_2/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}}
�
4trainable_variables
5regularization_losses
6	variables
7	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_4", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
�
8trainable_variables
9regularization_losses
:	variables
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["strided_slice_5", "Mul_1/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}}
�
<trainable_variables
=regularization_losses
>	variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["strided_slice_6", "Mul_2/y"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {"1": 2}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_4", "dtype": "int64", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "input_4"}}
�
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["rev_block2_conv1/Identity", "Prod_2"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}}
�
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ScatterNd/shape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ScatterNd/shape", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd/shape", "op": "Pack", "input": ["strided_slice_4", "Mul_1", "Mul_2", "ScatterNd/shape/3"], "attr": {"N": {"i": "4"}, "axis": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"3": 64}}}
�
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ScatterNd", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ScatterNd", "trainable": true, "dtype": "float32", "node_def": {"name": "ScatterNd", "op": "ScatterNd", "input": ["input_4", "Reshape_1", "ScatterNd/shape"], "attr": {"Tindices": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�	

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "rev_block1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rev_block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
�	

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "rev_block1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rev_block1_conv1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
�
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Minimum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["rev_block1_conv1/Identity", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}}
�
\trainable_variables
]regularization_losses
^	variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Minimum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}}
X
0
1
2
3
L4
M5
R6
S7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
L4
M5
R6
S7"
trackable_list_wrapper
�
trainable_variables

`layers
alayer_metrics
bmetrics
cnon_trainable_variables
regularization_losses
	variables
dlayer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
3:1��2rev_block2_conv2/kernel
$:"�2rev_block2_conv2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables

elayers
flayer_metrics
gmetrics
hnon_trainable_variables
regularization_losses
	variables
ilayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0�@2rev_block2_conv1/kernel
#:!@2rev_block2_conv1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 trainable_variables

jlayers
klayer_metrics
lmetrics
mnon_trainable_variables
!regularization_losses
"	variables
nlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$trainable_variables

olayers
player_metrics
qmetrics
rnon_trainable_variables
%regularization_losses
&	variables
slayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
(trainable_variables

tlayers
ulayer_metrics
vmetrics
wnon_trainable_variables
)regularization_losses
*	variables
xlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,trainable_variables

ylayers
zlayer_metrics
{metrics
|non_trainable_variables
-regularization_losses
.	variables
}layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0trainable_variables

~layers
layer_metrics
�metrics
�non_trainable_variables
1regularization_losses
2	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
4trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
5regularization_losses
6	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
8trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
9regularization_losses
:	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
<trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
=regularization_losses
>	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
@trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Aregularization_losses
B	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Eregularization_losses
F	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Htrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Iregularization_losses
J	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1:/@@2rev_block1_conv2/kernel
#:!@2rev_block1_conv2/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
�
Ntrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Oregularization_losses
P	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1:/@2rev_block1_conv1/kernel
#:!2rev_block1_conv1/bias
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
�
Ttrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Uregularization_losses
V	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Yregularization_losses
Z	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
]regularization_losses
^	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
(__inference_model_1_layer_call_fn_841968
(__inference_model_1_layer_call_fn_842215
(__inference_model_1_layer_call_fn_842027
(__inference_model_1_layer_call_fn_842193�
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
�2�
!__inference__wrapped_model_841576�
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
annotations� *i�f
d�a
<�9
input_3,����������������������������
!�
input_4���������	
�2�
C__inference_model_1_layer_call_and_return_conditional_losses_842111
C__inference_model_1_layer_call_and_return_conditional_losses_842171
C__inference_model_1_layer_call_and_return_conditional_losses_841908
C__inference_model_1_layer_call_and_return_conditional_losses_841871�
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
�2�
1__inference_rev_block2_conv2_layer_call_fn_841598�
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
annotations� *8�5
3�0,����������������������������
�2�
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_841588�
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
annotations� *8�5
3�0,����������������������������
�2�
1__inference_rev_block2_conv1_layer_call_fn_841620�
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
annotations� *8�5
3�0,����������������������������
�2�
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_841610�
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
annotations� *8�5
3�0,����������������������������
�2�
4__inference_tf_op_layer_Shape_2_layer_call_fn_842225�
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
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_842220�
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
<__inference_tf_op_layer_strided_slice_5_layer_call_fn_842238�
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
�2�
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_842233�
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
<__inference_tf_op_layer_strided_slice_6_layer_call_fn_842251�
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
�2�
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_842246�
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
3__inference_tf_op_layer_Prod_2_layer_call_fn_842262�
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
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_842257�
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
<__inference_tf_op_layer_strided_slice_4_layer_call_fn_842275�
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
�2�
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_842270�
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
2__inference_tf_op_layer_Mul_1_layer_call_fn_842286�
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
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_842281�
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
2__inference_tf_op_layer_Mul_2_layer_call_fn_842297�
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
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_842292�
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
6__inference_tf_op_layer_Reshape_1_layer_call_fn_842309�
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
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_842303�
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
<__inference_tf_op_layer_ScatterNd/shape_layer_call_fn_842324�
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
�2�
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_842317�
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
6__inference_tf_op_layer_ScatterNd_layer_call_fn_842338�
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
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_842331�
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
1__inference_rev_block1_conv2_layer_call_fn_841642�
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
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_841632�
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
1__inference_rev_block1_conv1_layer_call_fn_841664�
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
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_841654�
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
�2�
4__inference_tf_op_layer_Minimum_layer_call_fn_842349�
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
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_842344�
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
4__inference_tf_op_layer_Maximum_layer_call_fn_842360�
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
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_842355�
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
:B8
$__inference_signature_wrapper_842051input_3input_4�
!__inference__wrapped_model_841576�LMRSs�p
i�f
d�a
<�9
input_3,����������������������������
!�
input_4���������	
� "c�`
^
tf_op_layer_MaximumG�D
tf_op_layer_Maximum+����������������������������
C__inference_model_1_layer_call_and_return_conditional_losses_841871�LMRS{�x
q�n
d�a
<�9
input_3,����������������������������
!�
input_4���������	
p

 
� "?�<
5�2
0+���������������������������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_841908�LMRS{�x
q�n
d�a
<�9
input_3,����������������������������
!�
input_4���������	
p 

 
� "?�<
5�2
0+���������������������������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_842111�LMRS}�z
s�p
f�c
=�:
inputs/0,����������������������������
"�
inputs/1���������	
p

 
� "?�<
5�2
0+���������������������������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_842171�LMRS}�z
s�p
f�c
=�:
inputs/0,����������������������������
"�
inputs/1���������	
p 

 
� "?�<
5�2
0+���������������������������
� �
(__inference_model_1_layer_call_fn_841968�LMRS{�x
q�n
d�a
<�9
input_3,����������������������������
!�
input_4���������	
p

 
� "2�/+����������������������������
(__inference_model_1_layer_call_fn_842027�LMRS{�x
q�n
d�a
<�9
input_3,����������������������������
!�
input_4���������	
p 

 
� "2�/+����������������������������
(__inference_model_1_layer_call_fn_842193�LMRS}�z
s�p
f�c
=�:
inputs/0,����������������������������
"�
inputs/1���������	
p

 
� "2�/+����������������������������
(__inference_model_1_layer_call_fn_842215�LMRS}�z
s�p
f�c
=�:
inputs/0,����������������������������
"�
inputs/1���������	
p 

 
� "2�/+����������������������������
L__inference_rev_block1_conv1_layer_call_and_return_conditional_losses_841654�RSI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������
� �
1__inference_rev_block1_conv1_layer_call_fn_841664�RSI�F
?�<
:�7
inputs+���������������������������@
� "2�/+����������������������������
L__inference_rev_block1_conv2_layer_call_and_return_conditional_losses_841632�LMI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_rev_block1_conv2_layer_call_fn_841642�LMI�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_rev_block2_conv1_layer_call_and_return_conditional_losses_841610�J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
1__inference_rev_block2_conv1_layer_call_fn_841620�J�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
L__inference_rev_block2_conv2_layer_call_and_return_conditional_losses_841588�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
1__inference_rev_block2_conv2_layer_call_fn_841598�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
$__inference_signature_wrapper_842051�LMRS���
� 
z�w
G
input_3<�9
input_3,����������������������������
,
input_4!�
input_4���������	"c�`
^
tf_op_layer_MaximumG�D
tf_op_layer_Maximum+����������������������������
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_842355�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
4__inference_tf_op_layer_Maximum_layer_call_fn_842360I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_842344�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
4__inference_tf_op_layer_Minimum_layer_call_fn_842349I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_8422816�
�
�
inputs 	
� "�

�
0 	
� _
2__inference_tf_op_layer_Mul_1_layer_call_fn_842286)�
�
�
inputs 	
� "� 	�
M__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_8422926�
�
�
inputs 	
� "�

�
0 	
� _
2__inference_tf_op_layer_Mul_2_layer_call_fn_842297)�
�
�
inputs 	
� "� 	�
N__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_842257>"�
�
�
inputs	
� "�
�
0	
� h
3__inference_tf_op_layer_Prod_2_layer_call_fn_8422621"�
�
�
inputs	
� "�	�
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_842303�g�d
]�Z
X�U
<�9
inputs/0+���������������������������@
�
inputs/1	
� "!�
�
0���������
� �
6__inference_tf_op_layer_Reshape_1_layer_call_fn_842309g�d
]�Z
X�U
<�9
inputs/0+���������������������������@
�
inputs/1	
� "�����������
W__inference_tf_op_layer_ScatterNd/shape_layer_call_and_return_conditional_losses_842317gK�H
A�>
<�9
�
inputs/0 	
�
inputs/1 	
�
inputs/2 	
� "�
�
0	
� �
<__inference_tf_op_layer_ScatterNd/shape_layer_call_fn_842324ZK�H
A�>
<�9
�
inputs/0 	
�
inputs/1 	
�
inputs/2 	
� "�	�
Q__inference_tf_op_layer_ScatterNd_layer_call_and_return_conditional_losses_842331�m�j
c�`
^�[
"�
inputs/0���������	
�
inputs/1���������
�
inputs/2	
� "H�E
>�;
04������������������������������������
� �
6__inference_tf_op_layer_ScatterNd_layer_call_fn_842338�m�j
c�`
^�[
"�
inputs/0���������	
�
inputs/1���������
�
inputs/2	
� ";�84�������������������������������������
O__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_842220eI�F
?�<
:�7
inputs+���������������������������@
� "�
�
0	
� �
4__inference_tf_op_layer_Shape_2_layer_call_fn_842225XI�F
?�<
:�7
inputs+���������������������������@
� "�	�
W__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_842270:"�
�
�
inputs	
� "�

�
0 	
� m
<__inference_tf_op_layer_strided_slice_4_layer_call_fn_842275-"�
�
�
inputs	
� "� 	�
W__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_842233:"�
�
�
inputs	
� "�

�
0 	
� m
<__inference_tf_op_layer_strided_slice_5_layer_call_fn_842238-"�
�
�
inputs	
� "� 	�
W__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_842246:"�
�
�
inputs	
� "�

�
0 	
� m
<__inference_tf_op_layer_strided_slice_6_layer_call_fn_842251-"�
�
�
inputs	
� "� 	