��
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
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��	
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
�
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*$
shared_nameblock2_conv1/kernel
�
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@�*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:�*
dtype0
�
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock2_conv2/kernel
�
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:��*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�=
value�=B�= B�=
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
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
layer-14
layer-15
layer-16
layer-17
layer_with_weights-2
layer-18
layer-19
layer_with_weights-3
layer-20
layer-21
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
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
R
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
R
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
R
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
R
`trainable_variables
aregularization_losses
b	variables
c	keras_api
h

dkernel
ebias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
R
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
h

nkernel
obias
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
R
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
 
 
8
$0
%1
*2
+3
d4
e5
n6
o7
�
trainable_variables

xlayers
ylayer_metrics
zmetrics
{non_trainable_variables
regularization_losses
	variables
|layer_regularization_losses
 
 
 
 
�
trainable_variables

}layers
~layer_metrics
metrics
�non_trainable_variables
regularization_losses
	variables
 �layer_regularization_losses
 
 
 
�
 trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
!regularization_losses
"	variables
 �layer_regularization_losses
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

$0
%1
�
&trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
'regularization_losses
(	variables
 �layer_regularization_losses
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

*0
+1
�
,trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
-regularization_losses
.	variables
 �layer_regularization_losses
 
 
 
�
0trainable_variables
�layers
�layer_metrics
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
 
 
 
�
Ltrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Mregularization_losses
N	variables
 �layer_regularization_losses
 
 
 
�
Ptrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Qregularization_losses
R	variables
 �layer_regularization_losses
 
 
 
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
 
 
 
�
`trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
aregularization_losses
b	variables
 �layer_regularization_losses
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

d0
e1
�
ftrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
gregularization_losses
h	variables
 �layer_regularization_losses
 
 
 
�
jtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
kregularization_losses
l	variables
 �layer_regularization_losses
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

n0
o1
�
ptrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
qregularization_losses
r	variables
 �layer_regularization_losses
 
 
 
�
ttrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
uregularization_losses
v	variables
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
18
19
20
21
 
 
8
$0
%1
*2
+3
d4
e5
n6
o7
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
$0
%1
 
 
 
 

*0
+1
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

d0
e1
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
n0
o1
 
 
 
 
 
 
�
serving_default_input_2Placeholder*A
_output_shapes/
-:+���������������������������*
dtype0*6
shape-:+���������������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/bias*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference_signature_wrapper_6411
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOpConst*
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
CPU2*0J 8*&
f!R
__inference__traced_save_6849
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/bias*
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
CPU2*0J 8*)
f$R"
 __inference__traced_restore_6885��	
�
V
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_6656

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
CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_60132
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
u
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_6733
inputs_0	
inputs_1	
identity	�
MulMulinputs_0inputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
Mul~
IdentityIdentityMul:z:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4������������������������������������: :t p
J
_output_shapes8
6:4������������������������������������
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
�
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_5942

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
�f
�
?__inference_model_layer_call_and_return_conditional_losses_6480

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource
identity

identity_1	��
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
block1_conv2/Relu�
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmaxblock1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+���������������������������@:+���������������������������@*
ksize
*
paddingSAME*
strides
21
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax�
tf_op_layer_Shape_1/Shape_1Shape8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_1/Shape_1�
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_1/strided_slice_1/begin�
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_1/strided_slice_1/end�
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_1/strided_slice_1/strides�
+tf_op_layer_strided_slice_1/strided_slice_1StridedSlice$tf_op_layer_Shape_1/Shape_1:output:0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_1/strided_slice_1�
tf_op_layer_Shape/ShapeShapeblock1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape/Shape�
tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
tf_op_layer_Range/Range/start�
tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Range/Range/delta�
tf_op_layer_Range/RangeRange&tf_op_layer_Range/Range/start:output:04tf_op_layer_strided_slice_1/strided_slice_1:output:0&tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:���������2
tf_op_layer_Range/Range�
1tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_3/strided_slice_3/begin�
/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_3/strided_slice_3/end�
3tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_3/strided_slice_3/strides�
+tf_op_layer_strided_slice_3/strided_slice_3StridedSlice tf_op_layer_Shape/Shape:output:0:tf_op_layer_strided_slice_3/strided_slice_3/begin:output:08tf_op_layer_strided_slice_3/strided_slice_3/end:output:0<tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_3/strided_slice_3�
1tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_2/strided_slice_2/begin�
/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_2/strided_slice_2/end�
3tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_2/strided_slice_2/strides�
+tf_op_layer_strided_slice_2/strided_slice_2StridedSlice tf_op_layer_Range/Range:output:0:tf_op_layer_strided_slice_2/strided_slice_2/begin:output:08tf_op_layer_strided_slice_2/strided_slice_2/end:output:0<tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:���������*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_2/strided_slice_2�
#tf_op_layer_BroadcastTo/BroadcastToBroadcastTo4tf_op_layer_strided_slice_2/strided_slice_2:output:0$tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2%
#tf_op_layer_BroadcastTo/BroadcastTo�
'tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2)
'tf_op_layer_Prod/Prod/reduction_indices�
tf_op_layer_Prod/ProdProd4tf_op_layer_strided_slice_3/strided_slice_3:output:00tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod/Prod�
tf_op_layer_Mul/MulMul,tf_op_layer_BroadcastTo/BroadcastTo:output:0tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
tf_op_layer_Mul/Mul�
tf_op_layer_AddV2/AddV2AddV28tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
tf_op_layer_AddV2/AddV2�
+tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_1/Prod_1/reduction_indices�
tf_op_layer_Prod_1/Prod_1Prod$tf_op_layer_Shape_1/Shape_1:output:04tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_1/Prod_1�
tf_op_layer_Reshape/ReshapeReshapetf_op_layer_AddV2/AddV2:z:0"tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:���������2
tf_op_layer_Reshape/Reshape�
%tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex$tf_op_layer_Reshape/Reshape:output:0 tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:���������2'
%tf_op_layer_UnravelIndex/UnravelIndex�
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp�
block2_conv1/Conv2DConv2D8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
block2_conv1/Conv2D�
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp�
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv1/BiasAdd�
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv1/Relu�
$tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$tf_op_layer_Transpose/Transpose/perm�
tf_op_layer_Transpose/Transpose	Transpose.tf_op_layer_UnravelIndex/UnravelIndex:output:0-tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:���������2!
tf_op_layer_Transpose/Transpose�
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp�
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
block2_conv2/Conv2D�
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp�
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv2/BiasAdd�
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv2/Relu�
IdentityIdentityblock2_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity{

Identity_1Identity#tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������:::::::::i e
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
�
L
0__inference_tf_op_layer_Range_layer_call_fn_6668

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_60412
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:���������2

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
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_5997

inputs	
identity	g
Shape_1Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_1W
IdentityIdentityShape_1:output:0*
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
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5876

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
g
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_6041

inputs	
identity	\
Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range/start\
Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range/delta�
RangeRangeRange/start:output:0inputsRange/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:���������2
Range^
IdentityIdentityRange:output:0*
T0	*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
s
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_6116

inputs	
inputs_1	
identity	�
MulMulinputsinputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
Mul~
IdentityIdentityMul:z:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4������������������������������������: :r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
�
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_6603

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
�

�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5920

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
�
|
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_6175

inputs	
inputs_1	
identity	�
UnravelIndexUnravelIndexinputsinputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:���������2
UnravelIndexi
IdentityIdentityUnravelIndex:output:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_input_shapes
:���������::K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
+__inference_block1_conv1_layer_call_fn_5864

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
CPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_58542
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
q
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_6686

inputs	
identity	�
strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/begin�
strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/end�
strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_2/strides�
strided_slice_2StridedSliceinputsstrided_slice_2/begin:output:0strided_slice_2/end:output:0 strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:���������*
ellipsis_mask*
new_axis_mask2
strided_slice_2t
IdentityIdentitystrided_slice_2:output:0*
T0	*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
q
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_6013

inputs	
identity	x
strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/begint
strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/end|
strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/strides�
strided_slice_1StridedSliceinputsstrided_slice_1/begin:output:0strided_slice_1/end:output:0 strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1[
IdentityIdentitystrided_slice_1:output:0*
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
f
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_6102

inputs	
identity	z
Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod/reduction_indicesm
ProdProdinputsProd/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
ProdP
IdentityIdentityProd:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�S
�
?__inference_model_layer_call_and_return_conditional_losses_6365

inputs
block1_conv1_6327
block1_conv1_6329
block1_conv2_6332
block1_conv2_6334
block2_conv1_6352
block2_conv1_6354
block2_conv2_6358
block2_conv2_6360
identity

identity_1	��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�
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
GPU

CPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_59422+
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
GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_59562%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_6327block1_conv1_6329*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_58542&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6332block1_conv2_6334*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_58762&
$block1_conv2/StatefulPartitionedCall�
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+���������������������������@:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_59812/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCall�
#tf_op_layer_Shape_1/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1*
Tin
2	*
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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_59972%
#tf_op_layer_Shape_1/PartitionedCall�
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_60132-
+tf_op_layer_strided_slice_1/PartitionedCall�
!tf_op_layer_Shape/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
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
CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_60262#
!tf_op_layer_Shape/PartitionedCall�
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_60412#
!tf_op_layer_Range/PartitionedCall�
+tf_op_layer_strided_slice_3/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_60572-
+tf_op_layer_strided_slice_3/PartitionedCall�
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_60732-
+tf_op_layer_strided_slice_2/PartitionedCall�
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_60872)
'tf_op_layer_BroadcastTo/PartitionedCall�
 tf_op_layer_Prod/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_3/PartitionedCall:output:0*
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
CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_61022"
 tf_op_layer_Prod/PartitionedCall�
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_61162!
tf_op_layer_Mul/PartitionedCall�
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_61312#
!tf_op_layer_AddV2/PartitionedCall�
"tf_op_layer_Prod_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_61462$
"tf_op_layer_Prod_1/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_61602%
#tf_op_layer_Reshape/PartitionedCall�
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_61752*
(tf_op_layer_UnravelIndex/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_6352block2_conv1_6354*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_58982&
$block2_conv1/StatefulPartitionedCall�
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_61952'
%tf_op_layer_Transpose/PartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6358block2_conv2_6360*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_59202&
$block2_conv2/StatefulPartitionedCall�
IdentityIdentity-block2_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:i e
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
�
h
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_6146

inputs	
identity	~
Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_1/reduction_indices�
Prod_1Prodinputs!Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_1V
IdentityIdentityProd_1:output:0*
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
�
T
8__inference_tf_op_layer_strided_slice_layer_call_fn_6608

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
CPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_59422
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
�
�
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_6626

inputs
identity

identity_1	�
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*
_cloned(*n
_output_shapes\
Z:+���������������������������@:+���������������������������@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax�
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�

Identity_1IdentityMaxPoolWithArgmax:argmax:0*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
q
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_6699

inputs	
identity	x
strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/begint
strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/end|
strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/strides�
strided_slice_3StridedSliceinputsstrided_slice_3/begin:output:0strided_slice_3/end:output:0 strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_3_
IdentityIdentitystrided_slice_3:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_5956

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
�
i
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_6638

inputs	
identity	g
Shape_1Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_1W
IdentityIdentityShape_1:output:0*
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
�
V
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_6704

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_60572
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
g
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_6026

inputs
identity	c
ShapeShapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
ShapeU
IdentityIdentityShape:output:0*
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
$__inference_model_layer_call_fn_6595

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_63652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::22
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
�
f
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_6722

inputs	
identity	z
Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod/reduction_indicesm
ProdProdinputsProd/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
ProdP
IdentityIdentityProd:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�'
�
__inference__traced_save_6849
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop
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
value3B1 B+_temp_1b75f63630074fa28bb7eaf8abdecbad/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop"/device:CPU:0*
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

identity_1Identity_1:output:0*|
_input_shapesk
i: :@:@:@@:@:@�:�:��:�: 2(
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
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:	

_output_shapes
: 
�
^
2__inference_tf_op_layer_Reshape_layer_call_fn_6774
inputs_0	
inputs_1	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_61602
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
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
�*
�
 __inference__traced_restore_6885
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias*
&assignvariableop_4_block2_conv1_kernel(
$assignvariableop_5_block2_conv1_bias*
&assignvariableop_6_block2_conv2_kernel(
$assignvariableop_7_block2_conv2_bias

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
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0*
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
�
~
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_6780
inputs_0	
inputs_1	
identity	�
UnravelIndexUnravelIndexinputs_0inputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:���������2
UnravelIndexi
IdentityIdentityUnravelIndex:output:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_input_shapes
:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
�
$__inference_model_layer_call_fn_6386
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_63652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::22
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
�
q
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_6057

inputs	
identity	x
strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/begint
strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/end|
strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/strides�
strided_slice_3StridedSliceinputsstrided_slice_3/begin:output:0strided_slice_3/end:output:0 strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_3_
IdentityIdentitystrided_slice_3:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
u
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_6131

inputs	
inputs_1	
identity	�
AddV2AddV2inputsinputs_1*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
AddV2w
IdentityIdentity	AddV2:z:0*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+���������������������������@:4������������������������������������:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:rn
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_6572

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_62992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::22
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
�
�
+__inference_block2_conv1_layer_call_fn_5908

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
CPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_58982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

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
\
0__inference_tf_op_layer_AddV2_layer_call_fn_6751
inputs_0	
inputs_1	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_61312
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+���������������������������@:4������������������������������������:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4������������������������������������
"
_user_specified_name
inputs/1
�
g
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_6663

inputs	
identity	\
Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range/start\
Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range/delta�
RangeRangeRange/start:output:0inputsRange/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:���������2
Range^
IdentityIdentityRange:output:0*
T0	*#
_output_shapes
:���������2

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
4__inference_tf_op_layer_Transpose_layer_call_fn_6797

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_61952
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_6411
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__wrapped_model_58422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::22
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
�f
�
?__inference_model_layer_call_and_return_conditional_losses_6549

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource
identity

identity_1	��
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
block1_conv2/Relu�
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmaxblock1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+���������������������������@:+���������������������������@*
ksize
*
paddingSAME*
strides
21
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax�
tf_op_layer_Shape_1/Shape_1Shape8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_1/Shape_1�
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_1/strided_slice_1/begin�
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_1/strided_slice_1/end�
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_1/strided_slice_1/strides�
+tf_op_layer_strided_slice_1/strided_slice_1StridedSlice$tf_op_layer_Shape_1/Shape_1:output:0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_1/strided_slice_1�
tf_op_layer_Shape/ShapeShapeblock1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape/Shape�
tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
tf_op_layer_Range/Range/start�
tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Range/Range/delta�
tf_op_layer_Range/RangeRange&tf_op_layer_Range/Range/start:output:04tf_op_layer_strided_slice_1/strided_slice_1:output:0&tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:���������2
tf_op_layer_Range/Range�
1tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_3/strided_slice_3/begin�
/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_3/strided_slice_3/end�
3tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_3/strided_slice_3/strides�
+tf_op_layer_strided_slice_3/strided_slice_3StridedSlice tf_op_layer_Shape/Shape:output:0:tf_op_layer_strided_slice_3/strided_slice_3/begin:output:08tf_op_layer_strided_slice_3/strided_slice_3/end:output:0<tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_3/strided_slice_3�
1tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_2/strided_slice_2/begin�
/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_2/strided_slice_2/end�
3tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_2/strided_slice_2/strides�
+tf_op_layer_strided_slice_2/strided_slice_2StridedSlice tf_op_layer_Range/Range:output:0:tf_op_layer_strided_slice_2/strided_slice_2/begin:output:08tf_op_layer_strided_slice_2/strided_slice_2/end:output:0<tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:���������*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_2/strided_slice_2�
#tf_op_layer_BroadcastTo/BroadcastToBroadcastTo4tf_op_layer_strided_slice_2/strided_slice_2:output:0$tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2%
#tf_op_layer_BroadcastTo/BroadcastTo�
'tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2)
'tf_op_layer_Prod/Prod/reduction_indices�
tf_op_layer_Prod/ProdProd4tf_op_layer_strided_slice_3/strided_slice_3:output:00tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod/Prod�
tf_op_layer_Mul/MulMul,tf_op_layer_BroadcastTo/BroadcastTo:output:0tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
tf_op_layer_Mul/Mul�
tf_op_layer_AddV2/AddV2AddV28tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
tf_op_layer_AddV2/AddV2�
+tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_1/Prod_1/reduction_indices�
tf_op_layer_Prod_1/Prod_1Prod$tf_op_layer_Shape_1/Shape_1:output:04tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_1/Prod_1�
tf_op_layer_Reshape/ReshapeReshapetf_op_layer_AddV2/AddV2:z:0"tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:���������2
tf_op_layer_Reshape/Reshape�
%tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex$tf_op_layer_Reshape/Reshape:output:0 tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:���������2'
%tf_op_layer_UnravelIndex/UnravelIndex�
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp�
block2_conv1/Conv2DConv2D8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
block2_conv1/Conv2D�
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp�
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv1/BiasAdd�
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv1/Relu�
$tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$tf_op_layer_Transpose/Transpose/perm�
tf_op_layer_Transpose/Transpose	Transpose.tf_op_layer_UnravelIndex/UnravelIndex:output:0-tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:���������2!
tf_op_layer_Transpose/Transpose�
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp�
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
block2_conv2/Conv2D�
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp�
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv2/BiasAdd�
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
block2_conv2/Relu�
IdentityIdentityblock2_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity{

Identity_1Identity#tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������:::::::::i e
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
�
K
/__inference_tf_op_layer_Prod_layer_call_fn_6727

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
CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_61022
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::B >

_output_shapes
:
 
_user_specified_nameinputs
�
w
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_6745
inputs_0	
inputs_1	
identity	�
AddV2AddV2inputs_0inputs_1*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
AddV2w
IdentityIdentity	AddV2:z:0*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+���������������������������@:4������������������������������������:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4������������������������������������
"
_user_specified_name
inputs/1
�
w
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_6160

inputs	
inputs_1	
identity	z
ReshapeReshapeinputsinputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:���������2	
Reshape`
IdentityIdentityReshape:output:0*
T0	*#
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
�
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_6614

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
�
k
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_6195

inputs	
identity	q
Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose/perm�
	Transpose	TransposeinputsTranspose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:���������2
	Transposea
IdentityIdentityTranspose:y:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�o
�
__inference__wrapped_model_5842
input_25
1model_block1_conv1_conv2d_readvariableop_resource6
2model_block1_conv1_biasadd_readvariableop_resource5
1model_block1_conv2_conv2d_readvariableop_resource6
2model_block1_conv2_biasadd_readvariableop_resource5
1model_block2_conv1_conv2d_readvariableop_resource6
2model_block2_conv1_biasadd_readvariableop_resource5
1model_block2_conv2_conv2d_readvariableop_resource6
2model_block2_conv2_biasadd_readvariableop_resource
identity

identity_1	��
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
model/block1_conv2/Relu�
5model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmax%model/block1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+���������������������������@:+���������������������������@*
ksize
*
paddingSAME*
strides
27
5model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax�
!model/tf_op_layer_Shape_1/Shape_1Shape>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_1/Shape_1�
7model/tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/tf_op_layer_strided_slice_1/strided_slice_1/begin�
5model/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:27
5model/tf_op_layer_strided_slice_1/strided_slice_1/end�
9model/tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_1/strided_slice_1/strides�
1model/tf_op_layer_strided_slice_1/strided_slice_1StridedSlice*model/tf_op_layer_Shape_1/Shape_1:output:0@model/tf_op_layer_strided_slice_1/strided_slice_1/begin:output:0>model/tf_op_layer_strided_slice_1/strided_slice_1/end:output:0Bmodel/tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask23
1model/tf_op_layer_strided_slice_1/strided_slice_1�
model/tf_op_layer_Shape/ShapeShape%model/block1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
model/tf_op_layer_Shape/Shape�
#model/tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#model/tf_op_layer_Range/Range/start�
#model/tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2%
#model/tf_op_layer_Range/Range/delta�
model/tf_op_layer_Range/RangeRange,model/tf_op_layer_Range/Range/start:output:0:model/tf_op_layer_strided_slice_1/strided_slice_1:output:0,model/tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:���������2
model/tf_op_layer_Range/Range�
7model/tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7model/tf_op_layer_strided_slice_3/strided_slice_3/begin�
5model/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/tf_op_layer_strided_slice_3/strided_slice_3/end�
9model/tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_3/strided_slice_3/strides�
1model/tf_op_layer_strided_slice_3/strided_slice_3StridedSlice&model/tf_op_layer_Shape/Shape:output:0@model/tf_op_layer_strided_slice_3/strided_slice_3/begin:output:0>model/tf_op_layer_strided_slice_3/strided_slice_3/end:output:0Bmodel/tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask23
1model/tf_op_layer_strided_slice_3/strided_slice_3�
7model/tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                29
7model/tf_op_layer_strided_slice_2/strided_slice_2/begin�
5model/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                27
5model/tf_op_layer_strided_slice_2/strided_slice_2/end�
9model/tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2;
9model/tf_op_layer_strided_slice_2/strided_slice_2/strides�
1model/tf_op_layer_strided_slice_2/strided_slice_2StridedSlice&model/tf_op_layer_Range/Range:output:0@model/tf_op_layer_strided_slice_2/strided_slice_2/begin:output:0>model/tf_op_layer_strided_slice_2/strided_slice_2/end:output:0Bmodel/tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:���������*
ellipsis_mask*
new_axis_mask23
1model/tf_op_layer_strided_slice_2/strided_slice_2�
)model/tf_op_layer_BroadcastTo/BroadcastToBroadcastTo:model/tf_op_layer_strided_slice_2/strided_slice_2:output:0*model/tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2+
)model/tf_op_layer_BroadcastTo/BroadcastTo�
-model/tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2/
-model/tf_op_layer_Prod/Prod/reduction_indices�
model/tf_op_layer_Prod/ProdProd:model/tf_op_layer_strided_slice_3/strided_slice_3:output:06model/tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
model/tf_op_layer_Prod/Prod�
model/tf_op_layer_Mul/MulMul2model/tf_op_layer_BroadcastTo/BroadcastTo:output:0$model/tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
model/tf_op_layer_Mul/Mul�
model/tf_op_layer_AddV2/AddV2AddV2>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0model/tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+���������������������������@2
model/tf_op_layer_AddV2/AddV2�
1model/tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_1/Prod_1/reduction_indices�
model/tf_op_layer_Prod_1/Prod_1Prod*model/tf_op_layer_Shape_1/Shape_1:output:0:model/tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2!
model/tf_op_layer_Prod_1/Prod_1�
!model/tf_op_layer_Reshape/ReshapeReshape!model/tf_op_layer_AddV2/AddV2:z:0(model/tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:���������2#
!model/tf_op_layer_Reshape/Reshape�
+model/tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex*model/tf_op_layer_Reshape/Reshape:output:0&model/tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:���������2-
+model/tf_op_layer_UnravelIndex/UnravelIndex�
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02*
(model/block2_conv1/Conv2D/ReadVariableOp�
model/block2_conv1/Conv2DConv2D>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
model/block2_conv1/Conv2D�
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)model/block2_conv1/BiasAdd/ReadVariableOp�
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
model/block2_conv1/BiasAdd�
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
model/block2_conv1/Relu�
*model/tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model/tf_op_layer_Transpose/Transpose/perm�
%model/tf_op_layer_Transpose/Transpose	Transpose4model/tf_op_layer_UnravelIndex/UnravelIndex:output:03model/tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:���������2'
%model/tf_op_layer_Transpose/Transpose�
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(model/block2_conv2/Conv2D/ReadVariableOp�
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
model/block2_conv2/Conv2D�
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)model/block2_conv2/BiasAdd/ReadVariableOp�
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2
model/block2_conv2/BiasAdd�
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
model/block2_conv2/Relu�
IdentityIdentity%model/block2_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity)model/tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������:::::::::j f
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
c
7__inference_tf_op_layer_UnravelIndex_layer_call_fn_6786
inputs_0	
inputs_1	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_61752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_input_shapes
:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
�
$__inference_model_layer_call_fn_6320
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2	*U
_output_shapesC
A:,����������������������������:���������**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_62992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::22
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
�S
�
?__inference_model_layer_call_and_return_conditional_losses_6253
input_2
block1_conv1_6215
block1_conv1_6217
block1_conv2_6220
block1_conv2_6222
block2_conv1_6240
block2_conv1_6242
block2_conv2_6246
block2_conv2_6248
identity

identity_1	��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�
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
GPU

CPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_59422+
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
GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_59562%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_6215block1_conv1_6217*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_58542&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6220block1_conv2_6222*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_58762&
$block1_conv2/StatefulPartitionedCall�
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+���������������������������@:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_59812/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCall�
#tf_op_layer_Shape_1/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1*
Tin
2	*
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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_59972%
#tf_op_layer_Shape_1/PartitionedCall�
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_60132-
+tf_op_layer_strided_slice_1/PartitionedCall�
!tf_op_layer_Shape/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
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
CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_60262#
!tf_op_layer_Shape/PartitionedCall�
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_60412#
!tf_op_layer_Range/PartitionedCall�
+tf_op_layer_strided_slice_3/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_60572-
+tf_op_layer_strided_slice_3/PartitionedCall�
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_60732-
+tf_op_layer_strided_slice_2/PartitionedCall�
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_60872)
'tf_op_layer_BroadcastTo/PartitionedCall�
 tf_op_layer_Prod/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_3/PartitionedCall:output:0*
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
CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_61022"
 tf_op_layer_Prod/PartitionedCall�
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_61162!
tf_op_layer_Mul/PartitionedCall�
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_61312#
!tf_op_layer_AddV2/PartitionedCall�
"tf_op_layer_Prod_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_61462$
"tf_op_layer_Prod_1/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_61602%
#tf_op_layer_Reshape/PartitionedCall�
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_61752*
(tf_op_layer_UnravelIndex/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_6240block2_conv1_6242*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_58982&
$block2_conv1/StatefulPartitionedCall�
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_61952'
%tf_op_layer_Transpose/PartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6246block2_conv2_6248*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_59202&
$block2_conv2/StatefulPartitionedCall�
IdentityIdentity-block2_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:j f
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
�S
�
?__inference_model_layer_call_and_return_conditional_losses_6299

inputs
block1_conv1_6261
block1_conv1_6263
block1_conv2_6266
block1_conv2_6268
block2_conv1_6286
block2_conv1_6288
block2_conv2_6292
block2_conv2_6294
identity

identity_1	��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�
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
GPU

CPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_59422+
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
GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_59562%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_6261block1_conv1_6263*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_58542&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6266block1_conv2_6268*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_58762&
$block1_conv2/StatefulPartitionedCall�
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+���������������������������@:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_59812/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCall�
#tf_op_layer_Shape_1/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1*
Tin
2	*
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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_59972%
#tf_op_layer_Shape_1/PartitionedCall�
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_60132-
+tf_op_layer_strided_slice_1/PartitionedCall�
!tf_op_layer_Shape/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
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
CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_60262#
!tf_op_layer_Shape/PartitionedCall�
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_60412#
!tf_op_layer_Range/PartitionedCall�
+tf_op_layer_strided_slice_3/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_60572-
+tf_op_layer_strided_slice_3/PartitionedCall�
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_60732-
+tf_op_layer_strided_slice_2/PartitionedCall�
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_60872)
'tf_op_layer_BroadcastTo/PartitionedCall�
 tf_op_layer_Prod/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_3/PartitionedCall:output:0*
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
CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_61022"
 tf_op_layer_Prod/PartitionedCall�
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_61162!
tf_op_layer_Mul/PartitionedCall�
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_61312#
!tf_op_layer_AddV2/PartitionedCall�
"tf_op_layer_Prod_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_61462$
"tf_op_layer_Prod_1/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_61602%
#tf_op_layer_Reshape/PartitionedCall�
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_61752*
(tf_op_layer_UnravelIndex/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_6286block2_conv1_6288*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_58982&
$block2_conv1/StatefulPartitionedCall�
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_61952'
%tf_op_layer_Transpose/PartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6292block2_conv2_6294*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_59202&
$block2_conv2/StatefulPartitionedCall�
IdentityIdentity-block2_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:i e
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
�
q
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_6651

inputs	
identity	x
strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/begint
strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/end|
strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/strides�
strided_slice_1StridedSliceinputsstrided_slice_1/begin:output:0strided_slice_1/end:output:0 strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1[
IdentityIdentitystrided_slice_1:output:0*
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
�
b
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_6716
inputs_0	
inputs_1	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_60872
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:���������::Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
q
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_6073

inputs	
identity	�
strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/begin�
strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/end�
strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_2/strides�
strided_slice_2StridedSliceinputsstrided_slice_2/begin:output:0strided_slice_2/end:output:0 strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:���������*
ellipsis_mask*
new_axis_mask2
strided_slice_2t
IdentityIdentitystrided_slice_2:output:0*
T0	*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
y
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_6768
inputs_0	
inputs_1	
identity	|
ReshapeReshapeinputs_0inputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:���������2	
Reshape`
IdentityIdentityReshape:output:0*
T0	*#
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
�
Z
.__inference_tf_op_layer_Mul_layer_call_fn_6739
inputs_0	
inputs_1	
identity	�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_61162
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4������������������������������������: :t p
J
_output_shapes8
6:4������������������������������������
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
�
h
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_6757

inputs	
identity	~
Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_1/reduction_indices�
Prod_1Prodinputs!Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_1V
IdentityIdentityProd_1:output:0*
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
�
�
+__inference_block2_conv2_layer_call_fn_5930

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
CPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_59202
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
�
L
0__inference_tf_op_layer_Shape_layer_call_fn_6678

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
CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_60262
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
�
k
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_6792

inputs	
identity	q
Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose/perm�
	Transpose	TransposeinputsTranspose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:���������2
	Transposea
IdentityIdentityTranspose:y:0*
T0	*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_6633

inputs
identity

identity_1	�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+���������������������������@:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_59812
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�

Identity_1IdentityPartitionedCall:output:1*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
N
2__inference_tf_op_layer_Shape_1_layer_call_fn_6643

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_59972
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
�

�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5854

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
�
�
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_5981

inputs
identity

identity_1	�
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*
_cloned(*n
_output_shapes\
Z:+���������������������������@:+���������������������������@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax�
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�

Identity_1IdentityMaxPoolWithArgmax:argmax:0*
T0	*A
_output_shapes/
-:+���������������������������@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
}
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_6710
inputs_0	
inputs_1	
identity	�
BroadcastToBroadcastToinputs_0inputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
BroadcastTo�
IdentityIdentityBroadcastTo:output:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:���������::Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
�
g
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_6673

inputs
identity	c
ShapeShapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
ShapeU
IdentityIdentityShape:output:0*
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
�S
�
?__inference_model_layer_call_and_return_conditional_losses_6210
input_2
block1_conv1_5964
block1_conv1_5966
block1_conv2_5969
block1_conv2_5971
block2_conv1_6184
block2_conv1_6186
block2_conv2_6203
block2_conv2_6205
identity

identity_1	��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�
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
GPU

CPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_59422+
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
GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_59562%
#tf_op_layer_BiasAdd/PartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_5964block1_conv1_5966*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_58542&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_5969block1_conv2_5971*
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
CPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_58762&
$block1_conv2/StatefulPartitionedCall�
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+���������������������������@:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_59812/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCall�
#tf_op_layer_Shape_1/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1*
Tin
2	*
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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_59972%
#tf_op_layer_Shape_1/PartitionedCall�
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_60132-
+tf_op_layer_strided_slice_1/PartitionedCall�
!tf_op_layer_Shape/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
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
CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_60262#
!tf_op_layer_Shape/PartitionedCall�
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_60412#
!tf_op_layer_Range/PartitionedCall�
+tf_op_layer_strided_slice_3/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2	*
Tout
2	*
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_60572-
+tf_op_layer_strided_slice_3/PartitionedCall�
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_60732-
+tf_op_layer_strided_slice_2/PartitionedCall�
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_60872)
'tf_op_layer_BroadcastTo/PartitionedCall�
 tf_op_layer_Prod/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_3/PartitionedCall:output:0*
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
CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_61022"
 tf_op_layer_Prod/PartitionedCall�
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_61162!
tf_op_layer_Mul/PartitionedCall�
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_61312#
!tf_op_layer_AddV2/PartitionedCall�
"tf_op_layer_Prod_1/PartitionedCallPartitionedCall,tf_op_layer_Shape_1/PartitionedCall:output:0*
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
CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_61462$
"tf_op_layer_Prod_1/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_61602%
#tf_op_layer_Reshape/PartitionedCall�
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_61752*
(tf_op_layer_UnravelIndex/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_6184block2_conv1_6186*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_58982&
$block2_conv1/StatefulPartitionedCall�
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_61952'
%tf_op_layer_Transpose/PartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6203block2_conv2_6205*
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
CPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_59202&
$block2_conv2/StatefulPartitionedCall�
IdentityIdentity-block2_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity�

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*`
_input_shapesO
M:+���������������������������::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:j f
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
{
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_6087

inputs	
inputs_1	
identity	�
BroadcastToBroadcastToinputsinputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4������������������������������������2
BroadcastTo�
IdentityIdentityBroadcastTo:output:0*
T0	*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:���������::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
M
1__inference_tf_op_layer_Prod_1_layer_call_fn_6762

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
CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_61462
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
�
N
2__inference_tf_op_layer_BiasAdd_layer_call_fn_6619

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
CPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_59562
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
�
V
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_6691

inputs	
identity	�
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*/
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_60732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0	*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5898

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
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
+__inference_block1_conv2_layer_call_fn_5886

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
CPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_58762
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
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
input_2J
serving_default_input_2:0+���������������������������[
block2_conv2K
StatefulPartitionedCall:0,����������������������������I
tf_op_layer_Transpose0
StatefulPartitionedCall:1	���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
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
layer-14
layer-15
layer-16
layer-17
layer_with_weights-2
layer-18
layer-19
layer_with_weights-3
layer-20
layer-21
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"��
_tf_keras_model��{"class_name": "Model", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "Targmax": {"type": "DT_INT64"}, "padding": {"s": "U0FNRQ=="}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "14"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_2", "inbound_nodes": [[["tf_op_layer_Range", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_3", "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo", "inbound_nodes": [[["tf_op_layer_strided_slice_2", 0, 0, {}], ["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"keep_dims": {"b": false}, "T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod", "inbound_nodes": [[["tf_op_layer_strided_slice_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_BroadcastTo", 0, 0, {}], ["tf_op_layer_Prod", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}], ["tf_op_layer_Mul", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}], ["tf_op_layer_Prod_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}], ["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose", "inbound_nodes": [[["tf_op_layer_UnravelIndex", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_conv2", 0, 0], ["tf_op_layer_Transpose", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "Targmax": {"type": "DT_INT64"}, "padding": {"s": "U0FNRQ=="}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "14"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_2", "inbound_nodes": [[["tf_op_layer_Range", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_3", "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo", "inbound_nodes": [[["tf_op_layer_strided_slice_2", 0, 0, {}], ["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"keep_dims": {"b": false}, "T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod", "inbound_nodes": [[["tf_op_layer_strided_slice_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_BroadcastTo", 0, 0, {}], ["tf_op_layer_Prod", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}], ["tf_op_layer_Mul", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}], ["tf_op_layer_Prod_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}], ["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose", "inbound_nodes": [[["tf_op_layer_UnravelIndex", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_conv2", 0, 0], ["tf_op_layer_Transpose", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "2"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}}
�
 trainable_variables
!regularization_losses
"	variables
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BiasAdd", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"data_format": {"s": "TkhXQw=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}}
�	

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
�	

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
�
0trainable_variables
1regularization_losses
2	variables
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPoolWithArgmax", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "Targmax": {"type": "DT_INT64"}, "padding": {"s": "U0FNRQ=="}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}}}, "constants": {}}}
�
4trainable_variables
5regularization_losses
6	variables
7	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
�
8trainable_variables
9regularization_losses
:	variables
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "1"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
�
<trainable_variables
=regularization_losses
>	variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Range", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}}
�
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
�
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "14"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}}
�
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}}
�
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BroadcastTo", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
�
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"keep_dims": {"b": false}, "T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}}
�
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
�
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
�
\trainable_variables
]regularization_losses
^	variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}}
�
`trainable_variables
aregularization_losses
b	variables
c	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}}
�	

dkernel
ebias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
�
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_UnravelIndex", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
�	

nkernel
obias
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
�
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
$0
%1
*2
+3
d4
e5
n6
o7"
trackable_list_wrapper
�
trainable_variables

xlayers
ylayer_metrics
zmetrics
{non_trainable_variables
regularization_losses
	variables
|layer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables

}layers
~layer_metrics
metrics
�non_trainable_variables
regularization_losses
	variables
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
 trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
!regularization_losses
"	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
&trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
'regularization_losses
(	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
,trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
-regularization_losses
.	variables
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
0trainable_variables
�layers
�layer_metrics
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ltrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Mregularization_losses
N	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ptrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
Qregularization_losses
R	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
`trainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
aregularization_losses
b	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,@�2block2_conv1/kernel
 :�2block2_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
�
ftrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
gregularization_losses
h	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
jtrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
kregularization_losses
l	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/:-��2block2_conv2/kernel
 :�2block2_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
�
ptrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
qregularization_losses
r	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ttrainable_variables
�layers
�layer_metrics
�metrics
�non_trainable_variables
uregularization_losses
v	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
$0
%1
*2
+3
d4
e5
n6
o7"
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
$0
%1"
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
*0
+1"
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
.
d0
e1"
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
n0
o1"
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
$__inference_model_layer_call_fn_6320
$__inference_model_layer_call_fn_6572
$__inference_model_layer_call_fn_6386
$__inference_model_layer_call_fn_6595�
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
__inference__wrapped_model_5842�
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
�2�
?__inference_model_layer_call_and_return_conditional_losses_6549
?__inference_model_layer_call_and_return_conditional_losses_6480
?__inference_model_layer_call_and_return_conditional_losses_6253
?__inference_model_layer_call_and_return_conditional_losses_6210�
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
8__inference_tf_op_layer_strided_slice_layer_call_fn_6608�
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
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_6603�
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
2__inference_tf_op_layer_BiasAdd_layer_call_fn_6619�
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
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_6614�
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
+__inference_block1_conv1_layer_call_fn_5864�
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
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5854�
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
+__inference_block1_conv2_layer_call_fn_5886�
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
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5876�
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
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_6633�
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
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_6626�
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
2__inference_tf_op_layer_Shape_1_layer_call_fn_6643�
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
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_6638�
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
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_6656�
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
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_6651�
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
0__inference_tf_op_layer_Range_layer_call_fn_6668�
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
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_6663�
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
0__inference_tf_op_layer_Shape_layer_call_fn_6678�
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
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_6673�
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
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_6691�
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
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_6686�
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
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_6704�
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
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_6699�
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
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_6716�
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
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_6710�
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
/__inference_tf_op_layer_Prod_layer_call_fn_6727�
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
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_6722�
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
.__inference_tf_op_layer_Mul_layer_call_fn_6739�
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
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_6733�
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
0__inference_tf_op_layer_AddV2_layer_call_fn_6751�
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
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_6745�
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
1__inference_tf_op_layer_Prod_1_layer_call_fn_6762�
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
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_6757�
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
2__inference_tf_op_layer_Reshape_layer_call_fn_6774�
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
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_6768�
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
+__inference_block2_conv1_layer_call_fn_5908�
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
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5898�
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
7__inference_tf_op_layer_UnravelIndex_layer_call_fn_6786�
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
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_6780�
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
+__inference_block2_conv2_layer_call_fn_5930�
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
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5920�
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
4__inference_tf_op_layer_Transpose_layer_call_fn_6797�
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
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_6792�
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
1B/
"__inference_signature_wrapper_6411input_2�
__inference__wrapped_model_5842�$%*+denoJ�G
@�=
;�8
input_2+���������������������������
� "���
Q
block2_conv2A�>
block2_conv2,����������������������������
H
tf_op_layer_Transpose/�,
tf_op_layer_Transpose���������	�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_5854�$%I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������@
� �
+__inference_block1_conv1_layer_call_fn_5864�$%I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������@�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_5876�*+I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
+__inference_block1_conv2_layer_call_fn_5886�*+I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_5898�deI�F
?�<
:�7
inputs+���������������������������@
� "@�=
6�3
0,����������������������������
� �
+__inference_block2_conv1_layer_call_fn_5908�deI�F
?�<
:�7
inputs+���������������������������@
� "3�0,�����������������������������
F__inference_block2_conv2_layer_call_and_return_conditional_losses_5920�noJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
+__inference_block2_conv2_layer_call_fn_5930�noJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
?__inference_model_layer_call_and_return_conditional_losses_6210�$%*+denoR�O
H�E
;�8
input_2+���������������������������
p

 
� "f�c
\�Y
8�5
0/0,����������������������������
�
0/1���������	
� �
?__inference_model_layer_call_and_return_conditional_losses_6253�$%*+denoR�O
H�E
;�8
input_2+���������������������������
p 

 
� "f�c
\�Y
8�5
0/0,����������������������������
�
0/1���������	
� �
?__inference_model_layer_call_and_return_conditional_losses_6480�$%*+denoQ�N
G�D
:�7
inputs+���������������������������
p

 
� "f�c
\�Y
8�5
0/0,����������������������������
�
0/1���������	
� �
?__inference_model_layer_call_and_return_conditional_losses_6549�$%*+denoQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "f�c
\�Y
8�5
0/0,����������������������������
�
0/1���������	
� �
$__inference_model_layer_call_fn_6320�$%*+denoR�O
H�E
;�8
input_2+���������������������������
p

 
� "X�U
6�3
0,����������������������������
�
1���������	�
$__inference_model_layer_call_fn_6386�$%*+denoR�O
H�E
;�8
input_2+���������������������������
p 

 
� "X�U
6�3
0,����������������������������
�
1���������	�
$__inference_model_layer_call_fn_6572�$%*+denoQ�N
G�D
:�7
inputs+���������������������������
p

 
� "X�U
6�3
0,����������������������������
�
1���������	�
$__inference_model_layer_call_fn_6595�$%*+denoQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "X�U
6�3
0,����������������������������
�
1���������	�
"__inference_signature_wrapper_6411�$%*+denoU�R
� 
K�H
F
input_2;�8
input_2+���������������������������"���
Q
block2_conv2A�>
block2_conv2,����������������������������
H
tf_op_layer_Transpose/�,
tf_op_layer_Transpose���������	�
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_6745����
���
���
<�9
inputs/0+���������������������������@	
E�B
inputs/14������������������������������������	
� "?�<
5�2
0+���������������������������@	
� �
0__inference_tf_op_layer_AddV2_layer_call_fn_6751����
���
���
<�9
inputs/0+���������������������������@	
E�B
inputs/14������������������������������������	
� "2�/+���������������������������@	�
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_6614�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
2__inference_tf_op_layer_BiasAdd_layer_call_fn_6619I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_6710�U�R
K�H
F�C
*�'
inputs/0���������	
�
inputs/1	
� "H�E
>�;
04������������������������������������	
� �
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_6716�U�R
K�H
F�C
*�'
inputs/0���������	
�
inputs/1	
� ";�84������������������������������������	�
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_6626�I�F
?�<
:�7
inputs+���������������������������@
� "�|
u�r
7�4
0/0+���������������������������@
7�4
0/1+���������������������������@	
� �
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_6633�I�F
?�<
:�7
inputs+���������������������������@
� "q�n
5�2
0+���������������������������@
5�2
1+���������������������������@	�
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_6733�l�i
b�_
]�Z
E�B
inputs/04������������������������������������	
�
inputs/1 	
� "H�E
>�;
04������������������������������������	
� �
.__inference_tf_op_layer_Mul_layer_call_fn_6739�l�i
b�_
]�Z
E�B
inputs/04������������������������������������	
�
inputs/1 	
� ";�84������������������������������������	�
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_6757>"�
�
�
inputs	
� "�
�
0	
� f
1__inference_tf_op_layer_Prod_1_layer_call_fn_67621"�
�
�
inputs	
� "�	�
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_6722:"�
�
�
inputs	
� "�

�
0 	
� `
/__inference_tf_op_layer_Prod_layer_call_fn_6727-"�
�
�
inputs	
� "� 	�
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_6663C�
�
�
inputs 	
� "!�
�
0���������	
� j
0__inference_tf_op_layer_Range_layer_call_fn_66686�
�
�
inputs 	
� "����������	�
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_6768�g�d
]�Z
X�U
<�9
inputs/0+���������������������������@	
�
inputs/1	
� "!�
�
0���������	
� �
2__inference_tf_op_layer_Reshape_layer_call_fn_6774g�d
]�Z
X�U
<�9
inputs/0+���������������������������@	
�
inputs/1	
� "����������	�
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_6638eI�F
?�<
:�7
inputs+���������������������������@	
� "�
�
0	
� �
2__inference_tf_op_layer_Shape_1_layer_call_fn_6643XI�F
?�<
:�7
inputs+���������������������������@	
� "�	�
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_6673eI�F
?�<
:�7
inputs+���������������������������@
� "�
�
0	
� �
0__inference_tf_op_layer_Shape_layer_call_fn_6678XI�F
?�<
:�7
inputs+���������������������������@
� "�	�
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_6792X/�,
%�"
 �
inputs���������	
� "%�"
�
0���������	
� �
4__inference_tf_op_layer_Transpose_layer_call_fn_6797K/�,
%�"
 �
inputs���������	
� "����������	�
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_6780rI�F
?�<
:�7
�
inputs/0���������	
�
inputs/1	
� "%�"
�
0���������	
� �
7__inference_tf_op_layer_UnravelIndex_layer_call_fn_6786eI�F
?�<
:�7
�
inputs/0���������	
�
inputs/1	
� "����������	�
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_6651:"�
�
�
inputs	
� "�

�
0 	
� k
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_6656-"�
�
�
inputs	
� "� 	�
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_6686\+�(
!�
�
inputs���������	
� "-�*
#� 
0���������	
� �
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_6691O+�(
!�
�
inputs���������	
� " ����������	�
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_6699>"�
�
�
inputs	
� "�
�
0	
� o
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_67041"�
�
�
inputs	
� "�	�
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_6603�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
8__inference_tf_op_layer_strided_slice_layer_call_fn_6608I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������