йЊ1
Њ§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8Г!

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

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

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

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

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv4/kernel

'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
јл
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Вл
valueЇлBЃл Bл
ў
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer_with_weights-12
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer_with_weights-13
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
signatures
 
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
R
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
h

[kernel
\bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
h

akernel
bbias
c	variables
dregularization_losses
etrainable_variables
f	keras_api
R
g	variables
hregularization_losses
itrainable_variables
j	keras_api
h

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
h

qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
R
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
i

{kernel
|bias
}	variables
~regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
n
Ѓkernel
	Єbias
Ѕ	variables
Іregularization_losses
Їtrainable_variables
Ј	keras_api
n
Љkernel
	Њbias
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
V
Џ	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
V
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
V
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
V
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
V
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
V
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
V
Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
V
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
V
Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
V
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
V
з	variables
иregularization_losses
йtrainable_variables
к	keras_api
V
л	variables
мregularization_losses
нtrainable_variables
о	keras_api
V
п	variables
рregularization_losses
сtrainable_variables
т	keras_api
V
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
V
ч	variables
шregularization_losses
щtrainable_variables
ъ	keras_api
V
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
V
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
V
ѓ	variables
єregularization_losses
ѕtrainable_variables
і	keras_api
V
ї	variables
јregularization_losses
љtrainable_variables
њ	keras_api
V
ћ	variables
ќregularization_losses
§trainable_variables
ў	keras_api
V
џ	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
V
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
V
Ї	variables
Јregularization_losses
Љtrainable_variables
Њ	keras_api
V
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
V
Џ	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
V
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
V
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
V
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
V
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
V
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
V
Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
V
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
V
Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
V
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
V
з	variables
иregularization_losses
йtrainable_variables
к	keras_api
V
л	variables
мregularization_losses
нtrainable_variables
о	keras_api
V
п	variables
рregularization_losses
сtrainable_variables
т	keras_api
V
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
V
ч	variables
шregularization_losses
щtrainable_variables
ъ	keras_api
V
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
V
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
n
ѓkernel
	єbias
ѕ	variables
іregularization_losses
їtrainable_variables
ј	keras_api
V
љ	variables
њregularization_losses
ћtrainable_variables
ќ	keras_api
V
§	variables
ўregularization_losses
џtrainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
 
 
ш
[0
\1
a2
b3
k4
l5
q6
r7
{8
|9
10
11
12
13
14
15
16
17
18
19
Ѓ20
Є21
Љ22
Њ23
ѓ24
є25
26
27
В
layer_metrics
 layers
 Ёlayer_regularization_losses
Nregularization_losses
Ђnon_trainable_variables
Otrainable_variables
P	variables
Ѓmetrics
 
 
 
 
В
Єlayer_metrics
Ѕlayers
S	variables
 Іlayer_regularization_losses
Tregularization_losses
Utrainable_variables
Їnon_trainable_variables
Јmetrics
 
 
 
В
Љlayer_metrics
Њlayers
W	variables
 Ћlayer_regularization_losses
Xregularization_losses
Ytrainable_variables
Ќnon_trainable_variables
­metrics
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 
 
В
Ўlayer_metrics
Џlayers
]	variables
 Аlayer_regularization_losses
^regularization_losses
_trainable_variables
Бnon_trainable_variables
Вmetrics
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

a0
b1
 
 
В
Гlayer_metrics
Дlayers
c	variables
 Еlayer_regularization_losses
dregularization_losses
etrainable_variables
Жnon_trainable_variables
Зmetrics
 
 
 
В
Иlayer_metrics
Йlayers
g	variables
 Кlayer_regularization_losses
hregularization_losses
itrainable_variables
Лnon_trainable_variables
Мmetrics
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
 
 
В
Нlayer_metrics
Оlayers
m	variables
 Пlayer_regularization_losses
nregularization_losses
otrainable_variables
Рnon_trainable_variables
Сmetrics
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
 
 
В
Тlayer_metrics
Уlayers
s	variables
 Фlayer_regularization_losses
tregularization_losses
utrainable_variables
Хnon_trainable_variables
Цmetrics
 
 
 
В
Чlayer_metrics
Шlayers
w	variables
 Щlayer_regularization_losses
xregularization_losses
ytrainable_variables
Ъnon_trainable_variables
Ыmetrics
_]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
 
 
В
Ьlayer_metrics
Эlayers
}	variables
 Юlayer_regularization_losses
~regularization_losses
trainable_variables
Яnon_trainable_variables
аmetrics
_]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
бlayer_metrics
вlayers
	variables
 гlayer_regularization_losses
regularization_losses
trainable_variables
дnon_trainable_variables
еmetrics
_]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
жlayer_metrics
зlayers
	variables
 иlayer_regularization_losses
regularization_losses
trainable_variables
йnon_trainable_variables
кmetrics
_]
VARIABLE_VALUEblock3_conv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
лlayer_metrics
мlayers
	variables
 нlayer_regularization_losses
regularization_losses
trainable_variables
оnon_trainable_variables
пmetrics
 
 
 
Е
рlayer_metrics
сlayers
	variables
 тlayer_regularization_losses
regularization_losses
trainable_variables
уnon_trainable_variables
фmetrics
_]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
хlayer_metrics
цlayers
	variables
 чlayer_regularization_losses
regularization_losses
trainable_variables
шnon_trainable_variables
щmetrics
_]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
ъlayer_metrics
ыlayers
	variables
 ьlayer_regularization_losses
 regularization_losses
Ёtrainable_variables
эnon_trainable_variables
юmetrics
`^
VARIABLE_VALUEblock4_conv3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Ѓ0
Є1
 
 
Е
яlayer_metrics
№layers
Ѕ	variables
 ёlayer_regularization_losses
Іregularization_losses
Їtrainable_variables
ђnon_trainable_variables
ѓmetrics
`^
VARIABLE_VALUEblock4_conv4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Љ0
Њ1
 
 
Е
єlayer_metrics
ѕlayers
Ћ	variables
 іlayer_regularization_losses
Ќregularization_losses
­trainable_variables
їnon_trainable_variables
јmetrics
 
 
 
Е
љlayer_metrics
њlayers
Џ	variables
 ћlayer_regularization_losses
Аregularization_losses
Бtrainable_variables
ќnon_trainable_variables
§metrics
 
 
 
Е
ўlayer_metrics
џlayers
Г	variables
 layer_regularization_losses
Дregularization_losses
Еtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
З	variables
 layer_regularization_losses
Иregularization_losses
Йtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Л	variables
 layer_regularization_losses
Мregularization_losses
Нtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
П	variables
 layer_regularization_losses
Рregularization_losses
Сtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
У	variables
 layer_regularization_losses
Фregularization_losses
Хtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Ч	variables
 layer_regularization_losses
Шregularization_losses
Щtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Ы	variables
 layer_regularization_losses
Ьregularization_losses
Эtrainable_variables
non_trainable_variables
 metrics
 
 
 
Е
Ёlayer_metrics
Ђlayers
Я	variables
 Ѓlayer_regularization_losses
аregularization_losses
бtrainable_variables
Єnon_trainable_variables
Ѕmetrics
 
 
 
Е
Іlayer_metrics
Їlayers
г	variables
 Јlayer_regularization_losses
дregularization_losses
еtrainable_variables
Љnon_trainable_variables
Њmetrics
 
 
 
Е
Ћlayer_metrics
Ќlayers
з	variables
 ­layer_regularization_losses
иregularization_losses
йtrainable_variables
Ўnon_trainable_variables
Џmetrics
 
 
 
Е
Аlayer_metrics
Бlayers
л	variables
 Вlayer_regularization_losses
мregularization_losses
нtrainable_variables
Гnon_trainable_variables
Дmetrics
 
 
 
Е
Еlayer_metrics
Жlayers
п	variables
 Зlayer_regularization_losses
рregularization_losses
сtrainable_variables
Иnon_trainable_variables
Йmetrics
 
 
 
Е
Кlayer_metrics
Лlayers
у	variables
 Мlayer_regularization_losses
фregularization_losses
хtrainable_variables
Нnon_trainable_variables
Оmetrics
 
 
 
Е
Пlayer_metrics
Рlayers
ч	variables
 Сlayer_regularization_losses
шregularization_losses
щtrainable_variables
Тnon_trainable_variables
Уmetrics
 
 
 
Е
Фlayer_metrics
Хlayers
ы	variables
 Цlayer_regularization_losses
ьregularization_losses
эtrainable_variables
Чnon_trainable_variables
Шmetrics
 
 
 
Е
Щlayer_metrics
Ъlayers
я	variables
 Ыlayer_regularization_losses
№regularization_losses
ёtrainable_variables
Ьnon_trainable_variables
Эmetrics
 
 
 
Е
Юlayer_metrics
Яlayers
ѓ	variables
 аlayer_regularization_losses
єregularization_losses
ѕtrainable_variables
бnon_trainable_variables
вmetrics
 
 
 
Е
гlayer_metrics
дlayers
ї	variables
 еlayer_regularization_losses
јregularization_losses
љtrainable_variables
жnon_trainable_variables
зmetrics
 
 
 
Е
иlayer_metrics
йlayers
ћ	variables
 кlayer_regularization_losses
ќregularization_losses
§trainable_variables
лnon_trainable_variables
мmetrics
 
 
 
Е
нlayer_metrics
оlayers
џ	variables
 пlayer_regularization_losses
regularization_losses
trainable_variables
рnon_trainable_variables
сmetrics
 
 
 
Е
тlayer_metrics
уlayers
	variables
 фlayer_regularization_losses
regularization_losses
trainable_variables
хnon_trainable_variables
цmetrics
 
 
 
Е
чlayer_metrics
шlayers
	variables
 щlayer_regularization_losses
regularization_losses
trainable_variables
ъnon_trainable_variables
ыmetrics
 
 
 
Е
ьlayer_metrics
эlayers
	variables
 юlayer_regularization_losses
regularization_losses
trainable_variables
яnon_trainable_variables
№metrics
 
 
 
Е
ёlayer_metrics
ђlayers
	variables
 ѓlayer_regularization_losses
regularization_losses
trainable_variables
єnon_trainable_variables
ѕmetrics
 
 
 
Е
іlayer_metrics
їlayers
	variables
 јlayer_regularization_losses
regularization_losses
trainable_variables
љnon_trainable_variables
њmetrics
 
 
 
Е
ћlayer_metrics
ќlayers
	variables
 §layer_regularization_losses
regularization_losses
trainable_variables
ўnon_trainable_variables
џmetrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
 regularization_losses
Ёtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Ѓ	variables
 layer_regularization_losses
Єregularization_losses
Ѕtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Ї	variables
 layer_regularization_losses
Јregularization_losses
Љtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Ћ	variables
 layer_regularization_losses
Ќregularization_losses
­trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Џ	variables
 layer_regularization_losses
Аregularization_losses
Бtrainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
Г	variables
  layer_regularization_losses
Дregularization_losses
Еtrainable_variables
Ёnon_trainable_variables
Ђmetrics
 
 
 
Е
Ѓlayer_metrics
Єlayers
З	variables
 Ѕlayer_regularization_losses
Иregularization_losses
Йtrainable_variables
Іnon_trainable_variables
Їmetrics
 
 
 
Е
Јlayer_metrics
Љlayers
Л	variables
 Њlayer_regularization_losses
Мregularization_losses
Нtrainable_variables
Ћnon_trainable_variables
Ќmetrics
 
 
 
Е
­layer_metrics
Ўlayers
П	variables
 Џlayer_regularization_losses
Рregularization_losses
Сtrainable_variables
Аnon_trainable_variables
Бmetrics
 
 
 
Е
Вlayer_metrics
Гlayers
У	variables
 Дlayer_regularization_losses
Фregularization_losses
Хtrainable_variables
Еnon_trainable_variables
Жmetrics
 
 
 
Е
Зlayer_metrics
Иlayers
Ч	variables
 Йlayer_regularization_losses
Шregularization_losses
Щtrainable_variables
Кnon_trainable_variables
Лmetrics
 
 
 
Е
Мlayer_metrics
Нlayers
Ы	variables
 Оlayer_regularization_losses
Ьregularization_losses
Эtrainable_variables
Пnon_trainable_variables
Рmetrics
 
 
 
Е
Сlayer_metrics
Тlayers
Я	variables
 Уlayer_regularization_losses
аregularization_losses
бtrainable_variables
Фnon_trainable_variables
Хmetrics
 
 
 
Е
Цlayer_metrics
Чlayers
г	variables
 Шlayer_regularization_losses
дregularization_losses
еtrainable_variables
Щnon_trainable_variables
Ъmetrics
 
 
 
Е
Ыlayer_metrics
Ьlayers
з	variables
 Эlayer_regularization_losses
иregularization_losses
йtrainable_variables
Юnon_trainable_variables
Яmetrics
 
 
 
Е
аlayer_metrics
бlayers
л	variables
 вlayer_regularization_losses
мregularization_losses
нtrainable_variables
гnon_trainable_variables
дmetrics
 
 
 
Е
еlayer_metrics
жlayers
п	variables
 зlayer_regularization_losses
рregularization_losses
сtrainable_variables
иnon_trainable_variables
йmetrics
 
 
 
Е
кlayer_metrics
лlayers
у	variables
 мlayer_regularization_losses
фregularization_losses
хtrainable_variables
нnon_trainable_variables
оmetrics
 
 
 
Е
пlayer_metrics
рlayers
ч	variables
 сlayer_regularization_losses
шregularization_losses
щtrainable_variables
тnon_trainable_variables
уmetrics
 
 
 
Е
фlayer_metrics
хlayers
ы	variables
 цlayer_regularization_losses
ьregularization_losses
эtrainable_variables
чnon_trainable_variables
шmetrics
 
 
 
Е
щlayer_metrics
ъlayers
я	variables
 ыlayer_regularization_losses
№regularization_losses
ёtrainable_variables
ьnon_trainable_variables
эmetrics
`^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

ѓ0
є1
 
 
Е
юlayer_metrics
яlayers
ѕ	variables
 №layer_regularization_losses
іregularization_losses
їtrainable_variables
ёnon_trainable_variables
ђmetrics
 
 
 
Е
ѓlayer_metrics
єlayers
љ	variables
 ѕlayer_regularization_losses
њregularization_losses
ћtrainable_variables
іnon_trainable_variables
їmetrics
 
 
 
Е
јlayer_metrics
љlayers
§	variables
 њlayer_regularization_losses
ўregularization_losses
џtrainable_variables
ћnon_trainable_variables
ќmetrics
 
 
 
Е
§layer_metrics
ўlayers
	variables
 џlayer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
`^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
 
 
Е
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
о
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
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
 
ш
[0
\1
a2
b3
k4
l5
q6
r7
{8
|9
10
11
12
13
14
15
16
17
18
19
Ѓ20
Є21
Љ22
Њ23
ѓ24
є25
26
27
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
[0
\1
 
 
 
 

a0
b1
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
k0
l1
 
 
 
 

q0
r1
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
{0
|1
 
 
 
 

0
1
 
 
 
 

0
1
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 

0
1
 
 
 
 

Ѓ0
Є1
 
 
 
 

Љ0
Њ1
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

ѓ0
є1
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

0
1
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
Ў
serving_default_input_2Placeholder*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*6
shape-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/bias*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_signature_wrapper_8932
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ї

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOpConst*)
Tin"
 2*
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
GPU2*0J 8*'
f"R 
__inference__traced_save_10385
Т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/bias*(
Tin!
2*
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
GPU2*0J 8**
f%R#
!__inference__traced_restore_10481у
ЃГ
Ф

?__inference_model_layer_call_and_return_conditional_losses_8794

inputs
block1_conv1_8655
block1_conv1_8657
block1_conv2_8660
block1_conv2_8662
block2_conv1_8667
block2_conv1_8669
block2_conv2_8672
block2_conv2_8674
block3_conv1_8679
block3_conv1_8681
block3_conv2_8684
block3_conv2_8686
block3_conv3_8689
block3_conv3_8691
block3_conv4_8694
block3_conv4_8696
block4_conv1_8701
block4_conv1_8703
block4_conv2_8706
block4_conv2_8708
block4_conv3_8711
block4_conv3_8713
block4_conv4_8716
block4_conv4_8718
block5_conv1_8775
block5_conv1_8777
block5_conv2_8784
block5_conv2_8786
identity

identity_1	

identity_2	

identity_3	

identity_4	Ђ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block3_conv4/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block4_conv4/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCall
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_72972+
)tf_op_layer_strided_slice/PartitionedCall
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_73112%
#tf_op_layer_BiasAdd/PartitionedCallУ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_8655block1_conv1_8657*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_69892&
$block1_conv1/StatefulPartitionedCallФ
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8660block1_conv2_8662*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_70112&
$block1_conv2/StatefulPartitionedCallс
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_73362/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallЮ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_8667block2_conv1_8669*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_70332&
$block2_conv1/StatefulPartitionedCallХ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8672block2_conv2_8674*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_70552&
$block2_conv2/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_736421
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallа
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:0block3_conv1_8679block3_conv1_8681*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_70772&
$block3_conv1/StatefulPartitionedCallХ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8684block3_conv2_8686*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_70992&
$block3_conv2/StatefulPartitionedCallХ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8689block3_conv3_8691*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_71212&
$block3_conv3/StatefulPartitionedCallХ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8694block3_conv4_8696*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_71432&
$block3_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_740221
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallа
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:0block4_conv1_8701block4_conv1_8703*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_71652&
$block4_conv1/StatefulPartitionedCallХ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_8706block4_conv2_8708*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_71872&
$block4_conv2/StatefulPartitionedCallХ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_8711block4_conv3_8713*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_72092&
$block4_conv3/StatefulPartitionedCallХ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_8716block4_conv4_8718*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_72312&
$block4_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_744021
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallљ
#tf_op_layer_Shape_7/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_74562%
#tf_op_layer_Shape_7/PartitionedCallљ
#tf_op_layer_Shape_5/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_74692%
#tf_op_layer_Shape_5/PartitionedCallљ
#tf_op_layer_Shape_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_74822%
#tf_op_layer_Shape_3/PartitionedCallї
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_74952%
#tf_op_layer_Shape_1/PartitionedCall
,tf_op_layer_strided_slice_10/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_75112.
,tf_op_layer_strided_slice_10/PartitionedCall
+tf_op_layer_strided_slice_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_75272-
+tf_op_layer_strided_slice_7/PartitionedCall
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_75432-
+tf_op_layer_strided_slice_4/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_75592-
+tf_op_layer_strided_slice_1/PartitionedCallю
#tf_op_layer_Shape_6/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_75722%
#tf_op_layer_Shape_6/PartitionedCallџ
#tf_op_layer_Range_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_10/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_75872%
#tf_op_layer_Range_3/PartitionedCallю
#tf_op_layer_Shape_4/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_76002%
#tf_op_layer_Shape_4/PartitionedCallў
#tf_op_layer_Range_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_7/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_76152%
#tf_op_layer_Range_2/PartitionedCallю
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_76282%
#tf_op_layer_Shape_2/PartitionedCallў
#tf_op_layer_Range_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_76432%
#tf_op_layer_Range_1/PartitionedCallш
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
CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_76562#
!tf_op_layer_Shape/PartitionedCallј
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_76712#
!tf_op_layer_Range/PartitionedCall
,tf_op_layer_strided_slice_12/PartitionedCallPartitionedCall,tf_op_layer_Shape_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_76872.
,tf_op_layer_strided_slice_12/PartitionedCall
,tf_op_layer_strided_slice_11/PartitionedCallPartitionedCall,tf_op_layer_Range_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_77032.
,tf_op_layer_strided_slice_11/PartitionedCall
+tf_op_layer_strided_slice_9/PartitionedCallPartitionedCall,tf_op_layer_Shape_4/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_77192-
+tf_op_layer_strided_slice_9/PartitionedCall
+tf_op_layer_strided_slice_8/PartitionedCallPartitionedCall,tf_op_layer_Range_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_77352-
+tf_op_layer_strided_slice_8/PartitionedCall
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_77512-
+tf_op_layer_strided_slice_6/PartitionedCall
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Range_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_77672-
+tf_op_layer_strided_slice_5/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_77832-
+tf_op_layer_strided_slice_3/PartitionedCall
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_77992-
+tf_op_layer_strided_slice_2/PartitionedCallч
)tf_op_layer_BroadcastTo_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_11/PartitionedCall:output:0,tf_op_layer_Shape_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_78132+
)tf_op_layer_BroadcastTo_3/PartitionedCallя
"tf_op_layer_Prod_6/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_12/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_78282$
"tf_op_layer_Prod_6/PartitionedCallц
)tf_op_layer_BroadcastTo_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_8/PartitionedCall:output:0,tf_op_layer_Shape_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_78422+
)tf_op_layer_BroadcastTo_2/PartitionedCallю
"tf_op_layer_Prod_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_9/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_78572$
"tf_op_layer_Prod_4/PartitionedCallц
)tf_op_layer_BroadcastTo_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0,tf_op_layer_Shape_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_78712+
)tf_op_layer_BroadcastTo_1/PartitionedCallю
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_78862$
"tf_op_layer_Prod_2/PartitionedCallр
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_79002)
'tf_op_layer_BroadcastTo/PartitionedCallш
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
CPU

GPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_79152"
 tf_op_layer_Prod/PartitionedCallЫ
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_3/PartitionedCall:output:0+tf_op_layer_Prod_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_79292#
!tf_op_layer_Mul_3/PartitionedCallЫ
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_2/PartitionedCall:output:0+tf_op_layer_Prod_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_79442#
!tf_op_layer_Mul_2/PartitionedCallЫ
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_1/PartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_79592#
!tf_op_layer_Mul_1/PartitionedCallС
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_79742!
tf_op_layer_Mul/PartitionedCallЮ
#tf_op_layer_AddV2_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*tf_op_layer_Mul_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_79892%
#tf_op_layer_AddV2_3/PartitionedCallъ
"tf_op_layer_Prod_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_80042$
"tf_op_layer_Prod_7/PartitionedCallЮ
#tf_op_layer_AddV2_2/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_80182%
#tf_op_layer_AddV2_2/PartitionedCallъ
"tf_op_layer_Prod_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_80332$
"tf_op_layer_Prod_5/PartitionedCallЮ
#tf_op_layer_AddV2_1/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*tf_op_layer_Mul_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_80472%
#tf_op_layer_AddV2_1/PartitionedCallъ
"tf_op_layer_Prod_3/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_80622$
"tf_op_layer_Prod_3/PartitionedCallУ
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_80762#
!tf_op_layer_AddV2/PartitionedCallъ
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_80912$
"tf_op_layer_Prod_1/PartitionedCallЊ
%tf_op_layer_Reshape_3/PartitionedCallPartitionedCall,tf_op_layer_AddV2_3/PartitionedCall:output:0+tf_op_layer_Prod_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_81052'
%tf_op_layer_Reshape_3/PartitionedCallЊ
%tf_op_layer_Reshape_2/PartitionedCallPartitionedCall,tf_op_layer_AddV2_2/PartitionedCall:output:0+tf_op_layer_Prod_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_81202'
%tf_op_layer_Reshape_2/PartitionedCallЊ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall,tf_op_layer_AddV2_1/PartitionedCall:output:0+tf_op_layer_Prod_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_81352'
%tf_op_layer_Reshape_1/PartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_81502%
#tf_op_layer_Reshape/PartitionedCallР
*tf_op_layer_UnravelIndex_3/PartitionedCallPartitionedCall.tf_op_layer_Reshape_3/PartitionedCall:output:0,tf_op_layer_Shape_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_81652,
*tf_op_layer_UnravelIndex_3/PartitionedCallР
*tf_op_layer_UnravelIndex_2/PartitionedCallPartitionedCall.tf_op_layer_Reshape_2/PartitionedCall:output:0,tf_op_layer_Shape_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_81802,
*tf_op_layer_UnravelIndex_2/PartitionedCallР
*tf_op_layer_UnravelIndex_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_81952,
*tf_op_layer_UnravelIndex_1/PartitionedCallЖ
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_82102*
(tf_op_layer_UnravelIndex/PartitionedCallа
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:0block5_conv1_8775block5_conv1_8777*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_72532&
$block5_conv1/StatefulPartitionedCall
'tf_op_layer_Transpose_3/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_82302)
'tf_op_layer_Transpose_3/PartitionedCall
'tf_op_layer_Transpose_2/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_82442)
'tf_op_layer_Transpose_2/PartitionedCall
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_82582)
'tf_op_layer_Transpose_1/PartitionedCall
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_82722'
%tf_op_layer_Transpose/PartitionedCallХ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_8784block5_conv2_8786*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_72752&
$block5_conv2/StatefulPartitionedCallО
IdentityIdentity-block5_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityЈ

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1Њ

Identity_2Identity0tf_op_layer_Transpose_1/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2Њ

Identity_3Identity0tf_op_layer_Transpose_2/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3Њ

Identity_4Identity0tf_op_layer_Transpose_3/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я
b
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_9910
inputs_0	
inputs_1	
identity	р
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_79002
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
й
V
:__inference_tf_op_layer_strided_slice_7_layer_call_fn_9693

inputs	
identity	Ѓ
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_75272
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
ю
[
/__inference_tf_op_layer_Mul_layer_call_fn_10002
inputs_0	
inputs_1	
identity	и
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_79742
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
ъ

+__inference_block3_conv1_layer_call_fn_7087

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_70772
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
М
а
$__inference_model_layer_call_fn_9465

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1	

identity_2	

identity_3	

identity_4	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_85812
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

W
;__inference_tf_op_layer_strided_slice_11_layer_call_fn_9885

inputs	
identity	Н
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_77032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
d
8__inference_tf_op_layer_BroadcastTo_2_layer_call_fn_9956
inputs_0	
inputs_1	
identity	т
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_78422
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
ю
N
2__inference_tf_op_layer_BiasAdd_layer_call_fn_9558

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_73112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д
q
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_9828

inputs	
identity	
strided_slice_5/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_5/begin
strided_slice_5/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_5/end
strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_5/strides
strided_slice_5StridedSliceinputsstrided_slice_5/begin:output:0strided_slice_5/end:output:0 strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_5t
IdentityIdentitystrided_slice_5:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К

Ў
F__inference_block2_conv1_layer_call_and_return_conditional_losses_7033

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

S
7__inference_tf_op_layer_Transpose_3_layer_call_fn_10270

inputs	
identity	А
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_82302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
q
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_9854

inputs	
identity	
strided_slice_8/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_8/begin
strided_slice_8/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_8/end
strided_slice_8/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_8/strides
strided_slice_8StridedSliceinputsstrided_slice_8/begin:output:0strided_slice_8/end:output:0 strided_slice_8/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_8t
IdentityIdentitystrided_slice_8:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_7336

inputs
identity

identity_1	
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*
_cloned(*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity

Identity_1IdentityMaxPoolWithArgmax:argmax:0*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

V
:__inference_tf_op_layer_strided_slice_8_layer_call_fn_9859

inputs	
identity	М
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_77352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_7297

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
valueB"   џџџџ2
strided_slice/stridesЏ
strided_sliceStridedSliceinputsstrided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice
IdentityIdentitystrided_slice:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_7402

inputs
identity

identity_1	
MaxPoolWithArgmax_2MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_2
IdentityIdentityMaxPoolWithArgmax_2:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_2:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
}
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_7842

inputs	
inputs_1	
identity	Џ
BroadcastTo_2BroadcastToinputsinputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_2
IdentityIdentityBroadcastTo_2:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Б
x
L__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_10032
inputs_0	
inputs_1	
identity	
Mul_3Mulinputs_0inputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_3
IdentityIdentity	Mul_3:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
Ј
u
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_7944

inputs	
inputs_1	
identity	
Mul_2Mulinputsinputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_2
IdentityIdentity	Mul_2:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
Н

Ў
F__inference_block5_conv1_layer_call_and_return_conditional_losses_7253

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ъ

+__inference_block5_conv2_layer_call_fn_7285

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_72752
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
к

U__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_10208
inputs_0	
inputs_1	
identity	
UnravelIndex_2UnravelIndexinputs_0inputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_2k
IdentityIdentityUnravelIndex_2:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1

_
3__inference_tf_op_layer_Reshape_layer_call_fn_10142
inputs_0	
inputs_1	
identity	Е
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_81502
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
г
d
8__inference_tf_op_layer_BroadcastTo_1_layer_call_fn_9933
inputs_0	
inputs_1	
identity	т
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_78712
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
а
~
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_8195

inputs	
inputs_1	
identity	
UnravelIndex_1UnravelIndexinputsinputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_1k
IdentityIdentityUnravelIndex_1:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs

L
0__inference_tf_op_layer_Shape_layer_call_fn_9728

inputs
identity	
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
CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_76562
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ь
i
M__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_10125

inputs	
identity	~
Prod_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_7/reduction_indices
Prod_7Prodinputs!Prod_7/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_7V
IdentityIdentityProd_7:output:0*
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
С
r
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_7703

inputs	
identity	
strided_slice_11/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_11/begin
strided_slice_11/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_11/end
strided_slice_11/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_11/strides
strided_slice_11StridedSliceinputsstrided_slice_11/begin:output:0strided_slice_11/end:output:0!strided_slice_11/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_11u
IdentityIdentitystrided_slice_11:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_9593

inputs
identity

identity_1	
MaxPoolWithArgmax_2MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_2
IdentityIdentityMaxPoolWithArgmax_2:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_2:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_9841

inputs	
identity	x
strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/begint
strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/end|
strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stridesы
strided_slice_6StridedSliceinputsstrided_slice_6/begin:output:0strided_slice_6/end:output:0 strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_6_
IdentityIdentitystrided_slice_6:output:0*
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
Ё
N
2__inference_tf_op_layer_Shape_5_layer_call_fn_9644

inputs	
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_74692
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

z
N__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_10113
inputs_0	
inputs_1	
identity	
AddV2_3AddV2inputs_0inputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_3z
IdentityIdentityAddV2_3:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Л

S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_9950
inputs_0	
inputs_1	
identity	Б
BroadcastTo_2BroadcastToinputs_0inputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_2
IdentityIdentityBroadcastTo_2:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
ъ

+__inference_block3_conv3_layer_call_fn_7131

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_71212
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ъ

+__inference_block3_conv4_layer_call_fn_7153

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_71432
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
а
~
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_8180

inputs	
inputs_1	
identity	
UnravelIndex_2UnravelIndexinputsinputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_2k
IdentityIdentityUnravelIndex_2:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ъ

+__inference_block3_conv2_layer_call_fn_7109

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_70992
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
i
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_9619

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ѓ
d
8__inference_tf_op_layer_UnravelIndex_layer_call_fn_10190
inputs_0	
inputs_1	
identity	О
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_82102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Ж
]
1__inference_tf_op_layer_AddV2_layer_call_fn_10050
inputs_0	
inputs_1	
identity	б
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_80762
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Л
r
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_9701

inputs	
identity	z
strided_slice_10/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_10/beginv
strided_slice_10/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/end~
strided_slice_10/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stridesє
strided_slice_10StridedSliceinputsstrided_slice_10/begin:output:0strided_slice_10/end:output:0!strided_slice_10/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_10\
IdentityIdentitystrided_slice_10:output:0*
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
з
L
0__inference_tf_op_layer_Range_layer_call_fn_9718

inputs	
identity	І
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_76712
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_7751

inputs	
identity	x
strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/begint
strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/end|
strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stridesы
strided_slice_6StridedSliceinputsstrided_slice_6/begin:output:0strided_slice_6/end:output:0 strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_6_
IdentityIdentitystrided_slice_6:output:0*
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
б

S__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_10184
inputs_0	
inputs_1	
identity	
UnravelIndexUnravelIndexinputs_0inputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndexi
IdentityIdentityUnravelIndex:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1

|
P__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_10172
inputs_0	
inputs_1	
identity	
	Reshape_3Reshapeinputs_0inputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_3b
IdentityIdentityReshape_3:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Н

Ў
F__inference_block4_conv2_layer_call_and_return_conditional_losses_7187

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ь
i
M__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_10079

inputs	
identity	~
Prod_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_3/reduction_indices
Prod_3Prodinputs!Prod_3/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_3V
IdentityIdentityProd_3:output:0*
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
О
_
3__inference_tf_op_layer_AddV2_2_layer_call_fn_10096
inputs_0	
inputs_1	
identity	д
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_80182
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Н

Ў
F__inference_block4_conv1_layer_call_and_return_conditional_losses_7165

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
]
1__inference_tf_op_layer_Mul_3_layer_call_fn_10038
inputs_0	
inputs_1	
identity	к
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_79292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1

s
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_7974

inputs	
inputs_1	
identity	
MulMulinputsinputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul~
IdentityIdentityMul:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
ау

?__inference_model_layer_call_and_return_conditional_losses_9164

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource
identity

identity_1	

identity_2	

identity_3	

identity_4	Џ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tf_op_layer_strided_slice/strided_slice/beginЋ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2-
+tf_op_layer_strided_slice/strided_slice/endГ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   џџџџ21
/tf_op_layer_strided_slice/strided_slice/stridesБ
'tf_op_layer_strided_slice/strided_sliceStridedSliceinputs6tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

begin_mask*
ellipsis_mask*
end_mask2)
'tf_op_layer_strided_slice/strided_slice
 tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"ХрЯТйщТ)\їТ2"
 tf_op_layer_BiasAdd/BiasAdd/bias§
tf_op_layer_BiasAdd/BiasAddBiasAdd0tf_op_layer_strided_slice/strided_slice:output:0)tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_BiasAdd/BiasAddМ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpњ
block1_conv1/Conv2DConv2D$tf_op_layer_BiasAdd/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
block1_conv1/Conv2DГ
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOpЮ
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv1/BiasAdd
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv1/ReluМ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpѕ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
block1_conv2/Conv2DГ
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOpЮ
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv2/BiasAdd
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv2/Reluл
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmaxblock1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingSAME*
strides
21
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxН
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp
block2_conv1/Conv2DConv2D8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block2_conv1/Conv2DД
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOpЯ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv1/ReluО
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpі
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block2_conv2/Conv2DД
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOpЯ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv2/Reluх
3tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1MaxPoolWithArgmaxblock2_conv2/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1О
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp
block3_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv1/Conv2DД
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOpЯ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv1/BiasAdd
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv1/ReluО
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpі
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv2/Conv2DД
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOpЯ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv2/BiasAdd
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv2/ReluО
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpі
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv3/Conv2DД
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOpЯ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv3/BiasAdd
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv3/ReluО
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpі
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv4/Conv2DД
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOpЯ
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv4/BiasAdd
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv4/Reluх
3tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2MaxPoolWithArgmaxblock3_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2О
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp
block4_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv1/Conv2DД
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOpЯ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv1/BiasAdd
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv1/ReluО
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpі
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv2/Conv2DД
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOpЯ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv2/BiasAdd
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv2/ReluО
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpі
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv3/Conv2DД
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOpЯ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv3/BiasAdd
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv3/ReluО
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpі
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv4/Conv2DД
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOpЯ
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv4/BiasAdd
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv4/Reluх
3tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3MaxPoolWithArgmaxblock4_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3Х
tf_op_layer_Shape_7/Shape_7Shape<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_7/Shape_7Х
tf_op_layer_Shape_5/Shape_5Shape<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_5/Shape_5Х
tf_op_layer_Shape_3/Shape_3Shape<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_3/Shape_3С
tf_op_layer_Shape_1/Shape_1Shape8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_1/Shape_1Д
3tf_op_layer_strided_slice_10/strided_slice_10/beginConst*
_output_shapes
:*
dtype0*
valueB: 25
3tf_op_layer_strided_slice_10/strided_slice_10/beginА
1tf_op_layer_strided_slice_10/strided_slice_10/endConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_10/strided_slice_10/endИ
5tf_op_layer_strided_slice_10/strided_slice_10/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5tf_op_layer_strided_slice_10/strided_slice_10/stridesЃ
-tf_op_layer_strided_slice_10/strided_slice_10StridedSlice$tf_op_layer_Shape_7/Shape_7:output:0<tf_op_layer_strided_slice_10/strided_slice_10/begin:output:0:tf_op_layer_strided_slice_10/strided_slice_10/end:output:0>tf_op_layer_strided_slice_10/strided_slice_10/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2/
-tf_op_layer_strided_slice_10/strided_slice_10А
1tf_op_layer_strided_slice_7/strided_slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_7/strided_slice_7/beginЌ
/tf_op_layer_strided_slice_7/strided_slice_7/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_7/strided_slice_7/endД
3tf_op_layer_strided_slice_7/strided_slice_7/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_7/strided_slice_7/strides
+tf_op_layer_strided_slice_7/strided_slice_7StridedSlice$tf_op_layer_Shape_5/Shape_5:output:0:tf_op_layer_strided_slice_7/strided_slice_7/begin:output:08tf_op_layer_strided_slice_7/strided_slice_7/end:output:0<tf_op_layer_strided_slice_7/strided_slice_7/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_7/strided_slice_7А
1tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_4/strided_slice_4/beginЌ
/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_4/strided_slice_4/endД
3tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_4/strided_slice_4/strides
+tf_op_layer_strided_slice_4/strided_slice_4StridedSlice$tf_op_layer_Shape_3/Shape_3:output:0:tf_op_layer_strided_slice_4/strided_slice_4/begin:output:08tf_op_layer_strided_slice_4/strided_slice_4/end:output:0<tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_4/strided_slice_4А
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_1/strided_slice_1/beginЌ
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_1/strided_slice_1/endД
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_1/strided_slice_1/strides
+tf_op_layer_strided_slice_1/strided_slice_1StridedSlice$tf_op_layer_Shape_1/Shape_1:output:0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_1/strided_slice_1Ј
tf_op_layer_Shape_6/Shape_6Shapeblock4_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_6/Shape_6
!tf_op_layer_Range_3/Range_3/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_3/Range_3/start
!tf_op_layer_Range_3/Range_3/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_3/Range_3/delta
tf_op_layer_Range_3/Range_3Range*tf_op_layer_Range_3/Range_3/start:output:06tf_op_layer_strided_slice_10/strided_slice_10:output:0*tf_op_layer_Range_3/Range_3/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_3/Range_3Ј
tf_op_layer_Shape_4/Shape_4Shapeblock3_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_4/Shape_4
!tf_op_layer_Range_2/Range_2/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_2/Range_2/start
!tf_op_layer_Range_2/Range_2/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_2/Range_2/delta
tf_op_layer_Range_2/Range_2Range*tf_op_layer_Range_2/Range_2/start:output:04tf_op_layer_strided_slice_7/strided_slice_7:output:0*tf_op_layer_Range_2/Range_2/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_2/Range_2Ј
tf_op_layer_Shape_2/Shape_2Shapeblock2_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_2/Shape_2
!tf_op_layer_Range_1/Range_1/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_1/Range_1/start
!tf_op_layer_Range_1/Range_1/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_1/Range_1/delta
tf_op_layer_Range_1/Range_1Range*tf_op_layer_Range_1/Range_1/start:output:04tf_op_layer_strided_slice_4/strided_slice_4:output:0*tf_op_layer_Range_1/Range_1/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_1/Range_1 
tf_op_layer_Shape/ShapeShapeblock1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape/Shape
tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
tf_op_layer_Range/Range/start
tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Range/Range/delta
tf_op_layer_Range/RangeRange&tf_op_layer_Range/Range/start:output:04tf_op_layer_strided_slice_1/strided_slice_1:output:0&tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range/RangeД
3tf_op_layer_strided_slice_12/strided_slice_12/beginConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_12/strided_slice_12/beginА
1tf_op_layer_strided_slice_12/strided_slice_12/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_12/strided_slice_12/endИ
5tf_op_layer_strided_slice_12/strided_slice_12/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5tf_op_layer_strided_slice_12/strided_slice_12/strides
-tf_op_layer_strided_slice_12/strided_slice_12StridedSlice$tf_op_layer_Shape_6/Shape_6:output:0<tf_op_layer_strided_slice_12/strided_slice_12/begin:output:0:tf_op_layer_strided_slice_12/strided_slice_12/end:output:0>tf_op_layer_strided_slice_12/strided_slice_12/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2/
-tf_op_layer_strided_slice_12/strided_slice_12У
3tf_op_layer_strided_slice_11/strided_slice_11/beginConst*
_output_shapes
:*
dtype0*%
valueB"                25
3tf_op_layer_strided_slice_11/strided_slice_11/beginП
1tf_op_layer_strided_slice_11/strided_slice_11/endConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_11/strided_slice_11/endЧ
5tf_op_layer_strided_slice_11/strided_slice_11/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            27
5tf_op_layer_strided_slice_11/strided_slice_11/stridesЮ
-tf_op_layer_strided_slice_11/strided_slice_11StridedSlice$tf_op_layer_Range_3/Range_3:output:0<tf_op_layer_strided_slice_11/strided_slice_11/begin:output:0:tf_op_layer_strided_slice_11/strided_slice_11/end:output:0>tf_op_layer_strided_slice_11/strided_slice_11/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2/
-tf_op_layer_strided_slice_11/strided_slice_11А
1tf_op_layer_strided_slice_9/strided_slice_9/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_9/strided_slice_9/beginЌ
/tf_op_layer_strided_slice_9/strided_slice_9/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_9/strided_slice_9/endД
3tf_op_layer_strided_slice_9/strided_slice_9/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_9/strided_slice_9/strides
+tf_op_layer_strided_slice_9/strided_slice_9StridedSlice$tf_op_layer_Shape_4/Shape_4:output:0:tf_op_layer_strided_slice_9/strided_slice_9/begin:output:08tf_op_layer_strided_slice_9/strided_slice_9/end:output:0<tf_op_layer_strided_slice_9/strided_slice_9/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_9/strided_slice_9П
1tf_op_layer_strided_slice_8/strided_slice_8/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_8/strided_slice_8/beginЛ
/tf_op_layer_strided_slice_8/strided_slice_8/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_8/strided_slice_8/endУ
3tf_op_layer_strided_slice_8/strided_slice_8/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_8/strided_slice_8/stridesФ
+tf_op_layer_strided_slice_8/strided_slice_8StridedSlice$tf_op_layer_Range_2/Range_2:output:0:tf_op_layer_strided_slice_8/strided_slice_8/begin:output:08tf_op_layer_strided_slice_8/strided_slice_8/end:output:0<tf_op_layer_strided_slice_8/strided_slice_8/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_8/strided_slice_8А
1tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_6/strided_slice_6/beginЌ
/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_6/strided_slice_6/endД
3tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_6/strided_slice_6/strides
+tf_op_layer_strided_slice_6/strided_slice_6StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_6/strided_slice_6/begin:output:08tf_op_layer_strided_slice_6/strided_slice_6/end:output:0<tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_6/strided_slice_6П
1tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_5/strided_slice_5/beginЛ
/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_5/strided_slice_5/endУ
3tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_5/strided_slice_5/stridesФ
+tf_op_layer_strided_slice_5/strided_slice_5StridedSlice$tf_op_layer_Range_1/Range_1:output:0:tf_op_layer_strided_slice_5/strided_slice_5/begin:output:08tf_op_layer_strided_slice_5/strided_slice_5/end:output:0<tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_5/strided_slice_5А
1tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_3/strided_slice_3/beginЌ
/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_3/strided_slice_3/endД
3tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_3/strided_slice_3/strides
+tf_op_layer_strided_slice_3/strided_slice_3StridedSlice tf_op_layer_Shape/Shape:output:0:tf_op_layer_strided_slice_3/strided_slice_3/begin:output:08tf_op_layer_strided_slice_3/strided_slice_3/end:output:0<tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_3/strided_slice_3П
1tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_2/strided_slice_2/beginЛ
/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_2/strided_slice_2/endУ
3tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_2/strided_slice_2/stridesР
+tf_op_layer_strided_slice_2/strided_slice_2StridedSlice tf_op_layer_Range/Range:output:0:tf_op_layer_strided_slice_2/strided_slice_2/begin:output:08tf_op_layer_strided_slice_2/strided_slice_2/end:output:0<tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_2/strided_slice_2Ї
'tf_op_layer_BroadcastTo_3/BroadcastTo_3BroadcastTo6tf_op_layer_strided_slice_11/strided_slice_11:output:0$tf_op_layer_Shape_7/Shape_7:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_3/BroadcastTo_3Є
+tf_op_layer_Prod_6/Prod_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_6/Prod_6/reduction_indicesм
tf_op_layer_Prod_6/Prod_6Prod6tf_op_layer_strided_slice_12/strided_slice_12:output:04tf_op_layer_Prod_6/Prod_6/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_6/Prod_6Ѕ
'tf_op_layer_BroadcastTo_2/BroadcastTo_2BroadcastTo4tf_op_layer_strided_slice_8/strided_slice_8:output:0$tf_op_layer_Shape_5/Shape_5:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_2/BroadcastTo_2Є
+tf_op_layer_Prod_4/Prod_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_4/Prod_4/reduction_indicesк
tf_op_layer_Prod_4/Prod_4Prod4tf_op_layer_strided_slice_9/strided_slice_9:output:04tf_op_layer_Prod_4/Prod_4/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_4/Prod_4Ѕ
'tf_op_layer_BroadcastTo_1/BroadcastTo_1BroadcastTo4tf_op_layer_strided_slice_5/strided_slice_5:output:0$tf_op_layer_Shape_3/Shape_3:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_1/BroadcastTo_1Є
+tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_2/Prod_2/reduction_indicesк
tf_op_layer_Prod_2/Prod_2Prod4tf_op_layer_strided_slice_6/strided_slice_6:output:04tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_2/Prod_2
#tf_op_layer_BroadcastTo/BroadcastToBroadcastTo4tf_op_layer_strided_slice_2/strided_slice_2:output:0$tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2%
#tf_op_layer_BroadcastTo/BroadcastTo
'tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2)
'tf_op_layer_Prod/Prod/reduction_indicesЮ
tf_op_layer_Prod/ProdProd4tf_op_layer_strided_slice_3/strided_slice_3:output:00tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod/Prodы
tf_op_layer_Mul_3/Mul_3Mul0tf_op_layer_BroadcastTo_3/BroadcastTo_3:output:0"tf_op_layer_Prod_6/Prod_6:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_3/Mul_3ы
tf_op_layer_Mul_2/Mul_2Mul0tf_op_layer_BroadcastTo_2/BroadcastTo_2:output:0"tf_op_layer_Prod_4/Prod_4:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_2/Mul_2ы
tf_op_layer_Mul_1/Mul_1Mul0tf_op_layer_BroadcastTo_1/BroadcastTo_1:output:0"tf_op_layer_Prod_2/Prod_2:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1к
tf_op_layer_Mul/MulMul,tf_op_layer_BroadcastTo/BroadcastTo:output:0tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
tf_op_layer_Mul/Mulњ
tf_op_layer_AddV2_3/AddV2_3AddV2<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0tf_op_layer_Mul_3/Mul_3:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Prod_7/Prod_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_7/Prod_7/reduction_indicesп
tf_op_layer_Prod_7/Prod_7Prod$tf_op_layer_Shape_7/Shape_7:output:04tf_op_layer_Prod_7/Prod_7/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_7/Prod_7њ
tf_op_layer_AddV2_2/AddV2_2AddV2<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0tf_op_layer_Mul_2/Mul_2:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Є
+tf_op_layer_Prod_5/Prod_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_5/Prod_5/reduction_indicesп
tf_op_layer_Prod_5/Prod_5Prod$tf_op_layer_Shape_5/Shape_5:output:04tf_op_layer_Prod_5/Prod_5/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_5/Prod_5њ
tf_op_layer_AddV2_1/AddV2_1AddV2<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0tf_op_layer_Mul_1/Mul_1:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1Є
+tf_op_layer_Prod_3/Prod_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_3/Prod_3/reduction_indicesп
tf_op_layer_Prod_3/Prod_3Prod$tf_op_layer_Shape_3/Shape_3:output:04tf_op_layer_Prod_3/Prod_3/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_3/Prod_3щ
tf_op_layer_AddV2/AddV2AddV28tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
tf_op_layer_AddV2/AddV2Є
+tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_1/Prod_1/reduction_indicesп
tf_op_layer_Prod_1/Prod_1Prod$tf_op_layer_Shape_1/Shape_1:output:04tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_1/Prod_1н
tf_op_layer_Reshape_3/Reshape_3Reshapetf_op_layer_AddV2_3/AddV2_3:z:0"tf_op_layer_Prod_7/Prod_7:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_3/Reshape_3н
tf_op_layer_Reshape_2/Reshape_2Reshapetf_op_layer_AddV2_2/AddV2_2:z:0"tf_op_layer_Prod_5/Prod_5:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_2/Reshape_2н
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_AddV2_1/AddV2_1:z:0"tf_op_layer_Prod_3/Prod_3:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_1/Reshape_1б
tf_op_layer_Reshape/ReshapeReshapetf_op_layer_AddV2/AddV2:z:0"tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Reshape/Reshapeњ
)tf_op_layer_UnravelIndex_3/UnravelIndex_3UnravelIndex(tf_op_layer_Reshape_3/Reshape_3:output:0$tf_op_layer_Shape_6/Shape_6:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_3/UnravelIndex_3њ
)tf_op_layer_UnravelIndex_2/UnravelIndex_2UnravelIndex(tf_op_layer_Reshape_2/Reshape_2:output:0$tf_op_layer_Shape_4/Shape_4:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_2/UnravelIndex_2њ
)tf_op_layer_UnravelIndex_1/UnravelIndex_1UnravelIndex(tf_op_layer_Reshape_1/Reshape_1:output:0$tf_op_layer_Shape_2/Shape_2:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_1/UnravelIndex_1ъ
%tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex$tf_op_layer_Reshape/Reshape:output:0 tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2'
%tf_op_layer_UnravelIndex/UnravelIndexО
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp
block5_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block5_conv1/Conv2DД
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOpЯ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv1/BiasAdd
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv1/ReluЅ
(tf_op_layer_Transpose_3/Transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_3/Transpose_3/permџ
#tf_op_layer_Transpose_3/Transpose_3	Transpose2tf_op_layer_UnravelIndex_3/UnravelIndex_3:output:01tf_op_layer_Transpose_3/Transpose_3/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_3/Transpose_3Ѕ
(tf_op_layer_Transpose_2/Transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_2/Transpose_2/permџ
#tf_op_layer_Transpose_2/Transpose_2	Transpose2tf_op_layer_UnravelIndex_2/UnravelIndex_2:output:01tf_op_layer_Transpose_2/Transpose_2/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_2/Transpose_2Ѕ
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_1/Transpose_1/permџ
#tf_op_layer_Transpose_1/Transpose_1	Transpose2tf_op_layer_UnravelIndex_1/UnravelIndex_1:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_1/Transpose_1
$tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$tf_op_layer_Transpose/Transpose/permя
tf_op_layer_Transpose/Transpose	Transpose.tf_op_layer_UnravelIndex/UnravelIndex:output:0-tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Transpose/TransposeО
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpі
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block5_conv2/Conv2DД
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOpЯ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv2/BiasAdd
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv2/Relu
IdentityIdentityblock5_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity{

Identity_1Identity#tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity'tf_op_layer_Transpose_1/Transpose_1:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity'tf_op_layer_Transpose_2/Transpose_2:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity'tf_op_layer_Transpose_3/Transpose_3:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::::::::::::::::::::::::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н

Ў
F__inference_block4_conv3_layer_call_and_return_conditional_losses_7209

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
r
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_7511

inputs	
identity	z
strided_slice_10/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_10/beginv
strided_slice_10/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/end~
strided_slice_10/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stridesє
strided_slice_10StridedSliceinputsstrided_slice_10/begin:output:0strided_slice_10/end:output:0!strided_slice_10/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_10\
IdentityIdentitystrided_slice_10:output:0*
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
ђ
n
R__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_10243

inputs	
identity	u
Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_1c
IdentityIdentityTranspose_1:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
m
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_8230

inputs	
identity	u
Transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_3/perm
Transpose_3	TransposeinputsTranspose_3/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_3c
IdentityIdentityTranspose_3:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
r
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_9880

inputs	
identity	
strided_slice_11/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_11/begin
strided_slice_11/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_11/end
strided_slice_11/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_11/strides
strided_slice_11StridedSliceinputsstrided_slice_11/begin:output:0strided_slice_11/end:output:0!strided_slice_11/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_11u
IdentityIdentitystrided_slice_11:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
f
:__inference_tf_op_layer_UnravelIndex_3_layer_call_fn_10226
inputs_0	
inputs_1	
identity	Р
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_81652
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_7440

inputs
identity

identity_1	
MaxPoolWithArgmax_3MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_3
IdentityIdentityMaxPoolWithArgmax_3:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_3:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
W
;__inference_tf_op_layer_strided_slice_10_layer_call_fn_9706

inputs	
identity	Є
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_75112
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
П
б
$__inference_model_layer_call_fn_8861
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1	

identity_2	

identity_3	

identity_4	ЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_87942
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ў
q
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_9815

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
strided_slice_3/stridesы
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
Ј
u
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_7959

inputs	
inputs_1	
identity	
Mul_1Mulinputsinputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_1
IdentityIdentity	Mul_1:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
Б
h
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_7886

inputs	
identity	~
Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_2/reduction_indicess
Prod_2Prodinputs!Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_2R
IdentityIdentityProd_2:output:0*
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

z
N__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_10067
inputs_0	
inputs_1	
identity	
AddV2_1AddV2inputs_0inputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_1z
IdentityIdentityAddV2_1:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Б
h
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_9985

inputs	
identity	~
Prod_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_6/reduction_indicess
Prod_6Prodinputs!Prod_6/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_6R
IdentityIdentityProd_6:output:0*
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
О
_
3__inference_tf_op_layer_AddV2_3_layer_call_fn_10119
inputs_0	
inputs_1	
identity	д
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_79892
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Ѓ
a
5__inference_tf_op_layer_Reshape_3_layer_call_fn_10178
inputs_0	
inputs_1	
identity	З
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_81052
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
у
k
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_8272

inputs	
identity	q
Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose/perm
	Transpose	TransposeinputsTranspose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
	Transposea
IdentityIdentityTranspose:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
N
2__inference_tf_op_layer_Range_2_layer_call_fn_9762

inputs	
identity	Ј
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_76152
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Д
q
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_7799

inputs	
identity	
strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/begin
strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/end
strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_2/strides
strided_slice_2StridedSliceinputsstrided_slice_2/begin:output:0strided_slice_2/end:output:0 strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_2t
IdentityIdentitystrided_slice_2:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

|
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_10148
inputs_0	
inputs_1	
identity	
	Reshape_1Reshapeinputs_0inputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_1b
IdentityIdentityReshape_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
ъ

+__inference_block4_conv4_layer_call_fn_7241

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_72312
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
к

U__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_10220
inputs_0	
inputs_1	
identity	
UnravelIndex_3UnravelIndexinputs_0inputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_3k
IdentityIdentityUnravelIndex_3:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Л

S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_9927
inputs_0	
inputs_1	
identity	Б
BroadcastTo_1BroadcastToinputs_0inputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_1
IdentityIdentityBroadcastTo_1:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
љ
i
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_7495

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_9745

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
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
N
2__inference_tf_op_layer_Shape_7_layer_call_fn_9654

inputs	
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_74562
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё
m
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_8258

inputs	
identity	u
Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_1c
IdentityIdentityTranspose_1:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_9662

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
strided_slice_1/stridesя
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
ї
y
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_8105

inputs	
inputs_1	
identity	~
	Reshape_3Reshapeinputsinputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_3b
IdentityIdentityReshape_3:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Ч
M
1__inference_tf_op_layer_Prod_6_layer_call_fn_9990

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_78282
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

z
N__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_10090
inputs_0	
inputs_1	
identity	
AddV2_2AddV2inputs_0inputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_2z
IdentityIdentityAddV2_2:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1

S
7__inference_tf_op_layer_Transpose_2_layer_call_fn_10259

inputs	
identity	А
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_82442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_7783

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
strided_slice_3/stridesы
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
ё
g
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_7656

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_7543

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
strided_slice_4/stridesя
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
Г
}
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_9904
inputs_0	
inputs_1	
identity	­
BroadcastToBroadcastToinputs_0inputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo
IdentityIdentityBroadcastTo:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Л
r
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_7687

inputs	
identity	z
strided_slice_12/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/beginv
strided_slice_12/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/end~
strided_slice_12/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/strides№
strided_slice_12StridedSliceinputsstrided_slice_12/begin:output:0strided_slice_12/end:output:0!strided_slice_12/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_12`
IdentityIdentitystrided_slice_12:output:0*
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
а
N
2__inference_tf_op_layer_Prod_3_layer_call_fn_10084

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_80622
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
Ѓ
f
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_7915

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
ё
m
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_8244

inputs	
identity	u
Transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_2/perm
Transpose_2	TransposeinputsTranspose_2/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_2c
IdentityIdentityTranspose_2:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
n
R__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_10254

inputs	
identity	u
Transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_2/perm
Transpose_2	TransposeinputsTranspose_2/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_2c
IdentityIdentityTranspose_2:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш
|
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_8210

inputs	
inputs_1	
identity	
UnravelIndexUnravelIndexinputsinputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndexi
IdentityIdentityUnravelIndex:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_7482

inputs	
identity	g
Shape_3Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_3W
IdentityIdentityShape_3:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н

Ў
F__inference_block4_conv4_layer_call_and_return_conditional_losses_7231

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
i
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_9639

inputs	
identity	g
Shape_5Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_5W
IdentityIdentityShape_5:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д
q
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_7767

inputs	
identity	
strided_slice_5/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_5/begin
strided_slice_5/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_5/end
strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_5/strides
strided_slice_5StridedSliceinputsstrided_slice_5/begin:output:0strided_slice_5/end:output:0 strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_5t
IdentityIdentitystrided_slice_5:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№
h
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_9572

inputs
identity

identity_1	ў
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_73362
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity

Identity_1IdentityPartitionedCall:output:1*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ё
N
2__inference_tf_op_layer_Shape_2_layer_call_fn_9750

inputs
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_76282
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У
K
/__inference_tf_op_layer_Prod_layer_call_fn_9921

inputs	
identity	
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
CPU

GPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_79152
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
Ћ
{
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_7900

inputs	
inputs_1	
identity	Ћ
BroadcastToBroadcastToinputsinputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo
IdentityIdentityBroadcastTo:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Д

Ў
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7011

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
э
w
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_8150

inputs	
inputs_1	
identity	z
ReshapeReshapeinputsinputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Reshape`
IdentityIdentityReshape:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Б
h
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_7828

inputs	
identity	~
Prod_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_6/reduction_indicess
Prod_6Prodinputs!Prod_6/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_6R
IdentityIdentityProd_6:output:0*
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
а
N
2__inference_tf_op_layer_Prod_1_layer_call_fn_10061

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_80912
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
Ы
h
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_8091

inputs	
identity	~
Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_1/reduction_indices
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
Ќz

!__inference__traced_restore_10481
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias*
&assignvariableop_4_block2_conv1_kernel(
$assignvariableop_5_block2_conv1_bias*
&assignvariableop_6_block2_conv2_kernel(
$assignvariableop_7_block2_conv2_bias*
&assignvariableop_8_block3_conv1_kernel(
$assignvariableop_9_block3_conv1_bias+
'assignvariableop_10_block3_conv2_kernel)
%assignvariableop_11_block3_conv2_bias+
'assignvariableop_12_block3_conv3_kernel)
%assignvariableop_13_block3_conv3_bias+
'assignvariableop_14_block3_conv4_kernel)
%assignvariableop_15_block3_conv4_bias+
'assignvariableop_16_block4_conv1_kernel)
%assignvariableop_17_block4_conv1_bias+
'assignvariableop_18_block4_conv2_kernel)
%assignvariableop_19_block4_conv2_bias+
'assignvariableop_20_block4_conv3_kernel)
%assignvariableop_21_block4_conv3_bias+
'assignvariableop_22_block4_conv4_kernel)
%assignvariableop_23_block4_conv4_bias+
'assignvariableop_24_block5_conv1_kernel)
%assignvariableop_25_block5_conv1_bias+
'assignvariableop_26_block5_conv2_kernel)
%assignvariableop_27_block5_conv2_bias
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesИ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10 
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12 
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14 
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv4_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv4_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16 
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv1_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv1_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18 
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv2_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv2_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20 
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv3_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv3_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22 
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv4_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv4_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24 
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26 
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
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
NoOpЦ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28г
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
x
L__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_10020
inputs_0	
inputs_1	
identity	
Mul_2Mulinputs_0inputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_2
IdentityIdentity	Mul_2:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
Г
}
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_7871

inputs	
inputs_1	
identity	Џ
BroadcastTo_1BroadcastToinputsinputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_1
IdentityIdentityBroadcastTo_1:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Н

Ў
F__inference_block5_conv2_layer_call_and_return_conditional_losses_7275

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
n
R__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_10265

inputs	
identity	u
Transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_3/perm
Transpose_3	TransposeinputsTranspose_3/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Transpose_3c
IdentityIdentityTranspose_3:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
w
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_8047

inputs	
inputs_1	
identity	
AddV2_1AddV2inputsinputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_1z
IdentityIdentityAddV2_1:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:rn
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
f
:__inference_tf_op_layer_UnravelIndex_2_layer_call_fn_10214
inputs_0	
inputs_1	
identity	Р
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_81802
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Л

S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_9973
inputs_0	
inputs_1	
identity	Б
BroadcastTo_3BroadcastToinputs_0inputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_3
IdentityIdentityBroadcastTo_3:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Н

Ў
F__inference_block2_conv2_layer_call_and_return_conditional_losses_7055

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
]
1__inference_tf_op_layer_Mul_1_layer_call_fn_10014
inputs_0	
inputs_1	
identity	к
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_79592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
ІГ
Х

?__inference_model_layer_call_and_return_conditional_losses_8434
input_2
block1_conv1_8295
block1_conv1_8297
block1_conv2_8300
block1_conv2_8302
block2_conv1_8307
block2_conv1_8309
block2_conv2_8312
block2_conv2_8314
block3_conv1_8319
block3_conv1_8321
block3_conv2_8324
block3_conv2_8326
block3_conv3_8329
block3_conv3_8331
block3_conv4_8334
block3_conv4_8336
block4_conv1_8341
block4_conv1_8343
block4_conv2_8346
block4_conv2_8348
block4_conv3_8351
block4_conv3_8353
block4_conv4_8356
block4_conv4_8358
block5_conv1_8415
block5_conv1_8417
block5_conv2_8424
block5_conv2_8426
identity

identity_1	

identity_2	

identity_3	

identity_4	Ђ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block3_conv4/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block4_conv4/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCall
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_72972+
)tf_op_layer_strided_slice/PartitionedCall
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_73112%
#tf_op_layer_BiasAdd/PartitionedCallУ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_8295block1_conv1_8297*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_69892&
$block1_conv1/StatefulPartitionedCallФ
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8300block1_conv2_8302*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_70112&
$block1_conv2/StatefulPartitionedCallс
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_73362/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallЮ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_8307block2_conv1_8309*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_70332&
$block2_conv1/StatefulPartitionedCallХ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8312block2_conv2_8314*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_70552&
$block2_conv2/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_736421
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallа
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:0block3_conv1_8319block3_conv1_8321*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_70772&
$block3_conv1/StatefulPartitionedCallХ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8324block3_conv2_8326*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_70992&
$block3_conv2/StatefulPartitionedCallХ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8329block3_conv3_8331*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_71212&
$block3_conv3/StatefulPartitionedCallХ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8334block3_conv4_8336*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_71432&
$block3_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_740221
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallа
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:0block4_conv1_8341block4_conv1_8343*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_71652&
$block4_conv1/StatefulPartitionedCallХ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_8346block4_conv2_8348*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_71872&
$block4_conv2/StatefulPartitionedCallХ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_8351block4_conv3_8353*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_72092&
$block4_conv3/StatefulPartitionedCallХ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_8356block4_conv4_8358*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_72312&
$block4_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_744021
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallљ
#tf_op_layer_Shape_7/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_74562%
#tf_op_layer_Shape_7/PartitionedCallљ
#tf_op_layer_Shape_5/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_74692%
#tf_op_layer_Shape_5/PartitionedCallљ
#tf_op_layer_Shape_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_74822%
#tf_op_layer_Shape_3/PartitionedCallї
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_74952%
#tf_op_layer_Shape_1/PartitionedCall
,tf_op_layer_strided_slice_10/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_75112.
,tf_op_layer_strided_slice_10/PartitionedCall
+tf_op_layer_strided_slice_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_75272-
+tf_op_layer_strided_slice_7/PartitionedCall
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_75432-
+tf_op_layer_strided_slice_4/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_75592-
+tf_op_layer_strided_slice_1/PartitionedCallю
#tf_op_layer_Shape_6/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_75722%
#tf_op_layer_Shape_6/PartitionedCallџ
#tf_op_layer_Range_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_10/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_75872%
#tf_op_layer_Range_3/PartitionedCallю
#tf_op_layer_Shape_4/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_76002%
#tf_op_layer_Shape_4/PartitionedCallў
#tf_op_layer_Range_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_7/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_76152%
#tf_op_layer_Range_2/PartitionedCallю
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_76282%
#tf_op_layer_Shape_2/PartitionedCallў
#tf_op_layer_Range_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_76432%
#tf_op_layer_Range_1/PartitionedCallш
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
CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_76562#
!tf_op_layer_Shape/PartitionedCallј
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_76712#
!tf_op_layer_Range/PartitionedCall
,tf_op_layer_strided_slice_12/PartitionedCallPartitionedCall,tf_op_layer_Shape_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_76872.
,tf_op_layer_strided_slice_12/PartitionedCall
,tf_op_layer_strided_slice_11/PartitionedCallPartitionedCall,tf_op_layer_Range_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_77032.
,tf_op_layer_strided_slice_11/PartitionedCall
+tf_op_layer_strided_slice_9/PartitionedCallPartitionedCall,tf_op_layer_Shape_4/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_77192-
+tf_op_layer_strided_slice_9/PartitionedCall
+tf_op_layer_strided_slice_8/PartitionedCallPartitionedCall,tf_op_layer_Range_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_77352-
+tf_op_layer_strided_slice_8/PartitionedCall
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_77512-
+tf_op_layer_strided_slice_6/PartitionedCall
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Range_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_77672-
+tf_op_layer_strided_slice_5/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_77832-
+tf_op_layer_strided_slice_3/PartitionedCall
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_77992-
+tf_op_layer_strided_slice_2/PartitionedCallч
)tf_op_layer_BroadcastTo_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_11/PartitionedCall:output:0,tf_op_layer_Shape_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_78132+
)tf_op_layer_BroadcastTo_3/PartitionedCallя
"tf_op_layer_Prod_6/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_12/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_78282$
"tf_op_layer_Prod_6/PartitionedCallц
)tf_op_layer_BroadcastTo_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_8/PartitionedCall:output:0,tf_op_layer_Shape_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_78422+
)tf_op_layer_BroadcastTo_2/PartitionedCallю
"tf_op_layer_Prod_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_9/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_78572$
"tf_op_layer_Prod_4/PartitionedCallц
)tf_op_layer_BroadcastTo_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0,tf_op_layer_Shape_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_78712+
)tf_op_layer_BroadcastTo_1/PartitionedCallю
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_78862$
"tf_op_layer_Prod_2/PartitionedCallр
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_79002)
'tf_op_layer_BroadcastTo/PartitionedCallш
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
CPU

GPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_79152"
 tf_op_layer_Prod/PartitionedCallЫ
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_3/PartitionedCall:output:0+tf_op_layer_Prod_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_79292#
!tf_op_layer_Mul_3/PartitionedCallЫ
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_2/PartitionedCall:output:0+tf_op_layer_Prod_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_79442#
!tf_op_layer_Mul_2/PartitionedCallЫ
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_1/PartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_79592#
!tf_op_layer_Mul_1/PartitionedCallС
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_79742!
tf_op_layer_Mul/PartitionedCallЮ
#tf_op_layer_AddV2_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*tf_op_layer_Mul_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_79892%
#tf_op_layer_AddV2_3/PartitionedCallъ
"tf_op_layer_Prod_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_80042$
"tf_op_layer_Prod_7/PartitionedCallЮ
#tf_op_layer_AddV2_2/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_80182%
#tf_op_layer_AddV2_2/PartitionedCallъ
"tf_op_layer_Prod_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_80332$
"tf_op_layer_Prod_5/PartitionedCallЮ
#tf_op_layer_AddV2_1/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*tf_op_layer_Mul_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_80472%
#tf_op_layer_AddV2_1/PartitionedCallъ
"tf_op_layer_Prod_3/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_80622$
"tf_op_layer_Prod_3/PartitionedCallУ
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_80762#
!tf_op_layer_AddV2/PartitionedCallъ
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_80912$
"tf_op_layer_Prod_1/PartitionedCallЊ
%tf_op_layer_Reshape_3/PartitionedCallPartitionedCall,tf_op_layer_AddV2_3/PartitionedCall:output:0+tf_op_layer_Prod_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_81052'
%tf_op_layer_Reshape_3/PartitionedCallЊ
%tf_op_layer_Reshape_2/PartitionedCallPartitionedCall,tf_op_layer_AddV2_2/PartitionedCall:output:0+tf_op_layer_Prod_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_81202'
%tf_op_layer_Reshape_2/PartitionedCallЊ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall,tf_op_layer_AddV2_1/PartitionedCall:output:0+tf_op_layer_Prod_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_81352'
%tf_op_layer_Reshape_1/PartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_81502%
#tf_op_layer_Reshape/PartitionedCallР
*tf_op_layer_UnravelIndex_3/PartitionedCallPartitionedCall.tf_op_layer_Reshape_3/PartitionedCall:output:0,tf_op_layer_Shape_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_81652,
*tf_op_layer_UnravelIndex_3/PartitionedCallР
*tf_op_layer_UnravelIndex_2/PartitionedCallPartitionedCall.tf_op_layer_Reshape_2/PartitionedCall:output:0,tf_op_layer_Shape_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_81802,
*tf_op_layer_UnravelIndex_2/PartitionedCallР
*tf_op_layer_UnravelIndex_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_81952,
*tf_op_layer_UnravelIndex_1/PartitionedCallЖ
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_82102*
(tf_op_layer_UnravelIndex/PartitionedCallа
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:0block5_conv1_8415block5_conv1_8417*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_72532&
$block5_conv1/StatefulPartitionedCall
'tf_op_layer_Transpose_3/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_82302)
'tf_op_layer_Transpose_3/PartitionedCall
'tf_op_layer_Transpose_2/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_82442)
'tf_op_layer_Transpose_2/PartitionedCall
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_82582)
'tf_op_layer_Transpose_1/PartitionedCall
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_82722'
%tf_op_layer_Transpose/PartitionedCallХ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_8424block5_conv2_8426*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_72752&
$block5_conv2/StatefulPartitionedCallО
IdentityIdentity-block5_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityЈ

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1Њ

Identity_2Identity0tf_op_layer_Transpose_1/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2Њ

Identity_3Identity0tf_op_layer_Transpose_2/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3Њ

Identity_4Identity0tf_op_layer_Transpose_3/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н

Ў
F__inference_block3_conv3_layer_call_and_return_conditional_losses_7121

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

g
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_7671

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
Range/delta
RangeRangeRange/start:output:0inputsRange/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
Range^
IdentityIdentityRange:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs

|
P__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_10160
inputs_0	
inputs_1	
identity	
	Reshape_2Reshapeinputs_0inputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_2b
IdentityIdentityReshape_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Ы
h
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_8062

inputs	
identity	~
Prod_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_3/reduction_indices
Prod_3Prodinputs!Prod_3/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_3V
IdentityIdentityProd_3:output:0*
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
і
x
L__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_10044
inputs_0	
inputs_1	
identity	
AddV2AddV2inputs_0inputs_1*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
AddV2w
IdentityIdentity	AddV2:z:0*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
ћ
i
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_7572

inputs
identity	g
Shape_6Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_6W
IdentityIdentityShape_6:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_7469

inputs	
identity	g
Shape_5Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_5W
IdentityIdentityShape_5:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н

Ў
F__inference_block3_conv4_layer_call_and_return_conditional_losses_7143

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ф
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_9553

inputs
identityq
BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"ХрЯТйщТ)\їТ2
BiasAdd/bias
BiasAddBiasAddinputsBiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_9675

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
strided_slice_4/stridesя
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
Д
q
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_9802

inputs	
identity	
strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/begin
strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_2/end
strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_2/strides
strided_slice_2StridedSliceinputsstrided_slice_2/begin:output:0strided_slice_2/end:output:0 strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_2t
IdentityIdentitystrided_slice_2:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
i
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_7311

inputs
identityq
BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"ХрЯТйщТ)\їТ2
BiasAdd/bias
BiasAddBiasAddinputsBiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц

+__inference_block1_conv2_layer_call_fn_7021

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_70112
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц

+__inference_block1_conv1_layer_call_fn_6999

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_69892
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
j
>__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_fn_9600

inputs
identity

identity_1	
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_74022
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityPartitionedCall:output:1*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_7456

inputs	
identity	g
Shape_7Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_7W
IdentityIdentityShape_7:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_7364

inputs
identity

identity_1	
MaxPoolWithArgmax_1MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_1
IdentityIdentityMaxPoolWithArgmax_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_1:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
N
2__inference_tf_op_layer_Shape_3_layer_call_fn_9634

inputs	
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_74822
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_9735

inputs	
identity	`
Range_1/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_1/start`
Range_1/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_1/delta
Range_1RangeRange_1/start:output:0inputsRange_1/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_1`
IdentityIdentityRange_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Ї
u
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_9996
inputs_0	
inputs_1	
identity	
MulMulinputs_0inputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul~
IdentityIdentityMul:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
њ
T
8__inference_tf_op_layer_strided_slice_layer_call_fn_9547

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_72972
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
э
u
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_8076

inputs	
inputs_1	
identity	
AddV2AddV2inputsinputs_1*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
AddV2w
IdentityIdentity	AddV2:z:0*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:rn
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
y
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_8120

inputs	
inputs_1	
identity	~
	Reshape_2Reshapeinputsinputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_2b
IdentityIdentityReshape_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
П
б
$__inference_model_layer_call_fn_8648
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1	

identity_2	

identity_3	

identity_4	ЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_85812
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
a
5__inference_tf_op_layer_Reshape_2_layer_call_fn_10166
inputs_0	
inputs_1	
identity	З
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_81202
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Г
}
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_7813

inputs	
inputs_1	
identity	Џ
BroadcastTo_3BroadcastToinputsinputs_1*
T0	*

Tidx0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
BroadcastTo_3
IdentityIdentityBroadcastTo_3:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
 H

__inference__traced_save_10385
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b3cf8b4c4173473d9677a26488f691ee/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 **
dtypes 
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesњ
ї: :@:@:@@:@:@:::::::::::::::::::::::: 2(
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
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
ї
y
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_8135

inputs	
inputs_1	
identity	~
	Reshape_1Reshapeinputsinputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
	Reshape_1b
IdentityIdentityReshape_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_9649

inputs	
identity	g
Shape_7Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_7W
IdentityIdentityShape_7:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ч
M
1__inference_tf_op_layer_Prod_4_layer_call_fn_9967

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_78572
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
ъ

+__inference_block4_conv2_layer_call_fn_7197

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_71872
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ь
i
M__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_10102

inputs	
identity	~
Prod_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_5/reduction_indices
Prod_5Prodinputs!Prod_5/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_5V
IdentityIdentityProd_5:output:0*
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
Ь
i
M__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_10056

inputs	
identity	~
Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_1/reduction_indices
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
у
W
;__inference_tf_op_layer_strided_slice_12_layer_call_fn_9898

inputs	
identity	Ј
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_76872
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
ъ

+__inference_block4_conv1_layer_call_fn_7175

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_71652
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
i
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_9789

inputs
identity	g
Shape_6Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_6W
IdentityIdentityShape_6:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
_
3__inference_tf_op_layer_AddV2_1_layer_call_fn_10073
inputs_0	
inputs_1	
identity	д
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_80472
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:tp
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Б
h
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_9962

inputs	
identity	~
Prod_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_4/reduction_indicess
Prod_4Prodinputs!Prod_4/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_4R
IdentityIdentityProd_4:output:0*
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
ћ
i
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_7628

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
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ч
M
1__inference_tf_op_layer_Prod_2_layer_call_fn_9944

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_78862
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
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_9579

inputs
identity

identity_1	
MaxPoolWithArgmax_1MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_1
IdentityIdentityMaxPoolWithArgmax_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_1:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
j
>__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_fn_9614

inputs
identity

identity_1	
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_74402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityPartitionedCall:output:1*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

V
:__inference_tf_op_layer_strided_slice_5_layer_call_fn_9833

inputs	
identity	М
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_77672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
q
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_7735

inputs	
identity	
strided_slice_8/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_8/begin
strided_slice_8/endConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_8/end
strided_slice_8/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_8/strides
strided_slice_8StridedSliceinputsstrided_slice_8/begin:output:0strided_slice_8/end:output:0 strided_slice_8/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2
strided_slice_8t
IdentityIdentitystrided_slice_8:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
]
1__inference_tf_op_layer_Mul_2_layer_call_fn_10026
inputs_0	
inputs_1	
identity	к
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_79442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
Ў
q
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_7719

inputs	
identity	x
strided_slice_9/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/begint
strided_slice_9/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/end|
strided_slice_9/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stridesы
strided_slice_9StridedSliceinputsstrided_slice_9/begin:output:0strided_slice_9/end:output:0 strided_slice_9/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_9_
IdentityIdentitystrided_slice_9:output:0*
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
Л
r
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_9893

inputs	
identity	z
strided_slice_12/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/beginv
strided_slice_12/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/end~
strided_slice_12/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/strides№
strided_slice_12StridedSliceinputsstrided_slice_12/begin:output:0strided_slice_12/end:output:0!strided_slice_12/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_12`
IdentityIdentitystrided_slice_12:output:0*
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
Ў
q
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_7559

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
strided_slice_1/stridesя
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
л
N
2__inference_tf_op_layer_Range_1_layer_call_fn_9740

inputs	
identity	Ј
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_76432
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs

g
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_9713

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
Range/delta
RangeRangeRange/start:output:0inputsRange/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
Range^
IdentityIdentityRange:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
а
N
2__inference_tf_op_layer_Prod_5_layer_call_fn_10107

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_80332
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
а
~
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_8165

inputs	
inputs_1	
identity	
UnravelIndex_3UnravelIndexinputsinputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_3k
IdentityIdentityUnravelIndex_3:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
й
V
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_9667

inputs	
identity	Ѓ
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_75592
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
Б
h
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_9939

inputs	
identity	~
Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_2/reduction_indicess
Prod_2Prodinputs!Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_2R
IdentityIdentityProd_2:output:0*
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

i
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_7587

inputs	
identity	`
Range_3/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_3/start`
Range_3/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_3/delta
Range_3RangeRange_3/start:output:0inputsRange_3/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_3`
IdentityIdentityRange_3:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
л
N
2__inference_tf_op_layer_Range_3_layer_call_fn_9784

inputs	
identity	Ј
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_75872
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
э
o
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_9542

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
valueB"   џџџџ2
strided_slice/stridesЏ
strided_sliceStridedSliceinputsstrided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice
IdentityIdentitystrided_slice:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


__inference__wrapped_model_6977
input_25
1model_block1_conv1_conv2d_readvariableop_resource6
2model_block1_conv1_biasadd_readvariableop_resource5
1model_block1_conv2_conv2d_readvariableop_resource6
2model_block1_conv2_biasadd_readvariableop_resource5
1model_block2_conv1_conv2d_readvariableop_resource6
2model_block2_conv1_biasadd_readvariableop_resource5
1model_block2_conv2_conv2d_readvariableop_resource6
2model_block2_conv2_biasadd_readvariableop_resource5
1model_block3_conv1_conv2d_readvariableop_resource6
2model_block3_conv1_biasadd_readvariableop_resource5
1model_block3_conv2_conv2d_readvariableop_resource6
2model_block3_conv2_biasadd_readvariableop_resource5
1model_block3_conv3_conv2d_readvariableop_resource6
2model_block3_conv3_biasadd_readvariableop_resource5
1model_block3_conv4_conv2d_readvariableop_resource6
2model_block3_conv4_biasadd_readvariableop_resource5
1model_block4_conv1_conv2d_readvariableop_resource6
2model_block4_conv1_biasadd_readvariableop_resource5
1model_block4_conv2_conv2d_readvariableop_resource6
2model_block4_conv2_biasadd_readvariableop_resource5
1model_block4_conv3_conv2d_readvariableop_resource6
2model_block4_conv3_biasadd_readvariableop_resource5
1model_block4_conv4_conv2d_readvariableop_resource6
2model_block4_conv4_biasadd_readvariableop_resource5
1model_block5_conv1_conv2d_readvariableop_resource6
2model_block5_conv1_biasadd_readvariableop_resource5
1model_block5_conv2_conv2d_readvariableop_resource6
2model_block5_conv2_biasadd_readvariableop_resource
identity

identity_1	

identity_2	

identity_3	

identity_4	Л
3model/tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/tf_op_layer_strided_slice/strided_slice/beginЗ
1model/tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        23
1model/tf_op_layer_strided_slice/strided_slice/endП
5model/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   џџџџ27
5model/tf_op_layer_strided_slice/strided_slice/stridesа
-model/tf_op_layer_strided_slice/strided_sliceStridedSliceinput_2<model/tf_op_layer_strided_slice/strided_slice/begin:output:0:model/tf_op_layer_strided_slice/strided_slice/end:output:0>model/tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

begin_mask*
ellipsis_mask*
end_mask2/
-model/tf_op_layer_strided_slice/strided_sliceЅ
&model/tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"ХрЯТйщТ)\їТ2(
&model/tf_op_layer_BiasAdd/BiasAdd/bias
!model/tf_op_layer_BiasAdd/BiasAddBiasAdd6model/tf_op_layer_strided_slice/strided_slice:output:0/model/tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!model/tf_op_layer_BiasAdd/BiasAddЮ
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model/block1_conv1/Conv2D/ReadVariableOp
model/block1_conv1/Conv2DConv2D*model/tf_op_layer_BiasAdd/BiasAdd:output:00model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
model/block1_conv1/Conv2DХ
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv1/BiasAdd/ReadVariableOpц
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/block1_conv1/BiasAddЋ
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/block1_conv1/ReluЮ
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(model/block1_conv2/Conv2D/ReadVariableOp
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
model/block1_conv2/Conv2DХ
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv2/BiasAdd/ReadVariableOpц
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/block1_conv2/BiasAddЋ
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/block1_conv2/Reluэ
5model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmax%model/block1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingSAME*
strides
27
5model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxЯ
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(model/block2_conv1/Conv2D/ReadVariableOpЇ
model/block2_conv1/Conv2DConv2D>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block2_conv1/Conv2DЦ
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block2_conv1/BiasAdd/ReadVariableOpч
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block2_conv1/BiasAddЌ
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block2_conv1/Reluа
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block2_conv2/Conv2D/ReadVariableOp
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block2_conv2/Conv2DЦ
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block2_conv2/BiasAdd/ReadVariableOpч
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block2_conv2/BiasAddЌ
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block2_conv2/Reluї
9model/tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1MaxPoolWithArgmax%model/block2_conv2/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2;
9model/tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1а
(model/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv1/Conv2D/ReadVariableOpЋ
model/block3_conv1/Conv2DConv2DBmodel/tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:output:00model/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block3_conv1/Conv2DЦ
)model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv1/BiasAdd/ReadVariableOpч
model/block3_conv1/BiasAddBiasAdd"model/block3_conv1/Conv2D:output:01model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv1/BiasAddЌ
model/block3_conv1/ReluRelu#model/block3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv1/Reluа
(model/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv2/Conv2D/ReadVariableOp
model/block3_conv2/Conv2DConv2D%model/block3_conv1/Relu:activations:00model/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block3_conv2/Conv2DЦ
)model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv2/BiasAdd/ReadVariableOpч
model/block3_conv2/BiasAddBiasAdd"model/block3_conv2/Conv2D:output:01model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv2/BiasAddЌ
model/block3_conv2/ReluRelu#model/block3_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv2/Reluа
(model/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv3/Conv2D/ReadVariableOp
model/block3_conv3/Conv2DConv2D%model/block3_conv2/Relu:activations:00model/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block3_conv3/Conv2DЦ
)model/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv3/BiasAdd/ReadVariableOpч
model/block3_conv3/BiasAddBiasAdd"model/block3_conv3/Conv2D:output:01model/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv3/BiasAddЌ
model/block3_conv3/ReluRelu#model/block3_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv3/Reluа
(model/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv4/Conv2D/ReadVariableOp
model/block3_conv4/Conv2DConv2D%model/block3_conv3/Relu:activations:00model/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block3_conv4/Conv2DЦ
)model/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv4/BiasAdd/ReadVariableOpч
model/block3_conv4/BiasAddBiasAdd"model/block3_conv4/Conv2D:output:01model/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv4/BiasAddЌ
model/block3_conv4/ReluRelu#model/block3_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block3_conv4/Reluї
9model/tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2MaxPoolWithArgmax%model/block3_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2;
9model/tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2а
(model/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv1/Conv2D/ReadVariableOpЋ
model/block4_conv1/Conv2DConv2DBmodel/tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:output:00model/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block4_conv1/Conv2DЦ
)model/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv1/BiasAdd/ReadVariableOpч
model/block4_conv1/BiasAddBiasAdd"model/block4_conv1/Conv2D:output:01model/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv1/BiasAddЌ
model/block4_conv1/ReluRelu#model/block4_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv1/Reluа
(model/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv2/Conv2D/ReadVariableOp
model/block4_conv2/Conv2DConv2D%model/block4_conv1/Relu:activations:00model/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block4_conv2/Conv2DЦ
)model/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv2/BiasAdd/ReadVariableOpч
model/block4_conv2/BiasAddBiasAdd"model/block4_conv2/Conv2D:output:01model/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv2/BiasAddЌ
model/block4_conv2/ReluRelu#model/block4_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv2/Reluа
(model/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv3/Conv2D/ReadVariableOp
model/block4_conv3/Conv2DConv2D%model/block4_conv2/Relu:activations:00model/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block4_conv3/Conv2DЦ
)model/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv3/BiasAdd/ReadVariableOpч
model/block4_conv3/BiasAddBiasAdd"model/block4_conv3/Conv2D:output:01model/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv3/BiasAddЌ
model/block4_conv3/ReluRelu#model/block4_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv3/Reluа
(model/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv4/Conv2D/ReadVariableOp
model/block4_conv4/Conv2DConv2D%model/block4_conv3/Relu:activations:00model/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block4_conv4/Conv2DЦ
)model/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv4/BiasAdd/ReadVariableOpч
model/block4_conv4/BiasAddBiasAdd"model/block4_conv4/Conv2D:output:01model/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv4/BiasAddЌ
model/block4_conv4/ReluRelu#model/block4_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block4_conv4/Reluї
9model/tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3MaxPoolWithArgmax%model/block4_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2;
9model/tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3з
!model/tf_op_layer_Shape_7/Shape_7ShapeBmodel/tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_7/Shape_7з
!model/tf_op_layer_Shape_5/Shape_5ShapeBmodel/tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_5/Shape_5з
!model/tf_op_layer_Shape_3/Shape_3ShapeBmodel/tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_3/Shape_3г
!model/tf_op_layer_Shape_1/Shape_1Shape>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_1/Shape_1Р
9model/tf_op_layer_strided_slice_10/strided_slice_10/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model/tf_op_layer_strided_slice_10/strided_slice_10/beginМ
7model/tf_op_layer_strided_slice_10/strided_slice_10/endConst*
_output_shapes
:*
dtype0*
valueB:29
7model/tf_op_layer_strided_slice_10/strided_slice_10/endФ
;model/tf_op_layer_strided_slice_10/strided_slice_10/stridesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model/tf_op_layer_strided_slice_10/strided_slice_10/stridesЧ
3model/tf_op_layer_strided_slice_10/strided_slice_10StridedSlice*model/tf_op_layer_Shape_7/Shape_7:output:0Bmodel/tf_op_layer_strided_slice_10/strided_slice_10/begin:output:0@model/tf_op_layer_strided_slice_10/strided_slice_10/end:output:0Dmodel/tf_op_layer_strided_slice_10/strided_slice_10/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask25
3model/tf_op_layer_strided_slice_10/strided_slice_10М
7model/tf_op_layer_strided_slice_7/strided_slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/tf_op_layer_strided_slice_7/strided_slice_7/beginИ
5model/tf_op_layer_strided_slice_7/strided_slice_7/endConst*
_output_shapes
:*
dtype0*
valueB:27
5model/tf_op_layer_strided_slice_7/strided_slice_7/endР
9model/tf_op_layer_strided_slice_7/strided_slice_7/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_7/strided_slice_7/stridesН
1model/tf_op_layer_strided_slice_7/strided_slice_7StridedSlice*model/tf_op_layer_Shape_5/Shape_5:output:0@model/tf_op_layer_strided_slice_7/strided_slice_7/begin:output:0>model/tf_op_layer_strided_slice_7/strided_slice_7/end:output:0Bmodel/tf_op_layer_strided_slice_7/strided_slice_7/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask23
1model/tf_op_layer_strided_slice_7/strided_slice_7М
7model/tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/tf_op_layer_strided_slice_4/strided_slice_4/beginИ
5model/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:27
5model/tf_op_layer_strided_slice_4/strided_slice_4/endР
9model/tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_4/strided_slice_4/stridesН
1model/tf_op_layer_strided_slice_4/strided_slice_4StridedSlice*model/tf_op_layer_Shape_3/Shape_3:output:0@model/tf_op_layer_strided_slice_4/strided_slice_4/begin:output:0>model/tf_op_layer_strided_slice_4/strided_slice_4/end:output:0Bmodel/tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask23
1model/tf_op_layer_strided_slice_4/strided_slice_4М
7model/tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/tf_op_layer_strided_slice_1/strided_slice_1/beginИ
5model/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:27
5model/tf_op_layer_strided_slice_1/strided_slice_1/endР
9model/tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_1/strided_slice_1/stridesН
1model/tf_op_layer_strided_slice_1/strided_slice_1StridedSlice*model/tf_op_layer_Shape_1/Shape_1:output:0@model/tf_op_layer_strided_slice_1/strided_slice_1/begin:output:0>model/tf_op_layer_strided_slice_1/strided_slice_1/end:output:0Bmodel/tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask23
1model/tf_op_layer_strided_slice_1/strided_slice_1К
!model/tf_op_layer_Shape_6/Shape_6Shape%model/block4_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_6/Shape_6
'model/tf_op_layer_Range_3/Range_3/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'model/tf_op_layer_Range_3/Range_3/start
'model/tf_op_layer_Range_3/Range_3/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2)
'model/tf_op_layer_Range_3/Range_3/deltaБ
!model/tf_op_layer_Range_3/Range_3Range0model/tf_op_layer_Range_3/Range_3/start:output:0<model/tf_op_layer_strided_slice_10/strided_slice_10:output:00model/tf_op_layer_Range_3/Range_3/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2#
!model/tf_op_layer_Range_3/Range_3К
!model/tf_op_layer_Shape_4/Shape_4Shape%model/block3_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_4/Shape_4
'model/tf_op_layer_Range_2/Range_2/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'model/tf_op_layer_Range_2/Range_2/start
'model/tf_op_layer_Range_2/Range_2/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2)
'model/tf_op_layer_Range_2/Range_2/deltaЏ
!model/tf_op_layer_Range_2/Range_2Range0model/tf_op_layer_Range_2/Range_2/start:output:0:model/tf_op_layer_strided_slice_7/strided_slice_7:output:00model/tf_op_layer_Range_2/Range_2/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2#
!model/tf_op_layer_Range_2/Range_2К
!model/tf_op_layer_Shape_2/Shape_2Shape%model/block2_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2#
!model/tf_op_layer_Shape_2/Shape_2
'model/tf_op_layer_Range_1/Range_1/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'model/tf_op_layer_Range_1/Range_1/start
'model/tf_op_layer_Range_1/Range_1/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2)
'model/tf_op_layer_Range_1/Range_1/deltaЏ
!model/tf_op_layer_Range_1/Range_1Range0model/tf_op_layer_Range_1/Range_1/start:output:0:model/tf_op_layer_strided_slice_4/strided_slice_4:output:00model/tf_op_layer_Range_1/Range_1/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2#
!model/tf_op_layer_Range_1/Range_1В
model/tf_op_layer_Shape/ShapeShape%model/block1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
model/tf_op_layer_Shape/Shape
#model/tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#model/tf_op_layer_Range/Range/start
#model/tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2%
#model/tf_op_layer_Range/Range/delta
model/tf_op_layer_Range/RangeRange,model/tf_op_layer_Range/Range/start:output:0:model/tf_op_layer_strided_slice_1/strided_slice_1:output:0,model/tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
model/tf_op_layer_Range/RangeР
9model/tf_op_layer_strided_slice_12/strided_slice_12/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_12/strided_slice_12/beginМ
7model/tf_op_layer_strided_slice_12/strided_slice_12/endConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/tf_op_layer_strided_slice_12/strided_slice_12/endФ
;model/tf_op_layer_strided_slice_12/strided_slice_12/stridesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model/tf_op_layer_strided_slice_12/strided_slice_12/stridesУ
3model/tf_op_layer_strided_slice_12/strided_slice_12StridedSlice*model/tf_op_layer_Shape_6/Shape_6:output:0Bmodel/tf_op_layer_strided_slice_12/strided_slice_12/begin:output:0@model/tf_op_layer_strided_slice_12/strided_slice_12/end:output:0Dmodel/tf_op_layer_strided_slice_12/strided_slice_12/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask25
3model/tf_op_layer_strided_slice_12/strided_slice_12Я
9model/tf_op_layer_strided_slice_11/strided_slice_11/beginConst*
_output_shapes
:*
dtype0*%
valueB"                2;
9model/tf_op_layer_strided_slice_11/strided_slice_11/beginЫ
7model/tf_op_layer_strided_slice_11/strided_slice_11/endConst*
_output_shapes
:*
dtype0*%
valueB"                29
7model/tf_op_layer_strided_slice_11/strided_slice_11/endг
;model/tf_op_layer_strided_slice_11/strided_slice_11/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2=
;model/tf_op_layer_strided_slice_11/strided_slice_11/stridesђ
3model/tf_op_layer_strided_slice_11/strided_slice_11StridedSlice*model/tf_op_layer_Range_3/Range_3:output:0Bmodel/tf_op_layer_strided_slice_11/strided_slice_11/begin:output:0@model/tf_op_layer_strided_slice_11/strided_slice_11/end:output:0Dmodel/tf_op_layer_strided_slice_11/strided_slice_11/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask25
3model/tf_op_layer_strided_slice_11/strided_slice_11М
7model/tf_op_layer_strided_slice_9/strided_slice_9/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7model/tf_op_layer_strided_slice_9/strided_slice_9/beginИ
5model/tf_op_layer_strided_slice_9/strided_slice_9/endConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/tf_op_layer_strided_slice_9/strided_slice_9/endР
9model/tf_op_layer_strided_slice_9/strided_slice_9/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_9/strided_slice_9/stridesЙ
1model/tf_op_layer_strided_slice_9/strided_slice_9StridedSlice*model/tf_op_layer_Shape_4/Shape_4:output:0@model/tf_op_layer_strided_slice_9/strided_slice_9/begin:output:0>model/tf_op_layer_strided_slice_9/strided_slice_9/end:output:0Bmodel/tf_op_layer_strided_slice_9/strided_slice_9/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask23
1model/tf_op_layer_strided_slice_9/strided_slice_9Ы
7model/tf_op_layer_strided_slice_8/strided_slice_8/beginConst*
_output_shapes
:*
dtype0*%
valueB"                29
7model/tf_op_layer_strided_slice_8/strided_slice_8/beginЧ
5model/tf_op_layer_strided_slice_8/strided_slice_8/endConst*
_output_shapes
:*
dtype0*%
valueB"                27
5model/tf_op_layer_strided_slice_8/strided_slice_8/endЯ
9model/tf_op_layer_strided_slice_8/strided_slice_8/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2;
9model/tf_op_layer_strided_slice_8/strided_slice_8/stridesш
1model/tf_op_layer_strided_slice_8/strided_slice_8StridedSlice*model/tf_op_layer_Range_2/Range_2:output:0@model/tf_op_layer_strided_slice_8/strided_slice_8/begin:output:0>model/tf_op_layer_strided_slice_8/strided_slice_8/end:output:0Bmodel/tf_op_layer_strided_slice_8/strided_slice_8/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask23
1model/tf_op_layer_strided_slice_8/strided_slice_8М
7model/tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7model/tf_op_layer_strided_slice_6/strided_slice_6/beginИ
5model/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/tf_op_layer_strided_slice_6/strided_slice_6/endР
9model/tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_6/strided_slice_6/stridesЙ
1model/tf_op_layer_strided_slice_6/strided_slice_6StridedSlice*model/tf_op_layer_Shape_2/Shape_2:output:0@model/tf_op_layer_strided_slice_6/strided_slice_6/begin:output:0>model/tf_op_layer_strided_slice_6/strided_slice_6/end:output:0Bmodel/tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask23
1model/tf_op_layer_strided_slice_6/strided_slice_6Ы
7model/tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*%
valueB"                29
7model/tf_op_layer_strided_slice_5/strided_slice_5/beginЧ
5model/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*%
valueB"                27
5model/tf_op_layer_strided_slice_5/strided_slice_5/endЯ
9model/tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2;
9model/tf_op_layer_strided_slice_5/strided_slice_5/stridesш
1model/tf_op_layer_strided_slice_5/strided_slice_5StridedSlice*model/tf_op_layer_Range_1/Range_1:output:0@model/tf_op_layer_strided_slice_5/strided_slice_5/begin:output:0>model/tf_op_layer_strided_slice_5/strided_slice_5/end:output:0Bmodel/tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask23
1model/tf_op_layer_strided_slice_5/strided_slice_5М
7model/tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7model/tf_op_layer_strided_slice_3/strided_slice_3/beginИ
5model/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/tf_op_layer_strided_slice_3/strided_slice_3/endР
9model/tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:2;
9model/tf_op_layer_strided_slice_3/strided_slice_3/stridesЕ
1model/tf_op_layer_strided_slice_3/strided_slice_3StridedSlice&model/tf_op_layer_Shape/Shape:output:0@model/tf_op_layer_strided_slice_3/strided_slice_3/begin:output:0>model/tf_op_layer_strided_slice_3/strided_slice_3/end:output:0Bmodel/tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask23
1model/tf_op_layer_strided_slice_3/strided_slice_3Ы
7model/tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                29
7model/tf_op_layer_strided_slice_2/strided_slice_2/beginЧ
5model/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                27
5model/tf_op_layer_strided_slice_2/strided_slice_2/endЯ
9model/tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            2;
9model/tf_op_layer_strided_slice_2/strided_slice_2/stridesф
1model/tf_op_layer_strided_slice_2/strided_slice_2StridedSlice&model/tf_op_layer_Range/Range:output:0@model/tf_op_layer_strided_slice_2/strided_slice_2/begin:output:0>model/tf_op_layer_strided_slice_2/strided_slice_2/end:output:0Bmodel/tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask23
1model/tf_op_layer_strided_slice_2/strided_slice_2П
-model/tf_op_layer_BroadcastTo_3/BroadcastTo_3BroadcastTo<model/tf_op_layer_strided_slice_11/strided_slice_11:output:0*model/tf_op_layer_Shape_7/Shape_7:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2/
-model/tf_op_layer_BroadcastTo_3/BroadcastTo_3А
1model/tf_op_layer_Prod_6/Prod_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_6/Prod_6/reduction_indicesє
model/tf_op_layer_Prod_6/Prod_6Prod<model/tf_op_layer_strided_slice_12/strided_slice_12:output:0:model/tf_op_layer_Prod_6/Prod_6/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2!
model/tf_op_layer_Prod_6/Prod_6Н
-model/tf_op_layer_BroadcastTo_2/BroadcastTo_2BroadcastTo:model/tf_op_layer_strided_slice_8/strided_slice_8:output:0*model/tf_op_layer_Shape_5/Shape_5:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2/
-model/tf_op_layer_BroadcastTo_2/BroadcastTo_2А
1model/tf_op_layer_Prod_4/Prod_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_4/Prod_4/reduction_indicesђ
model/tf_op_layer_Prod_4/Prod_4Prod:model/tf_op_layer_strided_slice_9/strided_slice_9:output:0:model/tf_op_layer_Prod_4/Prod_4/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2!
model/tf_op_layer_Prod_4/Prod_4Н
-model/tf_op_layer_BroadcastTo_1/BroadcastTo_1BroadcastTo:model/tf_op_layer_strided_slice_5/strided_slice_5:output:0*model/tf_op_layer_Shape_3/Shape_3:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2/
-model/tf_op_layer_BroadcastTo_1/BroadcastTo_1А
1model/tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_2/Prod_2/reduction_indicesђ
model/tf_op_layer_Prod_2/Prod_2Prod:model/tf_op_layer_strided_slice_6/strided_slice_6:output:0:model/tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2!
model/tf_op_layer_Prod_2/Prod_2Д
)model/tf_op_layer_BroadcastTo/BroadcastToBroadcastTo:model/tf_op_layer_strided_slice_2/strided_slice_2:output:0*model/tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2+
)model/tf_op_layer_BroadcastTo/BroadcastToЈ
-model/tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2/
-model/tf_op_layer_Prod/Prod/reduction_indicesц
model/tf_op_layer_Prod/ProdProd:model/tf_op_layer_strided_slice_3/strided_slice_3:output:06model/tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
model/tf_op_layer_Prod/Prod
model/tf_op_layer_Mul_3/Mul_3Mul6model/tf_op_layer_BroadcastTo_3/BroadcastTo_3:output:0(model/tf_op_layer_Prod_6/Prod_6:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/tf_op_layer_Mul_3/Mul_3
model/tf_op_layer_Mul_2/Mul_2Mul6model/tf_op_layer_BroadcastTo_2/BroadcastTo_2:output:0(model/tf_op_layer_Prod_4/Prod_4:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/tf_op_layer_Mul_2/Mul_2
model/tf_op_layer_Mul_1/Mul_1Mul6model/tf_op_layer_BroadcastTo_1/BroadcastTo_1:output:0(model/tf_op_layer_Prod_2/Prod_2:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/tf_op_layer_Mul_1/Mul_1ђ
model/tf_op_layer_Mul/MulMul2model/tf_op_layer_BroadcastTo/BroadcastTo:output:0$model/tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/tf_op_layer_Mul/Mul
!model/tf_op_layer_AddV2_3/AddV2_3AddV2Bmodel/tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0!model/tf_op_layer_Mul_3/Mul_3:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!model/tf_op_layer_AddV2_3/AddV2_3А
1model/tf_op_layer_Prod_7/Prod_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_7/Prod_7/reduction_indicesї
model/tf_op_layer_Prod_7/Prod_7Prod*model/tf_op_layer_Shape_7/Shape_7:output:0:model/tf_op_layer_Prod_7/Prod_7/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2!
model/tf_op_layer_Prod_7/Prod_7
!model/tf_op_layer_AddV2_2/AddV2_2AddV2Bmodel/tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0!model/tf_op_layer_Mul_2/Mul_2:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!model/tf_op_layer_AddV2_2/AddV2_2А
1model/tf_op_layer_Prod_5/Prod_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_5/Prod_5/reduction_indicesї
model/tf_op_layer_Prod_5/Prod_5Prod*model/tf_op_layer_Shape_5/Shape_5:output:0:model/tf_op_layer_Prod_5/Prod_5/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2!
model/tf_op_layer_Prod_5/Prod_5
!model/tf_op_layer_AddV2_1/AddV2_1AddV2Bmodel/tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0!model/tf_op_layer_Mul_1/Mul_1:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!model/tf_op_layer_AddV2_1/AddV2_1А
1model/tf_op_layer_Prod_3/Prod_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_3/Prod_3/reduction_indicesї
model/tf_op_layer_Prod_3/Prod_3Prod*model/tf_op_layer_Shape_3/Shape_3:output:0:model/tf_op_layer_Prod_3/Prod_3/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2!
model/tf_op_layer_Prod_3/Prod_3
model/tf_op_layer_AddV2/AddV2AddV2>model/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0model/tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
model/tf_op_layer_AddV2/AddV2А
1model/tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/tf_op_layer_Prod_1/Prod_1/reduction_indicesї
model/tf_op_layer_Prod_1/Prod_1Prod*model/tf_op_layer_Shape_1/Shape_1:output:0:model/tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2!
model/tf_op_layer_Prod_1/Prod_1ѕ
%model/tf_op_layer_Reshape_3/Reshape_3Reshape%model/tf_op_layer_AddV2_3/AddV2_3:z:0(model/tf_op_layer_Prod_7/Prod_7:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2'
%model/tf_op_layer_Reshape_3/Reshape_3ѕ
%model/tf_op_layer_Reshape_2/Reshape_2Reshape%model/tf_op_layer_AddV2_2/AddV2_2:z:0(model/tf_op_layer_Prod_5/Prod_5:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2'
%model/tf_op_layer_Reshape_2/Reshape_2ѕ
%model/tf_op_layer_Reshape_1/Reshape_1Reshape%model/tf_op_layer_AddV2_1/AddV2_1:z:0(model/tf_op_layer_Prod_3/Prod_3:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2'
%model/tf_op_layer_Reshape_1/Reshape_1щ
!model/tf_op_layer_Reshape/ReshapeReshape!model/tf_op_layer_AddV2/AddV2:z:0(model/tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2#
!model/tf_op_layer_Reshape/Reshape
/model/tf_op_layer_UnravelIndex_3/UnravelIndex_3UnravelIndex.model/tf_op_layer_Reshape_3/Reshape_3:output:0*model/tf_op_layer_Shape_6/Shape_6:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ21
/model/tf_op_layer_UnravelIndex_3/UnravelIndex_3
/model/tf_op_layer_UnravelIndex_2/UnravelIndex_2UnravelIndex.model/tf_op_layer_Reshape_2/Reshape_2:output:0*model/tf_op_layer_Shape_4/Shape_4:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ21
/model/tf_op_layer_UnravelIndex_2/UnravelIndex_2
/model/tf_op_layer_UnravelIndex_1/UnravelIndex_1UnravelIndex.model/tf_op_layer_Reshape_1/Reshape_1:output:0*model/tf_op_layer_Shape_2/Shape_2:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ21
/model/tf_op_layer_UnravelIndex_1/UnravelIndex_1
+model/tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex*model/tf_op_layer_Reshape/Reshape:output:0&model/tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2-
+model/tf_op_layer_UnravelIndex/UnravelIndexа
(model/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv1/Conv2D/ReadVariableOpЋ
model/block5_conv1/Conv2DConv2DBmodel/tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:output:00model/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block5_conv1/Conv2DЦ
)model/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv1/BiasAdd/ReadVariableOpч
model/block5_conv1/BiasAddBiasAdd"model/block5_conv1/Conv2D:output:01model/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block5_conv1/BiasAddЌ
model/block5_conv1/ReluRelu#model/block5_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block5_conv1/ReluБ
.model/tf_op_layer_Transpose_3/Transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.model/tf_op_layer_Transpose_3/Transpose_3/perm
)model/tf_op_layer_Transpose_3/Transpose_3	Transpose8model/tf_op_layer_UnravelIndex_3/UnravelIndex_3:output:07model/tf_op_layer_Transpose_3/Transpose_3/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)model/tf_op_layer_Transpose_3/Transpose_3Б
.model/tf_op_layer_Transpose_2/Transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.model/tf_op_layer_Transpose_2/Transpose_2/perm
)model/tf_op_layer_Transpose_2/Transpose_2	Transpose8model/tf_op_layer_UnravelIndex_2/UnravelIndex_2:output:07model/tf_op_layer_Transpose_2/Transpose_2/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)model/tf_op_layer_Transpose_2/Transpose_2Б
.model/tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.model/tf_op_layer_Transpose_1/Transpose_1/perm
)model/tf_op_layer_Transpose_1/Transpose_1	Transpose8model/tf_op_layer_UnravelIndex_1/UnravelIndex_1:output:07model/tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)model/tf_op_layer_Transpose_1/Transpose_1Љ
*model/tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model/tf_op_layer_Transpose/Transpose/perm
%model/tf_op_layer_Transpose/Transpose	Transpose4model/tf_op_layer_UnravelIndex/UnravelIndex:output:03model/tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2'
%model/tf_op_layer_Transpose/Transposeа
(model/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv2/Conv2D/ReadVariableOp
model/block5_conv2/Conv2DConv2D%model/block5_conv1/Relu:activations:00model/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/block5_conv2/Conv2DЦ
)model/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv2/BiasAdd/ReadVariableOpч
model/block5_conv2/BiasAddBiasAdd"model/block5_conv2/Conv2D:output:01model/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block5_conv2/BiasAddЌ
model/block5_conv2/ReluRelu#model/block5_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/block5_conv2/Relu
IdentityIdentity%model/block5_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity)model/tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity-model/tf_op_layer_Transpose_1/Transpose_1:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity-model/tf_op_layer_Transpose_2/Transpose_2:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity-model/tf_op_layer_Transpose_3/Transpose_3:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::::::::::::::::::::::::::j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ

+__inference_block2_conv2_layer_call_fn_7065

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_70552
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў
q
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_9867

inputs	
identity	x
strided_slice_9/beginConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/begint
strided_slice_9/endConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/end|
strided_slice_9/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stridesы
strided_slice_9StridedSliceinputsstrided_slice_9/begin:output:0strided_slice_9/end:output:0 strided_slice_9/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2
strided_slice_9_
IdentityIdentitystrided_slice_9:output:0*
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
с
V
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_9820

inputs	
identity	Ї
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_77832
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

i
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_7615

inputs	
identity	`
Range_2/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_2/start`
Range_2/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_2/delta
Range_2RangeRange_2/start:output:0inputsRange_2/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_2`
IdentityIdentityRange_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
а
N
2__inference_tf_op_layer_Prod_7_layer_call_fn_10130

inputs	
identity	
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_80042
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
й
V
:__inference_tf_op_layer_strided_slice_4_layer_call_fn_9680

inputs	
identity	Ѓ
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_75432
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


W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_9565

inputs
identity

identity_1	
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*
_cloned(*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity

Identity_1IdentityMaxPoolWithArgmax:argmax:0*
T0	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ЃГ
Ф

?__inference_model_layer_call_and_return_conditional_losses_8581

inputs
block1_conv1_8442
block1_conv1_8444
block1_conv2_8447
block1_conv2_8449
block2_conv1_8454
block2_conv1_8456
block2_conv2_8459
block2_conv2_8461
block3_conv1_8466
block3_conv1_8468
block3_conv2_8471
block3_conv2_8473
block3_conv3_8476
block3_conv3_8478
block3_conv4_8481
block3_conv4_8483
block4_conv1_8488
block4_conv1_8490
block4_conv2_8493
block4_conv2_8495
block4_conv3_8498
block4_conv3_8500
block4_conv4_8503
block4_conv4_8505
block5_conv1_8562
block5_conv1_8564
block5_conv2_8571
block5_conv2_8573
identity

identity_1	

identity_2	

identity_3	

identity_4	Ђ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block3_conv4/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block4_conv4/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCall
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_72972+
)tf_op_layer_strided_slice/PartitionedCall
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_73112%
#tf_op_layer_BiasAdd/PartitionedCallУ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_8442block1_conv1_8444*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_69892&
$block1_conv1/StatefulPartitionedCallФ
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8447block1_conv2_8449*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_70112&
$block1_conv2/StatefulPartitionedCallс
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_73362/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallЮ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_8454block2_conv1_8456*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_70332&
$block2_conv1/StatefulPartitionedCallХ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8459block2_conv2_8461*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_70552&
$block2_conv2/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_736421
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallа
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:0block3_conv1_8466block3_conv1_8468*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_70772&
$block3_conv1/StatefulPartitionedCallХ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8471block3_conv2_8473*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_70992&
$block3_conv2/StatefulPartitionedCallХ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8476block3_conv3_8478*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_71212&
$block3_conv3/StatefulPartitionedCallХ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8481block3_conv4_8483*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_71432&
$block3_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_740221
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallа
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:0block4_conv1_8488block4_conv1_8490*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_71652&
$block4_conv1/StatefulPartitionedCallХ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_8493block4_conv2_8495*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_71872&
$block4_conv2/StatefulPartitionedCallХ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_8498block4_conv3_8500*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_72092&
$block4_conv3/StatefulPartitionedCallХ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_8503block4_conv4_8505*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_72312&
$block4_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_744021
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallљ
#tf_op_layer_Shape_7/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_74562%
#tf_op_layer_Shape_7/PartitionedCallљ
#tf_op_layer_Shape_5/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_74692%
#tf_op_layer_Shape_5/PartitionedCallљ
#tf_op_layer_Shape_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_74822%
#tf_op_layer_Shape_3/PartitionedCallї
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_74952%
#tf_op_layer_Shape_1/PartitionedCall
,tf_op_layer_strided_slice_10/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_75112.
,tf_op_layer_strided_slice_10/PartitionedCall
+tf_op_layer_strided_slice_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_75272-
+tf_op_layer_strided_slice_7/PartitionedCall
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_75432-
+tf_op_layer_strided_slice_4/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_75592-
+tf_op_layer_strided_slice_1/PartitionedCallю
#tf_op_layer_Shape_6/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_75722%
#tf_op_layer_Shape_6/PartitionedCallџ
#tf_op_layer_Range_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_10/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_75872%
#tf_op_layer_Range_3/PartitionedCallю
#tf_op_layer_Shape_4/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_76002%
#tf_op_layer_Shape_4/PartitionedCallў
#tf_op_layer_Range_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_7/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_76152%
#tf_op_layer_Range_2/PartitionedCallю
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_76282%
#tf_op_layer_Shape_2/PartitionedCallў
#tf_op_layer_Range_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_76432%
#tf_op_layer_Range_1/PartitionedCallш
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
CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_76562#
!tf_op_layer_Shape/PartitionedCallј
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_76712#
!tf_op_layer_Range/PartitionedCall
,tf_op_layer_strided_slice_12/PartitionedCallPartitionedCall,tf_op_layer_Shape_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_76872.
,tf_op_layer_strided_slice_12/PartitionedCall
,tf_op_layer_strided_slice_11/PartitionedCallPartitionedCall,tf_op_layer_Range_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_77032.
,tf_op_layer_strided_slice_11/PartitionedCall
+tf_op_layer_strided_slice_9/PartitionedCallPartitionedCall,tf_op_layer_Shape_4/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_77192-
+tf_op_layer_strided_slice_9/PartitionedCall
+tf_op_layer_strided_slice_8/PartitionedCallPartitionedCall,tf_op_layer_Range_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_77352-
+tf_op_layer_strided_slice_8/PartitionedCall
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_77512-
+tf_op_layer_strided_slice_6/PartitionedCall
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Range_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_77672-
+tf_op_layer_strided_slice_5/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_77832-
+tf_op_layer_strided_slice_3/PartitionedCall
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_77992-
+tf_op_layer_strided_slice_2/PartitionedCallч
)tf_op_layer_BroadcastTo_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_11/PartitionedCall:output:0,tf_op_layer_Shape_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_78132+
)tf_op_layer_BroadcastTo_3/PartitionedCallя
"tf_op_layer_Prod_6/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_12/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_78282$
"tf_op_layer_Prod_6/PartitionedCallц
)tf_op_layer_BroadcastTo_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_8/PartitionedCall:output:0,tf_op_layer_Shape_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_78422+
)tf_op_layer_BroadcastTo_2/PartitionedCallю
"tf_op_layer_Prod_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_9/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_78572$
"tf_op_layer_Prod_4/PartitionedCallц
)tf_op_layer_BroadcastTo_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0,tf_op_layer_Shape_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_78712+
)tf_op_layer_BroadcastTo_1/PartitionedCallю
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_78862$
"tf_op_layer_Prod_2/PartitionedCallр
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_79002)
'tf_op_layer_BroadcastTo/PartitionedCallш
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
CPU

GPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_79152"
 tf_op_layer_Prod/PartitionedCallЫ
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_3/PartitionedCall:output:0+tf_op_layer_Prod_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_79292#
!tf_op_layer_Mul_3/PartitionedCallЫ
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_2/PartitionedCall:output:0+tf_op_layer_Prod_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_79442#
!tf_op_layer_Mul_2/PartitionedCallЫ
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_1/PartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_79592#
!tf_op_layer_Mul_1/PartitionedCallС
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_79742!
tf_op_layer_Mul/PartitionedCallЮ
#tf_op_layer_AddV2_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*tf_op_layer_Mul_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_79892%
#tf_op_layer_AddV2_3/PartitionedCallъ
"tf_op_layer_Prod_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_80042$
"tf_op_layer_Prod_7/PartitionedCallЮ
#tf_op_layer_AddV2_2/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_80182%
#tf_op_layer_AddV2_2/PartitionedCallъ
"tf_op_layer_Prod_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_80332$
"tf_op_layer_Prod_5/PartitionedCallЮ
#tf_op_layer_AddV2_1/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*tf_op_layer_Mul_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_80472%
#tf_op_layer_AddV2_1/PartitionedCallъ
"tf_op_layer_Prod_3/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_80622$
"tf_op_layer_Prod_3/PartitionedCallУ
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_80762#
!tf_op_layer_AddV2/PartitionedCallъ
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_80912$
"tf_op_layer_Prod_1/PartitionedCallЊ
%tf_op_layer_Reshape_3/PartitionedCallPartitionedCall,tf_op_layer_AddV2_3/PartitionedCall:output:0+tf_op_layer_Prod_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_81052'
%tf_op_layer_Reshape_3/PartitionedCallЊ
%tf_op_layer_Reshape_2/PartitionedCallPartitionedCall,tf_op_layer_AddV2_2/PartitionedCall:output:0+tf_op_layer_Prod_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_81202'
%tf_op_layer_Reshape_2/PartitionedCallЊ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall,tf_op_layer_AddV2_1/PartitionedCall:output:0+tf_op_layer_Prod_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_81352'
%tf_op_layer_Reshape_1/PartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_81502%
#tf_op_layer_Reshape/PartitionedCallР
*tf_op_layer_UnravelIndex_3/PartitionedCallPartitionedCall.tf_op_layer_Reshape_3/PartitionedCall:output:0,tf_op_layer_Shape_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_81652,
*tf_op_layer_UnravelIndex_3/PartitionedCallР
*tf_op_layer_UnravelIndex_2/PartitionedCallPartitionedCall.tf_op_layer_Reshape_2/PartitionedCall:output:0,tf_op_layer_Shape_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_81802,
*tf_op_layer_UnravelIndex_2/PartitionedCallР
*tf_op_layer_UnravelIndex_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_81952,
*tf_op_layer_UnravelIndex_1/PartitionedCallЖ
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_82102*
(tf_op_layer_UnravelIndex/PartitionedCallа
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:0block5_conv1_8562block5_conv1_8564*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_72532&
$block5_conv1/StatefulPartitionedCall
'tf_op_layer_Transpose_3/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_82302)
'tf_op_layer_Transpose_3/PartitionedCall
'tf_op_layer_Transpose_2/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_82442)
'tf_op_layer_Transpose_2/PartitionedCall
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_82582)
'tf_op_layer_Transpose_1/PartitionedCall
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_82722'
%tf_op_layer_Transpose/PartitionedCallХ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_8571block5_conv2_8573*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_72752&
$block5_conv2/StatefulPartitionedCallО
IdentityIdentity-block5_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityЈ

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1Њ

Identity_2Identity0tf_op_layer_Transpose_1/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2Њ

Identity_3Identity0tf_op_layer_Transpose_2/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3Њ

Identity_4Identity0tf_op_layer_Transpose_3/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
j
>__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_fn_9586

inputs
identity

identity_1	
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_73642
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityPartitionedCall:output:1*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
q
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_7527

inputs	
identity	x
strided_slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/begint
strided_slice_7/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/end|
strided_slice_7/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stridesя
strided_slice_7StridedSliceinputsstrided_slice_7/begin:output:0strided_slice_7/end:output:0 strided_slice_7/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7[
IdentityIdentitystrided_slice_7:output:0*
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

i
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_7643

inputs	
identity	`
Range_1/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_1/start`
Range_1/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_1/delta
Range_1RangeRange_1/start:output:0inputsRange_1/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_1`
IdentityIdentityRange_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Д

Ў
F__inference_block1_conv1_layer_call_and_return_conditional_losses_6989

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ау

?__inference_model_layer_call_and_return_conditional_losses_9396

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource
identity

identity_1	

identity_2	

identity_3	

identity_4	Џ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tf_op_layer_strided_slice/strided_slice/beginЋ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*
valueB"        2-
+tf_op_layer_strided_slice/strided_slice/endГ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*
valueB"   џџџџ21
/tf_op_layer_strided_slice/strided_slice/stridesБ
'tf_op_layer_strided_slice/strided_sliceStridedSliceinputs6tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

begin_mask*
ellipsis_mask*
end_mask2)
'tf_op_layer_strided_slice/strided_slice
 tf_op_layer_BiasAdd/BiasAdd/biasConst*
_output_shapes
:*
dtype0*!
valueB"ХрЯТйщТ)\їТ2"
 tf_op_layer_BiasAdd/BiasAdd/bias§
tf_op_layer_BiasAdd/BiasAddBiasAdd0tf_op_layer_strided_slice/strided_slice:output:0)tf_op_layer_BiasAdd/BiasAdd/bias:output:0*
T0*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_BiasAdd/BiasAddМ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpњ
block1_conv1/Conv2DConv2D$tf_op_layer_BiasAdd/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
block1_conv1/Conv2DГ
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOpЮ
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv1/BiasAdd
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv1/ReluМ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpѕ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
block1_conv2/Conv2DГ
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOpЮ
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv2/BiasAdd
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
block1_conv2/Reluл
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxMaxPoolWithArgmaxblock1_conv2/Relu:activations:0*
T0*
_cloned(*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
ksize
*
paddingSAME*
strides
21
/tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmaxН
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp
block2_conv1/Conv2DConv2D8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block2_conv1/Conv2DД
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOpЯ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv1/ReluО
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpі
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block2_conv2/Conv2DД
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOpЯ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block2_conv2/Reluх
3tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1MaxPoolWithArgmaxblock2_conv2/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1О
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp
block3_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv1/Conv2DД
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOpЯ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv1/BiasAdd
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv1/ReluО
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpі
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv2/Conv2DД
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOpЯ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv2/BiasAdd
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv2/ReluО
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpі
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv3/Conv2DД
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOpЯ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv3/BiasAdd
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv3/ReluО
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpі
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block3_conv4/Conv2DД
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOpЯ
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv4/BiasAdd
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block3_conv4/Reluх
3tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2MaxPoolWithArgmaxblock3_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2О
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp
block4_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv1/Conv2DД
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOpЯ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv1/BiasAdd
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv1/ReluО
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpі
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv2/Conv2DД
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOpЯ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv2/BiasAdd
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv2/ReluО
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpі
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv3/Conv2DД
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOpЯ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv3/BiasAdd
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv3/ReluО
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpі
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block4_conv4/Conv2DД
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOpЯ
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv4/BiasAdd
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block4_conv4/Reluх
3tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3MaxPoolWithArgmaxblock4_conv4/Relu:activations:0*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
25
3tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3Х
tf_op_layer_Shape_7/Shape_7Shape<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_7/Shape_7Х
tf_op_layer_Shape_5/Shape_5Shape<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_5/Shape_5Х
tf_op_layer_Shape_3/Shape_3Shape<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_3/Shape_3С
tf_op_layer_Shape_1/Shape_1Shape8tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_1/Shape_1Д
3tf_op_layer_strided_slice_10/strided_slice_10/beginConst*
_output_shapes
:*
dtype0*
valueB: 25
3tf_op_layer_strided_slice_10/strided_slice_10/beginА
1tf_op_layer_strided_slice_10/strided_slice_10/endConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_10/strided_slice_10/endИ
5tf_op_layer_strided_slice_10/strided_slice_10/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5tf_op_layer_strided_slice_10/strided_slice_10/stridesЃ
-tf_op_layer_strided_slice_10/strided_slice_10StridedSlice$tf_op_layer_Shape_7/Shape_7:output:0<tf_op_layer_strided_slice_10/strided_slice_10/begin:output:0:tf_op_layer_strided_slice_10/strided_slice_10/end:output:0>tf_op_layer_strided_slice_10/strided_slice_10/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2/
-tf_op_layer_strided_slice_10/strided_slice_10А
1tf_op_layer_strided_slice_7/strided_slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_7/strided_slice_7/beginЌ
/tf_op_layer_strided_slice_7/strided_slice_7/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_7/strided_slice_7/endД
3tf_op_layer_strided_slice_7/strided_slice_7/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_7/strided_slice_7/strides
+tf_op_layer_strided_slice_7/strided_slice_7StridedSlice$tf_op_layer_Shape_5/Shape_5:output:0:tf_op_layer_strided_slice_7/strided_slice_7/begin:output:08tf_op_layer_strided_slice_7/strided_slice_7/end:output:0<tf_op_layer_strided_slice_7/strided_slice_7/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_7/strided_slice_7А
1tf_op_layer_strided_slice_4/strided_slice_4/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_4/strided_slice_4/beginЌ
/tf_op_layer_strided_slice_4/strided_slice_4/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_4/strided_slice_4/endД
3tf_op_layer_strided_slice_4/strided_slice_4/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_4/strided_slice_4/strides
+tf_op_layer_strided_slice_4/strided_slice_4StridedSlice$tf_op_layer_Shape_3/Shape_3:output:0:tf_op_layer_strided_slice_4/strided_slice_4/begin:output:08tf_op_layer_strided_slice_4/strided_slice_4/end:output:0<tf_op_layer_strided_slice_4/strided_slice_4/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_4/strided_slice_4А
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_1/strided_slice_1/beginЌ
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
_output_shapes
:*
dtype0*
valueB:21
/tf_op_layer_strided_slice_1/strided_slice_1/endД
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_1/strided_slice_1/strides
+tf_op_layer_strided_slice_1/strided_slice_1StridedSlice$tf_op_layer_Shape_1/Shape_1:output:0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2-
+tf_op_layer_strided_slice_1/strided_slice_1Ј
tf_op_layer_Shape_6/Shape_6Shapeblock4_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_6/Shape_6
!tf_op_layer_Range_3/Range_3/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_3/Range_3/start
!tf_op_layer_Range_3/Range_3/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_3/Range_3/delta
tf_op_layer_Range_3/Range_3Range*tf_op_layer_Range_3/Range_3/start:output:06tf_op_layer_strided_slice_10/strided_slice_10:output:0*tf_op_layer_Range_3/Range_3/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_3/Range_3Ј
tf_op_layer_Shape_4/Shape_4Shapeblock3_conv4/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_4/Shape_4
!tf_op_layer_Range_2/Range_2/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_2/Range_2/start
!tf_op_layer_Range_2/Range_2/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_2/Range_2/delta
tf_op_layer_Range_2/Range_2Range*tf_op_layer_Range_2/Range_2/start:output:04tf_op_layer_strided_slice_7/strided_slice_7:output:0*tf_op_layer_Range_2/Range_2/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_2/Range_2Ј
tf_op_layer_Shape_2/Shape_2Shapeblock2_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape_2/Shape_2
!tf_op_layer_Range_1/Range_1/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!tf_op_layer_Range_1/Range_1/start
!tf_op_layer_Range_1/Range_1/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!tf_op_layer_Range_1/Range_1/delta
tf_op_layer_Range_1/Range_1Range*tf_op_layer_Range_1/Range_1/start:output:04tf_op_layer_strided_slice_4/strided_slice_4:output:0*tf_op_layer_Range_1/Range_1/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range_1/Range_1 
tf_op_layer_Shape/ShapeShapeblock1_conv2/Relu:activations:0*
T0*
_cloned(*
_output_shapes
:*
out_type0	2
tf_op_layer_Shape/Shape
tf_op_layer_Range/Range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
tf_op_layer_Range/Range/start
tf_op_layer_Range/Range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
tf_op_layer_Range/Range/delta
tf_op_layer_Range/RangeRange&tf_op_layer_Range/Range/start:output:04tf_op_layer_strided_slice_1/strided_slice_1:output:0&tf_op_layer_Range/Range/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Range/RangeД
3tf_op_layer_strided_slice_12/strided_slice_12/beginConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_12/strided_slice_12/beginА
1tf_op_layer_strided_slice_12/strided_slice_12/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1tf_op_layer_strided_slice_12/strided_slice_12/endИ
5tf_op_layer_strided_slice_12/strided_slice_12/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5tf_op_layer_strided_slice_12/strided_slice_12/strides
-tf_op_layer_strided_slice_12/strided_slice_12StridedSlice$tf_op_layer_Shape_6/Shape_6:output:0<tf_op_layer_strided_slice_12/strided_slice_12/begin:output:0:tf_op_layer_strided_slice_12/strided_slice_12/end:output:0>tf_op_layer_strided_slice_12/strided_slice_12/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2/
-tf_op_layer_strided_slice_12/strided_slice_12У
3tf_op_layer_strided_slice_11/strided_slice_11/beginConst*
_output_shapes
:*
dtype0*%
valueB"                25
3tf_op_layer_strided_slice_11/strided_slice_11/beginП
1tf_op_layer_strided_slice_11/strided_slice_11/endConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_11/strided_slice_11/endЧ
5tf_op_layer_strided_slice_11/strided_slice_11/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            27
5tf_op_layer_strided_slice_11/strided_slice_11/stridesЮ
-tf_op_layer_strided_slice_11/strided_slice_11StridedSlice$tf_op_layer_Range_3/Range_3:output:0<tf_op_layer_strided_slice_11/strided_slice_11/begin:output:0:tf_op_layer_strided_slice_11/strided_slice_11/end:output:0>tf_op_layer_strided_slice_11/strided_slice_11/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2/
-tf_op_layer_strided_slice_11/strided_slice_11А
1tf_op_layer_strided_slice_9/strided_slice_9/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_9/strided_slice_9/beginЌ
/tf_op_layer_strided_slice_9/strided_slice_9/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_9/strided_slice_9/endД
3tf_op_layer_strided_slice_9/strided_slice_9/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_9/strided_slice_9/strides
+tf_op_layer_strided_slice_9/strided_slice_9StridedSlice$tf_op_layer_Shape_4/Shape_4:output:0:tf_op_layer_strided_slice_9/strided_slice_9/begin:output:08tf_op_layer_strided_slice_9/strided_slice_9/end:output:0<tf_op_layer_strided_slice_9/strided_slice_9/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_9/strided_slice_9П
1tf_op_layer_strided_slice_8/strided_slice_8/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_8/strided_slice_8/beginЛ
/tf_op_layer_strided_slice_8/strided_slice_8/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_8/strided_slice_8/endУ
3tf_op_layer_strided_slice_8/strided_slice_8/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_8/strided_slice_8/stridesФ
+tf_op_layer_strided_slice_8/strided_slice_8StridedSlice$tf_op_layer_Range_2/Range_2:output:0:tf_op_layer_strided_slice_8/strided_slice_8/begin:output:08tf_op_layer_strided_slice_8/strided_slice_8/end:output:0<tf_op_layer_strided_slice_8/strided_slice_8/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_8/strided_slice_8А
1tf_op_layer_strided_slice_6/strided_slice_6/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_6/strided_slice_6/beginЌ
/tf_op_layer_strided_slice_6/strided_slice_6/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_6/strided_slice_6/endД
3tf_op_layer_strided_slice_6/strided_slice_6/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_6/strided_slice_6/strides
+tf_op_layer_strided_slice_6/strided_slice_6StridedSlice$tf_op_layer_Shape_2/Shape_2:output:0:tf_op_layer_strided_slice_6/strided_slice_6/begin:output:08tf_op_layer_strided_slice_6/strided_slice_6/end:output:0<tf_op_layer_strided_slice_6/strided_slice_6/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_6/strided_slice_6П
1tf_op_layer_strided_slice_5/strided_slice_5/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_5/strided_slice_5/beginЛ
/tf_op_layer_strided_slice_5/strided_slice_5/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_5/strided_slice_5/endУ
3tf_op_layer_strided_slice_5/strided_slice_5/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_5/strided_slice_5/stridesФ
+tf_op_layer_strided_slice_5/strided_slice_5StridedSlice$tf_op_layer_Range_1/Range_1:output:0:tf_op_layer_strided_slice_5/strided_slice_5/begin:output:08tf_op_layer_strided_slice_5/strided_slice_5/end:output:0<tf_op_layer_strided_slice_5/strided_slice_5/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_5/strided_slice_5А
1tf_op_layer_strided_slice_3/strided_slice_3/beginConst*
_output_shapes
:*
dtype0*
valueB:23
1tf_op_layer_strided_slice_3/strided_slice_3/beginЌ
/tf_op_layer_strided_slice_3/strided_slice_3/endConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf_op_layer_strided_slice_3/strided_slice_3/endД
3tf_op_layer_strided_slice_3/strided_slice_3/stridesConst*
_output_shapes
:*
dtype0*
valueB:25
3tf_op_layer_strided_slice_3/strided_slice_3/strides
+tf_op_layer_strided_slice_3/strided_slice_3StridedSlice tf_op_layer_Shape/Shape:output:0:tf_op_layer_strided_slice_3/strided_slice_3/begin:output:08tf_op_layer_strided_slice_3/strided_slice_3/end:output:0<tf_op_layer_strided_slice_3/strided_slice_3/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
:*
end_mask2-
+tf_op_layer_strided_slice_3/strided_slice_3П
1tf_op_layer_strided_slice_2/strided_slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                23
1tf_op_layer_strided_slice_2/strided_slice_2/beginЛ
/tf_op_layer_strided_slice_2/strided_slice_2/endConst*
_output_shapes
:*
dtype0*%
valueB"                21
/tf_op_layer_strided_slice_2/strided_slice_2/endУ
3tf_op_layer_strided_slice_2/strided_slice_2/stridesConst*
_output_shapes
:*
dtype0*%
valueB"            25
3tf_op_layer_strided_slice_2/strided_slice_2/stridesР
+tf_op_layer_strided_slice_2/strided_slice_2StridedSlice tf_op_layer_Range/Range:output:0:tf_op_layer_strided_slice_2/strided_slice_2/begin:output:08tf_op_layer_strided_slice_2/strided_slice_2/end:output:0<tf_op_layer_strided_slice_2/strided_slice_2/strides:output:0*
Index0*
T0	*
_cloned(*/
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
new_axis_mask2-
+tf_op_layer_strided_slice_2/strided_slice_2Ї
'tf_op_layer_BroadcastTo_3/BroadcastTo_3BroadcastTo6tf_op_layer_strided_slice_11/strided_slice_11:output:0$tf_op_layer_Shape_7/Shape_7:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_3/BroadcastTo_3Є
+tf_op_layer_Prod_6/Prod_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_6/Prod_6/reduction_indicesм
tf_op_layer_Prod_6/Prod_6Prod6tf_op_layer_strided_slice_12/strided_slice_12:output:04tf_op_layer_Prod_6/Prod_6/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_6/Prod_6Ѕ
'tf_op_layer_BroadcastTo_2/BroadcastTo_2BroadcastTo4tf_op_layer_strided_slice_8/strided_slice_8:output:0$tf_op_layer_Shape_5/Shape_5:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_2/BroadcastTo_2Є
+tf_op_layer_Prod_4/Prod_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_4/Prod_4/reduction_indicesк
tf_op_layer_Prod_4/Prod_4Prod4tf_op_layer_strided_slice_9/strided_slice_9:output:04tf_op_layer_Prod_4/Prod_4/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_4/Prod_4Ѕ
'tf_op_layer_BroadcastTo_1/BroadcastTo_1BroadcastTo4tf_op_layer_strided_slice_5/strided_slice_5:output:0$tf_op_layer_Shape_3/Shape_3:output:0*
T0	*

Tidx0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'tf_op_layer_BroadcastTo_1/BroadcastTo_1Є
+tf_op_layer_Prod_2/Prod_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_2/Prod_2/reduction_indicesк
tf_op_layer_Prod_2/Prod_2Prod4tf_op_layer_strided_slice_6/strided_slice_6:output:04tf_op_layer_Prod_2/Prod_2/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod_2/Prod_2
#tf_op_layer_BroadcastTo/BroadcastToBroadcastTo4tf_op_layer_strided_slice_2/strided_slice_2:output:0$tf_op_layer_Shape_1/Shape_1:output:0*
T0	*

Tidx0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2%
#tf_op_layer_BroadcastTo/BroadcastTo
'tf_op_layer_Prod/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2)
'tf_op_layer_Prod/Prod/reduction_indicesЮ
tf_op_layer_Prod/ProdProd4tf_op_layer_strided_slice_3/strided_slice_3:output:00tf_op_layer_Prod/Prod/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
tf_op_layer_Prod/Prodы
tf_op_layer_Mul_3/Mul_3Mul0tf_op_layer_BroadcastTo_3/BroadcastTo_3:output:0"tf_op_layer_Prod_6/Prod_6:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_3/Mul_3ы
tf_op_layer_Mul_2/Mul_2Mul0tf_op_layer_BroadcastTo_2/BroadcastTo_2:output:0"tf_op_layer_Prod_4/Prod_4:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_2/Mul_2ы
tf_op_layer_Mul_1/Mul_1Mul0tf_op_layer_BroadcastTo_1/BroadcastTo_1:output:0"tf_op_layer_Prod_2/Prod_2:output:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1к
tf_op_layer_Mul/MulMul,tf_op_layer_BroadcastTo/BroadcastTo:output:0tf_op_layer_Prod/Prod:output:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
tf_op_layer_Mul/Mulњ
tf_op_layer_AddV2_3/AddV2_3AddV2<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:argmax:0tf_op_layer_Mul_3/Mul_3:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Prod_7/Prod_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_7/Prod_7/reduction_indicesп
tf_op_layer_Prod_7/Prod_7Prod$tf_op_layer_Shape_7/Shape_7:output:04tf_op_layer_Prod_7/Prod_7/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_7/Prod_7њ
tf_op_layer_AddV2_2/AddV2_2AddV2<tf_op_layer_MaxPoolWithArgmax_2/MaxPoolWithArgmax_2:argmax:0tf_op_layer_Mul_2/Mul_2:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Є
+tf_op_layer_Prod_5/Prod_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_5/Prod_5/reduction_indicesп
tf_op_layer_Prod_5/Prod_5Prod$tf_op_layer_Shape_5/Shape_5:output:04tf_op_layer_Prod_5/Prod_5/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_5/Prod_5њ
tf_op_layer_AddV2_1/AddV2_1AddV2<tf_op_layer_MaxPoolWithArgmax_1/MaxPoolWithArgmax_1:argmax:0tf_op_layer_Mul_1/Mul_1:z:0*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1Є
+tf_op_layer_Prod_3/Prod_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_3/Prod_3/reduction_indicesп
tf_op_layer_Prod_3/Prod_3Prod$tf_op_layer_Shape_3/Shape_3:output:04tf_op_layer_Prod_3/Prod_3/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_3/Prod_3щ
tf_op_layer_AddV2/AddV2AddV28tf_op_layer_MaxPoolWithArgmax/MaxPoolWithArgmax:argmax:0tf_op_layer_Mul/Mul:z:0*
T0	*
_cloned(*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
tf_op_layer_AddV2/AddV2Є
+tf_op_layer_Prod_1/Prod_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Prod_1/Prod_1/reduction_indicesп
tf_op_layer_Prod_1/Prod_1Prod$tf_op_layer_Shape_1/Shape_1:output:04tf_op_layer_Prod_1/Prod_1/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
tf_op_layer_Prod_1/Prod_1н
tf_op_layer_Reshape_3/Reshape_3Reshapetf_op_layer_AddV2_3/AddV2_3:z:0"tf_op_layer_Prod_7/Prod_7:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_3/Reshape_3н
tf_op_layer_Reshape_2/Reshape_2Reshapetf_op_layer_AddV2_2/AddV2_2:z:0"tf_op_layer_Prod_5/Prod_5:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_2/Reshape_2н
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_AddV2_1/AddV2_1:z:0"tf_op_layer_Prod_3/Prod_3:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Reshape_1/Reshape_1б
tf_op_layer_Reshape/ReshapeReshapetf_op_layer_AddV2/AddV2:z:0"tf_op_layer_Prod_1/Prod_1:output:0*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Reshape/Reshapeњ
)tf_op_layer_UnravelIndex_3/UnravelIndex_3UnravelIndex(tf_op_layer_Reshape_3/Reshape_3:output:0$tf_op_layer_Shape_6/Shape_6:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_3/UnravelIndex_3њ
)tf_op_layer_UnravelIndex_2/UnravelIndex_2UnravelIndex(tf_op_layer_Reshape_2/Reshape_2:output:0$tf_op_layer_Shape_4/Shape_4:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_2/UnravelIndex_2њ
)tf_op_layer_UnravelIndex_1/UnravelIndex_1UnravelIndex(tf_op_layer_Reshape_1/Reshape_1:output:0$tf_op_layer_Shape_2/Shape_2:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)tf_op_layer_UnravelIndex_1/UnravelIndex_1ъ
%tf_op_layer_UnravelIndex/UnravelIndexUnravelIndex$tf_op_layer_Reshape/Reshape:output:0 tf_op_layer_Shape/Shape:output:0*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2'
%tf_op_layer_UnravelIndex/UnravelIndexО
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp
block5_conv1/Conv2DConv2D<tf_op_layer_MaxPoolWithArgmax_3/MaxPoolWithArgmax_3:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block5_conv1/Conv2DД
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOpЯ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv1/BiasAdd
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv1/ReluЅ
(tf_op_layer_Transpose_3/Transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_3/Transpose_3/permџ
#tf_op_layer_Transpose_3/Transpose_3	Transpose2tf_op_layer_UnravelIndex_3/UnravelIndex_3:output:01tf_op_layer_Transpose_3/Transpose_3/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_3/Transpose_3Ѕ
(tf_op_layer_Transpose_2/Transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_2/Transpose_2/permџ
#tf_op_layer_Transpose_2/Transpose_2	Transpose2tf_op_layer_UnravelIndex_2/UnravelIndex_2:output:01tf_op_layer_Transpose_2/Transpose_2/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_2/Transpose_2Ѕ
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_1/Transpose_1/permџ
#tf_op_layer_Transpose_1/Transpose_1	Transpose2tf_op_layer_UnravelIndex_1/UnravelIndex_1:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2%
#tf_op_layer_Transpose_1/Transpose_1
$tf_op_layer_Transpose/Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$tf_op_layer_Transpose/Transpose/permя
tf_op_layer_Transpose/Transpose	Transpose.tf_op_layer_UnravelIndex/UnravelIndex:output:0-tf_op_layer_Transpose/Transpose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Transpose/TransposeО
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpі
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
block5_conv2/Conv2DД
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOpЯ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv2/BiasAdd
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
block5_conv2/Relu
IdentityIdentityblock5_conv2/Relu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity{

Identity_1Identity#tf_op_layer_Transpose/Transpose:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity'tf_op_layer_Transpose_1/Transpose_1:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity'tf_op_layer_Transpose_2/Transpose_2:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity'tf_op_layer_Transpose_3/Transpose_3:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::::::::::::::::::::::::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ы
h
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_8004

inputs	
identity	~
Prod_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_7/reduction_indices
Prod_7Prodinputs!Prod_7/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_7V
IdentityIdentityProd_7:output:0*
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
љ
w
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_8018

inputs	
inputs_1	
identity	
AddV2_2AddV2inputsinputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_2z
IdentityIdentityAddV2_2:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:rn
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё
g
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_9723

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Б
x
L__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_10008
inputs_0	
inputs_1	
identity	
Mul_1Mulinputs_0inputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_1
IdentityIdentity	Mul_1:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :t p
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:@<

_output_shapes
: 
"
_user_specified_name
inputs/1
і
z
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_10136
inputs_0	
inputs_1	
identity	|
ReshapeReshapeinputs_0inputs_1*
T0	*
Tshape0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Reshape`
IdentityIdentityReshape:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
ш

+__inference_block2_conv1_layer_call_fn_7043

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_70332
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ї
f
:__inference_tf_op_layer_UnravelIndex_1_layer_call_fn_10202
inputs_0	
inputs_1	
identity	Р
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_81952
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Ј
u
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_7929

inputs	
inputs_1	
identity	
Mul_3Mulinputsinputs_1*
T0	*
_cloned(*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Mul_3
IdentityIdentity	Mul_3:z:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: :r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs
с
V
:__inference_tf_op_layer_strided_slice_6_layer_call_fn_9846

inputs	
identity	Ї
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_77512
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

N
2__inference_tf_op_layer_Shape_1_layer_call_fn_9624

inputs	
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_74952
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

V
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_9807

inputs	
identity	М
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_77992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0	*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н

Ў
F__inference_block3_conv2_layer_call_and_return_conditional_losses_7099

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ё
N
2__inference_tf_op_layer_Shape_4_layer_call_fn_9772

inputs
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_76002
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_9607

inputs
identity

identity_1	
MaxPoolWithArgmax_3MaxPoolWithArgmaxinputs*
T0*
_cloned(*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax_3
IdentityIdentityMaxPoolWithArgmax_3:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityMaxPoolWithArgmax_3:argmax:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

S
7__inference_tf_op_layer_Transpose_1_layer_call_fn_10248

inputs	
identity	А
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_82582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ

+__inference_block5_conv1_layer_call_fn_7263

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_72532
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ф
l
P__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_10232

inputs	
identity	q
Transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose/perm
	Transpose	TransposeinputsTranspose/perm:output:0*
T0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
	Transposea
IdentityIdentityTranspose:y:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Q
5__inference_tf_op_layer_Transpose_layer_call_fn_10237

inputs	
identity	Ў
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_82722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ

+__inference_block4_conv3_layer_call_fn_7219

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_72092
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
w
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_7989

inputs	
inputs_1	
identity	
AddV2_3AddV2inputsinputs_1*
T0	*
_cloned(*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
AddV2_3z
IdentityIdentityAddV2_3:z:0*
T0	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:rn
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ы
h
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_8033

inputs	
identity	~
Prod_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_5/reduction_indices
Prod_5Prodinputs!Prod_5/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
:*
	keep_dims(2
Prod_5V
IdentityIdentityProd_5:output:0*
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
Ѓ
f
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_9916

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
Б
h
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_7857

inputs	
identity	~
Prod_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
Prod_4/reduction_indicess
Prod_4Prodinputs!Prod_4/reduction_indices:output:0*
T0	*
_cloned(*
_output_shapes
: 2
Prod_4R
IdentityIdentityProd_4:output:0*
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
ІГ
Х

?__inference_model_layer_call_and_return_conditional_losses_8290
input_2
block1_conv1_7319
block1_conv1_7321
block1_conv2_7324
block1_conv2_7326
block2_conv1_7347
block2_conv1_7349
block2_conv2_7352
block2_conv2_7354
block3_conv1_7375
block3_conv1_7377
block3_conv2_7380
block3_conv2_7382
block3_conv3_7385
block3_conv3_7387
block3_conv4_7390
block3_conv4_7392
block4_conv1_7413
block4_conv1_7415
block4_conv2_7418
block4_conv2_7420
block4_conv3_7423
block4_conv3_7425
block4_conv4_7428
block4_conv4_7430
block5_conv1_8219
block5_conv1_8221
block5_conv2_8280
block5_conv2_8282
identity

identity_1	

identity_2	

identity_3	

identity_4	Ђ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block3_conv4/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block4_conv4/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCall
)tf_op_layer_strided_slice/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_72972+
)tf_op_layer_strided_slice/PartitionedCall
#tf_op_layer_BiasAdd/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_73112%
#tf_op_layer_BiasAdd/PartitionedCallУ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_BiasAdd/PartitionedCall:output:0block1_conv1_7319block1_conv1_7321*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_69892&
$block1_conv1/StatefulPartitionedCallФ
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_7324block1_conv2_7326*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_70112&
$block1_conv2/StatefulPartitionedCallс
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*n
_output_shapes\
Z:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*`
f[RY
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_73362/
-tf_op_layer_MaxPoolWithArgmax/PartitionedCallЮ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:0block2_conv1_7347block2_conv1_7349*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_70332&
$block2_conv1/StatefulPartitionedCallХ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_7352block2_conv2_7354*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_70552&
$block2_conv2/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_736421
/tf_op_layer_MaxPoolWithArgmax_1/PartitionedCallа
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:0block3_conv1_7375block3_conv1_7377*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_70772&
$block3_conv1/StatefulPartitionedCallХ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_7380block3_conv2_7382*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_70992&
$block3_conv2/StatefulPartitionedCallХ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_7385block3_conv3_7387*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_71212&
$block3_conv3/StatefulPartitionedCallХ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_7390block3_conv4_7392*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_71432&
$block3_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_740221
/tf_op_layer_MaxPoolWithArgmax_2/PartitionedCallа
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:0block4_conv1_7413block4_conv1_7415*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_71652&
$block4_conv1/StatefulPartitionedCallХ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_7418block4_conv2_7420*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_71872&
$block4_conv2/StatefulPartitionedCallХ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_7423block4_conv3_7425*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_72092&
$block4_conv3/StatefulPartitionedCallХ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_7428block4_conv4_7430*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_72312&
$block4_conv4/StatefulPartitionedCallщ
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2	*p
_output_shapes^
\:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_744021
/tf_op_layer_MaxPoolWithArgmax_3/PartitionedCallљ
#tf_op_layer_Shape_7/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_74562%
#tf_op_layer_Shape_7/PartitionedCallљ
#tf_op_layer_Shape_5/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_74692%
#tf_op_layer_Shape_5/PartitionedCallљ
#tf_op_layer_Shape_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_74822%
#tf_op_layer_Shape_3/PartitionedCallї
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_74952%
#tf_op_layer_Shape_1/PartitionedCall
,tf_op_layer_strided_slice_10/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_75112.
,tf_op_layer_strided_slice_10/PartitionedCall
+tf_op_layer_strided_slice_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_75272-
+tf_op_layer_strided_slice_7/PartitionedCall
+tf_op_layer_strided_slice_4/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_75432-
+tf_op_layer_strided_slice_4/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_75592-
+tf_op_layer_strided_slice_1/PartitionedCallю
#tf_op_layer_Shape_6/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_75722%
#tf_op_layer_Shape_6/PartitionedCallџ
#tf_op_layer_Range_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_10/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_75872%
#tf_op_layer_Range_3/PartitionedCallю
#tf_op_layer_Shape_4/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_76002%
#tf_op_layer_Shape_4/PartitionedCallў
#tf_op_layer_Range_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_7/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_76152%
#tf_op_layer_Range_2/PartitionedCallю
#tf_op_layer_Shape_2/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_76282%
#tf_op_layer_Shape_2/PartitionedCallў
#tf_op_layer_Range_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_4/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_76432%
#tf_op_layer_Range_1/PartitionedCallш
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
CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_76562#
!tf_op_layer_Shape/PartitionedCallј
!tf_op_layer_Range/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_76712#
!tf_op_layer_Range/PartitionedCall
,tf_op_layer_strided_slice_12/PartitionedCallPartitionedCall,tf_op_layer_Shape_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_76872.
,tf_op_layer_strided_slice_12/PartitionedCall
,tf_op_layer_strided_slice_11/PartitionedCallPartitionedCall,tf_op_layer_Range_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_77032.
,tf_op_layer_strided_slice_11/PartitionedCall
+tf_op_layer_strided_slice_9/PartitionedCallPartitionedCall,tf_op_layer_Shape_4/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_77192-
+tf_op_layer_strided_slice_9/PartitionedCall
+tf_op_layer_strided_slice_8/PartitionedCallPartitionedCall,tf_op_layer_Range_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_77352-
+tf_op_layer_strided_slice_8/PartitionedCall
+tf_op_layer_strided_slice_6/PartitionedCallPartitionedCall,tf_op_layer_Shape_2/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_77512-
+tf_op_layer_strided_slice_6/PartitionedCall
+tf_op_layer_strided_slice_5/PartitionedCallPartitionedCall,tf_op_layer_Range_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_77672-
+tf_op_layer_strided_slice_5/PartitionedCall
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_77832-
+tf_op_layer_strided_slice_3/PartitionedCall
+tf_op_layer_strided_slice_2/PartitionedCallPartitionedCall*tf_op_layer_Range/PartitionedCall:output:0*
Tin
2	*
Tout
2	*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_77992-
+tf_op_layer_strided_slice_2/PartitionedCallч
)tf_op_layer_BroadcastTo_3/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_11/PartitionedCall:output:0,tf_op_layer_Shape_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_78132+
)tf_op_layer_BroadcastTo_3/PartitionedCallя
"tf_op_layer_Prod_6/PartitionedCallPartitionedCall5tf_op_layer_strided_slice_12/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_78282$
"tf_op_layer_Prod_6/PartitionedCallц
)tf_op_layer_BroadcastTo_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_8/PartitionedCall:output:0,tf_op_layer_Shape_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_78422+
)tf_op_layer_BroadcastTo_2/PartitionedCallю
"tf_op_layer_Prod_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_9/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_78572$
"tf_op_layer_Prod_4/PartitionedCallц
)tf_op_layer_BroadcastTo_1/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_5/PartitionedCall:output:0,tf_op_layer_Shape_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_78712+
)tf_op_layer_BroadcastTo_1/PartitionedCallю
"tf_op_layer_Prod_2/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_6/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_78862$
"tf_op_layer_Prod_2/PartitionedCallр
'tf_op_layer_BroadcastTo/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_2/PartitionedCall:output:0,tf_op_layer_Shape_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_79002)
'tf_op_layer_BroadcastTo/PartitionedCallш
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
CPU

GPU2*0J 8*S
fNRL
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_79152"
 tf_op_layer_Prod/PartitionedCallЫ
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_3/PartitionedCall:output:0+tf_op_layer_Prod_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_79292#
!tf_op_layer_Mul_3/PartitionedCallЫ
!tf_op_layer_Mul_2/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_2/PartitionedCall:output:0+tf_op_layer_Prod_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_79442#
!tf_op_layer_Mul_2/PartitionedCallЫ
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall2tf_op_layer_BroadcastTo_1/PartitionedCall:output:0+tf_op_layer_Prod_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_79592#
!tf_op_layer_Mul_1/PartitionedCallС
tf_op_layer_Mul/PartitionedCallPartitionedCall0tf_op_layer_BroadcastTo/PartitionedCall:output:0)tf_op_layer_Prod/PartitionedCall:output:0*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_79742!
tf_op_layer_Mul/PartitionedCallЮ
#tf_op_layer_AddV2_3/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:1*tf_op_layer_Mul_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_79892%
#tf_op_layer_AddV2_3/PartitionedCallъ
"tf_op_layer_Prod_7/PartitionedCallPartitionedCall,tf_op_layer_Shape_7/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_80042$
"tf_op_layer_Prod_7/PartitionedCallЮ
#tf_op_layer_AddV2_2/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_2/PartitionedCall:output:1*tf_op_layer_Mul_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_80182%
#tf_op_layer_AddV2_2/PartitionedCallъ
"tf_op_layer_Prod_5/PartitionedCallPartitionedCall,tf_op_layer_Shape_5/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_80332$
"tf_op_layer_Prod_5/PartitionedCallЮ
#tf_op_layer_AddV2_1/PartitionedCallPartitionedCall8tf_op_layer_MaxPoolWithArgmax_1/PartitionedCall:output:1*tf_op_layer_Mul_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_80472%
#tf_op_layer_AddV2_1/PartitionedCallъ
"tf_op_layer_Prod_3/PartitionedCallPartitionedCall,tf_op_layer_Shape_3/PartitionedCall:output:0*
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_80622$
"tf_op_layer_Prod_3/PartitionedCallУ
!tf_op_layer_AddV2/PartitionedCallPartitionedCall6tf_op_layer_MaxPoolWithArgmax/PartitionedCall:output:1(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2		*
Tout
2	*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_80762#
!tf_op_layer_AddV2/PartitionedCallъ
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
CPU

GPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_80912$
"tf_op_layer_Prod_1/PartitionedCallЊ
%tf_op_layer_Reshape_3/PartitionedCallPartitionedCall,tf_op_layer_AddV2_3/PartitionedCall:output:0+tf_op_layer_Prod_7/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_81052'
%tf_op_layer_Reshape_3/PartitionedCallЊ
%tf_op_layer_Reshape_2/PartitionedCallPartitionedCall,tf_op_layer_AddV2_2/PartitionedCall:output:0+tf_op_layer_Prod_5/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_81202'
%tf_op_layer_Reshape_2/PartitionedCallЊ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall,tf_op_layer_AddV2_1/PartitionedCall:output:0+tf_op_layer_Prod_3/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_81352'
%tf_op_layer_Reshape_1/PartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall*tf_op_layer_AddV2/PartitionedCall:output:0+tf_op_layer_Prod_1/PartitionedCall:output:0*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_81502%
#tf_op_layer_Reshape/PartitionedCallР
*tf_op_layer_UnravelIndex_3/PartitionedCallPartitionedCall.tf_op_layer_Reshape_3/PartitionedCall:output:0,tf_op_layer_Shape_6/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_81652,
*tf_op_layer_UnravelIndex_3/PartitionedCallР
*tf_op_layer_UnravelIndex_2/PartitionedCallPartitionedCall.tf_op_layer_Reshape_2/PartitionedCall:output:0,tf_op_layer_Shape_4/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_81802,
*tf_op_layer_UnravelIndex_2/PartitionedCallР
*tf_op_layer_UnravelIndex_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0,tf_op_layer_Shape_2/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_81952,
*tf_op_layer_UnravelIndex_1/PartitionedCallЖ
(tf_op_layer_UnravelIndex/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*tf_op_layer_Shape/PartitionedCall:output:0*
Tin
2		*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_82102*
(tf_op_layer_UnravelIndex/PartitionedCallа
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall8tf_op_layer_MaxPoolWithArgmax_3/PartitionedCall:output:0block5_conv1_8219block5_conv1_8221*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_72532&
$block5_conv1/StatefulPartitionedCall
'tf_op_layer_Transpose_3/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_3/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_82302)
'tf_op_layer_Transpose_3/PartitionedCall
'tf_op_layer_Transpose_2/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_2/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_82442)
'tf_op_layer_Transpose_2/PartitionedCall
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall3tf_op_layer_UnravelIndex_1/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_82582)
'tf_op_layer_Transpose_1/PartitionedCall
%tf_op_layer_Transpose/PartitionedCallPartitionedCall1tf_op_layer_UnravelIndex/PartitionedCall:output:0*
Tin
2	*
Tout
2	*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_82722'
%tf_op_layer_Transpose/PartitionedCallХ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_8280block5_conv2_8282*
Tin
2*
Tout
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_72752&
$block5_conv2/StatefulPartitionedCallО
IdentityIdentity-block5_conv2/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityЈ

Identity_1Identity.tf_op_layer_Transpose/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1Њ

Identity_2Identity0tf_op_layer_Transpose_1/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2Њ

Identity_3Identity0tf_op_layer_Transpose_2/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3Њ

Identity_4Identity0tf_op_layer_Transpose_3/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ѓ
a
5__inference_tf_op_layer_Reshape_1_layer_call_fn_10154
inputs_0	
inputs_1	
identity	З
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*#
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_81352
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::l h
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
М
а
$__inference_model_layer_call_fn_9534

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1	

identity_2	

identity_3	

identity_4	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_87942
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
к

U__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_10196
inputs_0	
inputs_1	
identity	
UnravelIndex_1UnravelIndexinputs_0inputs_1*

Tidx0	*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
UnravelIndex_1k
IdentityIdentityUnravelIndex_1:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ::M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1

i
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_9779

inputs	
identity	`
Range_3/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_3/start`
Range_3/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_3/delta
Range_3RangeRange_3/start:output:0inputsRange_3/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_3`
IdentityIdentityRange_3:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
г
d
8__inference_tf_op_layer_BroadcastTo_3_layer_call_fn_9979
inputs_0	
inputs_1	
identity	т
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2		*
Tout
2	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_78132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0	*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:џџџџџџџџџ::Y U
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:D@

_output_shapes
:
"
_user_specified_name
inputs/1
Ё
N
2__inference_tf_op_layer_Shape_6_layer_call_fn_9794

inputs
identity	
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
CPU

GPU2*0J 8*V
fQRO
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_75722
PartitionedCall_
IdentityIdentityPartitionedCall:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_9629

inputs	
identity	g
Shape_3Shapeinputs*
T0	*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_3W
IdentityIdentityShape_3:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ
i
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_9767

inputs
identity	g
Shape_4Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_4W
IdentityIdentityShape_4:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н

Ў
F__inference_block3_conv1_layer_call_and_return_conditional_losses_7077

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
i
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_7600

inputs
identity	g
Shape_4Shapeinputs*
T0*
_cloned(*
_output_shapes
:*
out_type0	2	
Shape_4W
IdentityIdentityShape_4:output:0*
T0	*
_output_shapes
:2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Я
"__inference_signature_wrapper_8932
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1	

identity_2	

identity_3	

identity_4	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2				*
_output_shapes|
z:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__wrapped_model_69772
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0	*'
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*В
_input_shapes 
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ў
q
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_9688

inputs	
identity	x
strided_slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/begint
strided_slice_7/endConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/end|
strided_slice_7/stridesConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stridesя
strided_slice_7StridedSliceinputsstrided_slice_7/begin:output:0strided_slice_7/end:output:0 strided_slice_7/strides:output:0*
Index0*
T0	*
_cloned(*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7[
IdentityIdentitystrided_slice_7:output:0*
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

i
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_9757

inputs	
identity	`
Range_2/startConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Range_2/start`
Range_2/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2
Range_2/delta
Range_2RangeRange_2/start:output:0inputsRange_2/delta:output:0*

Tidx0	*
_cloned(*#
_output_shapes
:џџџџџџџџџ2	
Range_2`
IdentityIdentityRange_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
с
V
:__inference_tf_op_layer_strided_slice_9_layer_call_fn_9872

inputs	
identity	Ї
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
CPU

GPU2*0J 8*^
fYRW
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_77192
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
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
U
input_2J
serving_default_input_2:0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
block5_conv2K
StatefulPartitionedCall:0,џџџџџџџџџџџџџџџџџџџџџџџџџџџI
tf_op_layer_Transpose0
StatefulPartitionedCall:1	џџџџџџџџџK
tf_op_layer_Transpose_10
StatefulPartitionedCall:2	џџџџџџџџџK
tf_op_layer_Transpose_20
StatefulPartitionedCall:3	џџџџџџџџџK
tf_op_layer_Transpose_30
StatefulPartitionedCall:4	џџџџџџџџџtensorflow/serving/predict:Нэ
щє
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer_with_weights-12
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer_with_weights-13
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
signatures
+ &call_and_return_all_conditional_losses
Ё__call__
Ђ_default_save_signature"ш
_tf_keras_modelђч{"class_name": "Model", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_1", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_1", "op": "MaxPoolWithArgmax", "input": ["block2_conv2_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_1", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_2", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_2", "op": "MaxPoolWithArgmax", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_2", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_3", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_3", "op": "MaxPoolWithArgmax", "input": ["block4_conv4_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_3", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_3", "op": "Shape", "input": ["MaxPoolWithArgmax_1:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_3", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_5", "op": "Shape", "input": ["MaxPoolWithArgmax_2:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_5", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_7", "op": "Shape", "input": ["MaxPoolWithArgmax_3:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_7", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_4", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_3", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_4", "inbound_nodes": [[["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_7", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_7", "op": "StridedSlice", "input": ["Shape_5", "strided_slice_7/begin", "strided_slice_7/end", "strided_slice_7/strides"], "attr": {"end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_7", "inbound_nodes": [[["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_10", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_10", "op": "StridedSlice", "input": ["Shape_7", "strided_slice_10/begin", "strided_slice_10/end", "strided_slice_10/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_10", "inbound_nodes": [[["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_1", "op": "Range", "input": ["Range_1/start", "strided_slice_4", "Range_1/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_1", "inbound_nodes": [[["tf_op_layer_strided_slice_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["block2_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_2", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_2", "op": "Range", "input": ["Range_2/start", "strided_slice_7", "Range_2/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_2", "inbound_nodes": [[["tf_op_layer_strided_slice_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_4", "op": "Shape", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_4", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_3", "op": "Range", "input": ["Range_3/start", "strided_slice_10", "Range_3/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_3", "inbound_nodes": [[["tf_op_layer_strided_slice_10", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_6", "op": "Shape", "input": ["block4_conv4_1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Shape_6", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_2", "inbound_nodes": [[["tf_op_layer_Range", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_3", "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_5", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Range_1", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_5", "inbound_nodes": [[["tf_op_layer_Range_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_6", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_6", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_8", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_8", "op": "StridedSlice", "input": ["Range_2", "strided_slice_8/begin", "strided_slice_8/end", "strided_slice_8/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_8", "inbound_nodes": [[["tf_op_layer_Range_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_9", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_9", "op": "StridedSlice", "input": ["Shape_4", "strided_slice_9/begin", "strided_slice_9/end", "strided_slice_9/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_9", "inbound_nodes": [[["tf_op_layer_Shape_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_11", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_11", "op": "StridedSlice", "input": ["Range_3", "strided_slice_11/begin", "strided_slice_11/end", "strided_slice_11/strides"], "attr": {"T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "14"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_11", "inbound_nodes": [[["tf_op_layer_Range_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_12", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_12", "op": "StridedSlice", "input": ["Shape_6", "strided_slice_12/begin", "strided_slice_12/end", "strided_slice_12/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_12", "inbound_nodes": [[["tf_op_layer_Shape_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"Tidx": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo", "inbound_nodes": [[["tf_op_layer_strided_slice_2", 0, 0, {}], ["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod", "inbound_nodes": [[["tf_op_layer_strided_slice_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_1", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_1", "op": "BroadcastTo", "input": ["strided_slice_5", "Shape_3"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_1", "inbound_nodes": [[["tf_op_layer_strided_slice_5", 0, 0, {}], ["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["strided_slice_6", "Prod_2/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_2", "inbound_nodes": [[["tf_op_layer_strided_slice_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_2", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_2", "op": "BroadcastTo", "input": ["strided_slice_8", "Shape_5"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_2", "inbound_nodes": [[["tf_op_layer_strided_slice_8", 0, 0, {}], ["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_4", "op": "Prod", "input": ["strided_slice_9", "Prod_4/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_4", "inbound_nodes": [[["tf_op_layer_strided_slice_9", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_3", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_3", "op": "BroadcastTo", "input": ["strided_slice_11", "Shape_7"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_3", "inbound_nodes": [[["tf_op_layer_strided_slice_11", 0, 0, {}], ["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_6", "op": "Prod", "input": ["strided_slice_12", "Prod_6/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_6", "inbound_nodes": [[["tf_op_layer_strided_slice_12", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_BroadcastTo", 0, 0, {}], ["tf_op_layer_Prod", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["BroadcastTo_1", "Prod_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_BroadcastTo_1", 0, 0, {}], ["tf_op_layer_Prod_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["BroadcastTo_2", "Prod_4"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_2", "inbound_nodes": [[["tf_op_layer_BroadcastTo_2", 0, 0, {}], ["tf_op_layer_Prod_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["BroadcastTo_3", "Prod_6"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_3", "inbound_nodes": [[["tf_op_layer_BroadcastTo_3", 0, 0, {}], ["tf_op_layer_Prod_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}], ["tf_op_layer_Mul", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["MaxPoolWithArgmax_1:1", "Mul_1"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 1, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_3", "op": "Prod", "input": ["Shape_3", "Prod_3/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_3", "inbound_nodes": [[["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["MaxPoolWithArgmax_2:1", "Mul_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 1, {}], ["tf_op_layer_Mul_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_5", "op": "Prod", "input": ["Shape_5", "Prod_5/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_5", "inbound_nodes": [[["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["MaxPoolWithArgmax_3:1", "Mul_3"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 1, {}], ["tf_op_layer_Mul_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_7", "op": "Prod", "input": ["Shape_7", "Prod_7/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_7", "inbound_nodes": [[["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}], ["tf_op_layer_Prod_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["AddV2_1", "Prod_3"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}], ["tf_op_layer_Prod_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_2", "op": "Reshape", "input": ["AddV2_2", "Prod_5"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_2", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Prod_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_3", "op": "Reshape", "input": ["AddV2_3", "Prod_7"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_3", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}], ["tf_op_layer_Prod_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}], ["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_1", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_1", "op": "UnravelIndex", "input": ["Reshape_1", "Shape_2"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_1", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}], ["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_2", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_2", "op": "UnravelIndex", "input": ["Reshape_2", "Shape_4"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_2", "inbound_nodes": [[["tf_op_layer_Reshape_2", 0, 0, {}], ["tf_op_layer_Shape_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_3", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_3", "op": "UnravelIndex", "input": ["Reshape_3", "Shape_6"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_3", "inbound_nodes": [[["tf_op_layer_Reshape_3", 0, 0, {}], ["tf_op_layer_Shape_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose", "inbound_nodes": [[["tf_op_layer_UnravelIndex", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["UnravelIndex_1", "Transpose_1/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["tf_op_layer_UnravelIndex_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_2", "op": "Transpose", "input": ["UnravelIndex_2", "Transpose_2/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_2", "inbound_nodes": [[["tf_op_layer_UnravelIndex_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_3", "op": "Transpose", "input": ["UnravelIndex_3", "Transpose_3/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_3", "inbound_nodes": [[["tf_op_layer_UnravelIndex_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block5_conv2", 0, 0], ["tf_op_layer_Transpose", 0, 0], ["tf_op_layer_Transpose_1", 0, 0], ["tf_op_layer_Transpose_2", 0, 0], ["tf_op_layer_Transpose_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}, "name": "tf_op_layer_BiasAdd", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["tf_op_layer_BiasAdd", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_1", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_1", "op": "MaxPoolWithArgmax", "input": ["block2_conv2_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_1", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_2", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_2", "op": "MaxPoolWithArgmax", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_2", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPoolWithArgmax_3", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_3", "op": "MaxPoolWithArgmax", "input": ["block4_conv4_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPoolWithArgmax_3", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_3", "op": "Shape", "input": ["MaxPoolWithArgmax_1:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_3", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_5", "op": "Shape", "input": ["MaxPoolWithArgmax_2:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_5", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_7", "op": "Shape", "input": ["MaxPoolWithArgmax_3:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_7", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 1, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_4", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_3", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_4", "inbound_nodes": [[["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_7", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_7", "op": "StridedSlice", "input": ["Shape_5", "strided_slice_7/begin", "strided_slice_7/end", "strided_slice_7/strides"], "attr": {"end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_7", "inbound_nodes": [[["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_10", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_10", "op": "StridedSlice", "input": ["Shape_7", "strided_slice_10/begin", "strided_slice_10/end", "strided_slice_10/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "name": "tf_op_layer_strided_slice_10", "inbound_nodes": [[["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_1", "op": "Range", "input": ["Range_1/start", "strided_slice_4", "Range_1/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_1", "inbound_nodes": [[["tf_op_layer_strided_slice_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["block2_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_2", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_2", "op": "Range", "input": ["Range_2/start", "strided_slice_7", "Range_2/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_2", "inbound_nodes": [[["tf_op_layer_strided_slice_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_4", "op": "Shape", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Shape_4", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Range_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_3", "op": "Range", "input": ["Range_3/start", "strided_slice_10", "Range_3/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}, "name": "tf_op_layer_Range_3", "inbound_nodes": [[["tf_op_layer_strided_slice_10", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Shape_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_6", "op": "Shape", "input": ["block4_conv4_1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Shape_6", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_2", "inbound_nodes": [[["tf_op_layer_Range", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_3", "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_5", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Range_1", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_5", "inbound_nodes": [[["tf_op_layer_Range_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_6", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_6", "inbound_nodes": [[["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_8", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_8", "op": "StridedSlice", "input": ["Range_2", "strided_slice_8/begin", "strided_slice_8/end", "strided_slice_8/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_8", "inbound_nodes": [[["tf_op_layer_Range_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_9", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_9", "op": "StridedSlice", "input": ["Shape_4", "strided_slice_9/begin", "strided_slice_9/end", "strided_slice_9/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_9", "inbound_nodes": [[["tf_op_layer_Shape_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_11", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_11", "op": "StridedSlice", "input": ["Range_3", "strided_slice_11/begin", "strided_slice_11/end", "strided_slice_11/strides"], "attr": {"T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "14"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_11", "inbound_nodes": [[["tf_op_layer_Range_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_12", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_12", "op": "StridedSlice", "input": ["Shape_6", "strided_slice_12/begin", "strided_slice_12/end", "strided_slice_12/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "name": "tf_op_layer_strided_slice_12", "inbound_nodes": [[["tf_op_layer_Shape_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"Tidx": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo", "inbound_nodes": [[["tf_op_layer_strided_slice_2", 0, 0, {}], ["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod", "inbound_nodes": [[["tf_op_layer_strided_slice_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_1", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_1", "op": "BroadcastTo", "input": ["strided_slice_5", "Shape_3"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_1", "inbound_nodes": [[["tf_op_layer_strided_slice_5", 0, 0, {}], ["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["strided_slice_6", "Prod_2/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_2", "inbound_nodes": [[["tf_op_layer_strided_slice_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_2", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_2", "op": "BroadcastTo", "input": ["strided_slice_8", "Shape_5"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_2", "inbound_nodes": [[["tf_op_layer_strided_slice_8", 0, 0, {}], ["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_4", "op": "Prod", "input": ["strided_slice_9", "Prod_4/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_4", "inbound_nodes": [[["tf_op_layer_strided_slice_9", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "BroadcastTo_3", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_3", "op": "BroadcastTo", "input": ["strided_slice_11", "Shape_7"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_BroadcastTo_3", "inbound_nodes": [[["tf_op_layer_strided_slice_11", 0, 0, {}], ["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_6", "op": "Prod", "input": ["strided_slice_12", "Prod_6/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_6", "inbound_nodes": [[["tf_op_layer_strided_slice_12", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_BroadcastTo", 0, 0, {}], ["tf_op_layer_Prod", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["BroadcastTo_1", "Prod_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_BroadcastTo_1", 0, 0, {}], ["tf_op_layer_Prod_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["BroadcastTo_2", "Prod_4"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_2", "inbound_nodes": [[["tf_op_layer_BroadcastTo_2", 0, 0, {}], ["tf_op_layer_Prod_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["BroadcastTo_3", "Prod_6"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Mul_3", "inbound_nodes": [[["tf_op_layer_BroadcastTo_3", 0, 0, {}], ["tf_op_layer_Prod_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax", 0, 1, {}], ["tf_op_layer_Mul", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_1", "inbound_nodes": [[["tf_op_layer_Shape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["MaxPoolWithArgmax_1:1", "Mul_1"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_1", 0, 1, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_3", "op": "Prod", "input": ["Shape_3", "Prod_3/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_3", "inbound_nodes": [[["tf_op_layer_Shape_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["MaxPoolWithArgmax_2:1", "Mul_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_2", 0, 1, {}], ["tf_op_layer_Mul_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_5", "op": "Prod", "input": ["Shape_5", "Prod_5/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_5", "inbound_nodes": [[["tf_op_layer_Shape_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["MaxPoolWithArgmax_3:1", "Mul_3"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 1, {}], ["tf_op_layer_Mul_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Prod_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_7", "op": "Prod", "input": ["Shape_7", "Prod_7/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Prod_7", "inbound_nodes": [[["tf_op_layer_Shape_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}], ["tf_op_layer_Prod_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["AddV2_1", "Prod_3"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}], ["tf_op_layer_Prod_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_2", "op": "Reshape", "input": ["AddV2_2", "Prod_5"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_2", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Prod_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_3", "op": "Reshape", "input": ["AddV2_3", "Prod_7"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_Reshape_3", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}], ["tf_op_layer_Prod_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["tf_op_layer_MaxPoolWithArgmax_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}], ["tf_op_layer_Shape", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_1", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_1", "op": "UnravelIndex", "input": ["Reshape_1", "Shape_2"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_1", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}], ["tf_op_layer_Shape_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_2", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_2", "op": "UnravelIndex", "input": ["Reshape_2", "Shape_4"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_2", "inbound_nodes": [[["tf_op_layer_Reshape_2", 0, 0, {}], ["tf_op_layer_Shape_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "UnravelIndex_3", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_3", "op": "UnravelIndex", "input": ["Reshape_3", "Shape_6"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}, "name": "tf_op_layer_UnravelIndex_3", "inbound_nodes": [[["tf_op_layer_Reshape_3", 0, 0, {}], ["tf_op_layer_Shape_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose", "inbound_nodes": [[["tf_op_layer_UnravelIndex", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["UnravelIndex_1", "Transpose_1/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["tf_op_layer_UnravelIndex_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_2", "op": "Transpose", "input": ["UnravelIndex_2", "Transpose_2/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_2", "inbound_nodes": [[["tf_op_layer_UnravelIndex_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_3", "op": "Transpose", "input": ["UnravelIndex_3", "Transpose_3/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_3", "inbound_nodes": [[["tf_op_layer_UnravelIndex_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block5_conv2", 0, 0], ["tf_op_layer_Transpose", 0, 0], ["tf_op_layer_Transpose_1", 0, 0], ["tf_op_layer_Transpose_2", 0, 0], ["tf_op_layer_Transpose_3", 0, 0]]}}}
"ў
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
д
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+Ѓ&call_and_return_all_conditional_losses
Є__call__"У
_tf_keras_layerЉ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_2", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"begin_mask": {"i": "2"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0], "2": [0, 0], "3": [1, -1]}}}
Ѓ
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+Ѕ&call_and_return_all_conditional_losses
І__call__"
_tf_keras_layerј{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BiasAdd", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BiasAdd", "trainable": false, "dtype": "float32", "node_def": {"name": "BiasAdd", "op": "BiasAdd", "input": ["strided_slice", "BiasAdd/bias"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}}}, "constants": {"1": [-103.93900299072266, -116.77899932861328, -123.68000030517578]}}}
а	

[kernel
\bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+Ї&call_and_return_all_conditional_losses
Ј__call__"Љ
_tf_keras_layer{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
в	

akernel
bbias
c	variables
dregularization_losses
etrainable_variables
f	keras_api
+Љ&call_and_return_all_conditional_losses
Њ__call__"Ћ
_tf_keras_layer{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
Ј
g	variables
hregularization_losses
itrainable_variables
j	keras_api
+Ћ&call_and_return_all_conditional_losses
Ќ__call__"
_tf_keras_layer§{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPoolWithArgmax", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPoolWithArgmax", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax", "op": "MaxPoolWithArgmax", "input": ["block1_conv2_1/Identity"], "attr": {"Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
г	

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"Ќ
_tf_keras_layer{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
е	

qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
Ў
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPoolWithArgmax_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPoolWithArgmax_1", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_1", "op": "MaxPoolWithArgmax", "input": ["block2_conv2_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
ж	

{kernel
|bias
}	variables
~regularization_losses
trainable_variables
	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
л	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
л	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
л	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
В
	variables
regularization_losses
trainable_variables
	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPoolWithArgmax_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPoolWithArgmax_2", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_2", "op": "MaxPoolWithArgmax", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "include_batch_in_index": {"b": false}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}}
л	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
л	
kernel
	bias
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 512]}}
л	
Ѓkernel
	Єbias
Ѕ	variables
Іregularization_losses
Їtrainable_variables
Ј	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 512]}}
л	
Љkernel
	Њbias
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 512]}}
В
Џ	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPoolWithArgmax_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPoolWithArgmax_3", "trainable": false, "dtype": "float32", "node_def": {"name": "MaxPoolWithArgmax_3", "op": "MaxPoolWithArgmax", "input": ["block4_conv4_1/Identity"], "attr": {"ksize": {"list": {"i": ["1", "2", "2", "1"]}}, "Targmax": {"type": "DT_INT64"}, "strides": {"list": {"i": ["1", "2", "2", "1"]}}, "padding": {"s": "U0FNRQ=="}, "include_batch_in_index": {"b": false}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
з
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"Т
_tf_keras_layerЈ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_1", "op": "Shape", "input": ["MaxPoolWithArgmax:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}}
й
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"Ф
_tf_keras_layerЊ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_3", "op": "Shape", "input": ["MaxPoolWithArgmax_1:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
й
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Ф
_tf_keras_layerЊ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_5", "op": "Shape", "input": ["MaxPoolWithArgmax_2:1"], "attr": {"T": {"type": "DT_INT64"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
й
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"Ф
_tf_keras_layerЊ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_7", "op": "Shape", "input": ["MaxPoolWithArgmax_3:1"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}}
к
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_1", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["Shape_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
к
Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_4", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_4", "op": "StridedSlice", "input": ["Shape_3", "strided_slice_4/begin", "strided_slice_4/end", "strided_slice_4/strides"], "attr": {"shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
к
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
+г&call_and_return_all_conditional_losses
д__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_7", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_7", "op": "StridedSlice", "input": ["Shape_5", "strided_slice_7/begin", "strided_slice_7/end", "strided_slice_7/strides"], "attr": {"end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
р
Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"Ы
_tf_keras_layerБ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_10", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_10", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_10", "op": "StridedSlice", "input": ["Shape_7", "strided_slice_10/begin", "strided_slice_10/end", "strided_slice_10/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "1"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}}
к
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
+з&call_and_return_all_conditional_losses
и__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Range", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Range", "trainable": false, "dtype": "float32", "node_def": {"name": "Range", "op": "Range", "input": ["Range/start", "strided_slice_1", "Range/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}}
е
з	variables
иregularization_losses
йtrainable_variables
к	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Р
_tf_keras_layerІ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape", "op": "Shape", "input": ["block1_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
ф
л	variables
мregularization_losses
нtrainable_variables
о	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Я
_tf_keras_layerЕ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Range_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Range_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_1", "op": "Range", "input": ["Range_1/start", "strided_slice_4", "Range_1/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}}
л
п	variables
рregularization_losses
сtrainable_variables
т	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_2", "op": "Shape", "input": ["block2_conv2_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
ф
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Я
_tf_keras_layerЕ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Range_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Range_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_2", "op": "Range", "input": ["Range_2/start", "strided_slice_7", "Range_2/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}}
л
ч	variables
шregularization_losses
щtrainable_variables
ъ	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_4", "op": "Shape", "input": ["block3_conv4_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT64"}}}, "constants": {}}}
х
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"а
_tf_keras_layerЖ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Range_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Range_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Range_3", "op": "Range", "input": ["Range_3/start", "strided_slice_10", "Range_3/delta"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {"0": 0, "2": 1}}}
л
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Shape_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Shape_6", "op": "Shape", "input": ["block4_conv4_1/Identity"], "attr": {"out_type": {"type": "DT_INT64"}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
є
ѓ	variables
єregularization_losses
ѕtrainable_variables
і	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"п
_tf_keras_layerХ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_2", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_2", "op": "StridedSlice", "input": ["Range", "strided_slice_2/begin", "strided_slice_2/end", "strided_slice_2/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}}
и
ї	variables
јregularization_losses
љtrainable_variables
њ	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"У
_tf_keras_layerЉ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_3", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_3", "op": "StridedSlice", "input": ["Shape", "strided_slice_3/begin", "strided_slice_3/end", "strided_slice_3/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}}
і
ћ	variables
ќregularization_losses
§trainable_variables
ў	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"с
_tf_keras_layerЧ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_5", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_5", "op": "StridedSlice", "input": ["Range_1", "strided_slice_5/begin", "strided_slice_5/end", "strided_slice_5/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}}
к
џ	variables
regularization_losses
trainable_variables
	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_6", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_6", "op": "StridedSlice", "input": ["Shape_2", "strided_slice_6/begin", "strided_slice_6/end", "strided_slice_6/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}}
і
	variables
regularization_losses
trainable_variables
	keras_api
+я&call_and_return_all_conditional_losses
№__call__"с
_tf_keras_layerЧ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_8", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_8", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_8", "op": "StridedSlice", "input": ["Range_2", "strided_slice_8/begin", "strided_slice_8/end", "strided_slice_8/strides"], "attr": {"T": {"type": "DT_INT64"}, "new_axis_mask": {"i": "14"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}}
к
	variables
regularization_losses
trainable_variables
	keras_api
+ё&call_and_return_all_conditional_losses
ђ__call__"Х
_tf_keras_layerЋ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_9", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_9", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_9", "op": "StridedSlice", "input": ["Shape_4", "strided_slice_9/begin", "strided_slice_9/end", "strided_slice_9/strides"], "attr": {"begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_INT64"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}}
ќ
	variables
regularization_losses
trainable_variables
	keras_api
+ѓ&call_and_return_all_conditional_losses
є__call__"ч
_tf_keras_layerЭ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_11", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_11", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_11", "op": "StridedSlice", "input": ["Range_3", "strided_slice_11/begin", "strided_slice_11/end", "strided_slice_11/strides"], "attr": {"T": {"type": "DT_INT64"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "14"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3": [1, 1, 1, 1]}}}
р
	variables
regularization_losses
trainable_variables
	keras_api
+ѕ&call_and_return_all_conditional_losses
і__call__"Ы
_tf_keras_layerБ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_12", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_12", "trainable": false, "dtype": "float32", "node_def": {"name": "strided_slice_12", "op": "StridedSlice", "input": ["Shape_6", "strided_slice_12/begin", "strided_slice_12/end", "strided_slice_12/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}}
ь
	variables
regularization_losses
trainable_variables
	keras_api
+ї&call_and_return_all_conditional_losses
ј__call__"з
_tf_keras_layerН{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BroadcastTo", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BroadcastTo", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo", "op": "BroadcastTo", "input": ["strided_slice_2", "Shape_1"], "attr": {"Tidx": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}}

	variables
regularization_losses
trainable_variables
	keras_api
+љ&call_and_return_all_conditional_losses
њ__call__"э
_tf_keras_layerг{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod", "op": "Prod", "input": ["strided_slice_3", "Prod/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}}
ђ
	variables
regularization_losses
trainable_variables
	keras_api
+ћ&call_and_return_all_conditional_losses
ќ__call__"н
_tf_keras_layerУ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BroadcastTo_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BroadcastTo_1", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_1", "op": "BroadcastTo", "input": ["strided_slice_5", "Shape_3"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}}

	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
+§&call_and_return_all_conditional_losses
ў__call__"ѕ
_tf_keras_layerл{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_2", "op": "Prod", "input": ["strided_slice_6", "Prod_2/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [0]}}}
ђ
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
+џ&call_and_return_all_conditional_losses
__call__"н
_tf_keras_layerУ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BroadcastTo_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BroadcastTo_2", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_2", "op": "BroadcastTo", "input": ["strided_slice_8", "Shape_5"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}}

Ї	variables
Јregularization_losses
Љtrainable_variables
Њ	keras_api
+&call_and_return_all_conditional_losses
__call__"ѕ
_tf_keras_layerл{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_4", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_4", "op": "Prod", "input": ["strided_slice_9", "Prod_4/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [0]}}}
ѓ
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
+&call_and_return_all_conditional_losses
__call__"о
_tf_keras_layerФ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_BroadcastTo_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "BroadcastTo_3", "trainable": false, "dtype": "float32", "node_def": {"name": "BroadcastTo_3", "op": "BroadcastTo", "input": ["strided_slice_11", "Shape_7"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT64"}}}, "constants": {}}}

Џ	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
+&call_and_return_all_conditional_losses
__call__"і
_tf_keras_layerм{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_6", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_6", "op": "Prod", "input": ["strided_slice_12", "Prod_6/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}}
Ї
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerј{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["BroadcastTo", "Prod"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
Б
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["BroadcastTo_1", "Prod_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
Б
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_2", "op": "Mul", "input": ["BroadcastTo_2", "Prod_4"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
Б
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["BroadcastTo_3", "Prod_6"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}
Ж
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
+&call_and_return_all_conditional_losses
__call__"Ё
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["MaxPoolWithArgmax:1", "Mul"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}

Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
+&call_and_return_all_conditional_losses
__call__"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_1", "op": "Prod", "input": ["Shape_1", "Prod_1/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [0]}}}
Р
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
+&call_and_return_all_conditional_losses
__call__"Ћ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_1", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["MaxPoolWithArgmax_1:1", "Mul_1"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}

Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
+&call_and_return_all_conditional_losses
__call__"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_3", "op": "Prod", "input": ["Shape_3", "Prod_3/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}}
Р
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
+&call_and_return_all_conditional_losses
__call__"Ћ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_2", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["MaxPoolWithArgmax_2:1", "Mul_2"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}

з	variables
иregularization_losses
йtrainable_variables
к	keras_api
+&call_and_return_all_conditional_losses
__call__"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_5", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_5", "op": "Prod", "input": ["Shape_5", "Prod_5/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}}
Р
л	variables
мregularization_losses
нtrainable_variables
о	keras_api
+&call_and_return_all_conditional_losses
__call__"Ћ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_3", "trainable": false, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["MaxPoolWithArgmax_3:1", "Mul_3"], "attr": {"T": {"type": "DT_INT64"}}}, "constants": {}}}

п	variables
рregularization_losses
сtrainable_variables
т	keras_api
+&call_and_return_all_conditional_losses
__call__"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Prod_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Prod_7", "trainable": false, "dtype": "float32", "node_def": {"name": "Prod_7", "op": "Prod", "input": ["Shape_7", "Prod_7/reduction_indices"], "attr": {"T": {"type": "DT_INT64"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}}}, "constants": {"1": [0]}}}
г
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
+&call_and_return_all_conditional_losses
 __call__"О
_tf_keras_layerЄ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["AddV2", "Prod_1"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}}
л
ч	variables
шregularization_losses
щtrainable_variables
ъ	keras_api
+Ё&call_and_return_all_conditional_losses
Ђ__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["AddV2_1", "Prod_3"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}}
л
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
+Ѓ&call_and_return_all_conditional_losses
Є__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_2", "op": "Reshape", "input": ["AddV2_2", "Prod_5"], "attr": {"Tshape": {"type": "DT_INT64"}, "T": {"type": "DT_INT64"}}}, "constants": {}}}
л
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
+Ѕ&call_and_return_all_conditional_losses
І__call__"Ц
_tf_keras_layerЌ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Reshape_3", "op": "Reshape", "input": ["AddV2_3", "Prod_7"], "attr": {"T": {"type": "DT_INT64"}, "Tshape": {"type": "DT_INT64"}}}, "constants": {}}}
л	
ѓkernel
	єbias
ѕ	variables
іregularization_losses
їtrainable_variables
ј	keras_api
+Ї&call_and_return_all_conditional_losses
Ј__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 512]}}
Ы
љ	variables
њregularization_losses
ћtrainable_variables
ќ	keras_api
+Љ&call_and_return_all_conditional_losses
Њ__call__"Ж
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_UnravelIndex", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UnravelIndex", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex", "op": "UnravelIndex", "input": ["Reshape", "Shape"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
е
§	variables
ўregularization_losses
џtrainable_variables
	keras_api
+Ћ&call_and_return_all_conditional_losses
Ќ__call__"Р
_tf_keras_layerІ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_UnravelIndex_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UnravelIndex_1", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_1", "op": "UnravelIndex", "input": ["Reshape_1", "Shape_2"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
е
	variables
regularization_losses
trainable_variables
	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"Р
_tf_keras_layerІ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_UnravelIndex_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UnravelIndex_2", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_2", "op": "UnravelIndex", "input": ["Reshape_2", "Shape_4"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
е
	variables
regularization_losses
trainable_variables
	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"Р
_tf_keras_layerІ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_UnravelIndex_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UnravelIndex_3", "trainable": false, "dtype": "float32", "node_def": {"name": "UnravelIndex_3", "op": "UnravelIndex", "input": ["Reshape_3", "Shape_6"], "attr": {"Tidx": {"type": "DT_INT64"}}}, "constants": {}}}
л	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Ў
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 512]}}
є
	variables
regularization_losses
trainable_variables
	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"п
_tf_keras_layerХ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Transpose", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose", "op": "Transpose", "input": ["UnravelIndex", "Transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}}
ў
	variables
regularization_losses
trainable_variables
	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"щ
_tf_keras_layerЯ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Transpose_1", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["UnravelIndex_1", "Transpose_1/perm"], "attr": {"T": {"type": "DT_INT64"}, "Tperm": {"type": "DT_INT32"}}}, "constants": {"1": [1, 0]}}}
ў
	variables
regularization_losses
trainable_variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"щ
_tf_keras_layerЯ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Transpose_2", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_2", "op": "Transpose", "input": ["UnravelIndex_2", "Transpose_2/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}}
ў
	variables
regularization_losses
trainable_variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"щ
_tf_keras_layerЯ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Transpose_3", "trainable": false, "dtype": "float32", "node_def": {"name": "Transpose_3", "op": "Transpose", "input": ["UnravelIndex_3", "Transpose_3/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_INT64"}}}, "constants": {"1": [1, 0]}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper

[0
\1
a2
b3
k4
l5
q6
r7
{8
|9
10
11
12
13
14
15
16
17
18
19
Ѓ20
Є21
Љ22
Њ23
ѓ24
є25
26
27"
trackable_list_wrapper
г
layer_metrics
 layers
 Ёlayer_regularization_losses
Nregularization_losses
Ђnon_trainable_variables
Otrainable_variables
P	variables
Ѓmetrics
Ё__call__
Ђ_default_save_signature
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
-
Лserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Єlayer_metrics
Ѕlayers
S	variables
 Іlayer_regularization_losses
Tregularization_losses
Utrainable_variables
Їnon_trainable_variables
Јmetrics
Є__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Љlayer_metrics
Њlayers
W	variables
 Ћlayer_regularization_losses
Xregularization_losses
Ytrainable_variables
Ќnon_trainable_variables
­metrics
І__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ўlayer_metrics
Џlayers
]	variables
 Аlayer_regularization_losses
^regularization_losses
_trainable_variables
Бnon_trainable_variables
Вmetrics
Ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Гlayer_metrics
Дlayers
c	variables
 Еlayer_regularization_losses
dregularization_losses
etrainable_variables
Жnon_trainable_variables
Зmetrics
Њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Иlayer_metrics
Йlayers
g	variables
 Кlayer_regularization_losses
hregularization_losses
itrainable_variables
Лnon_trainable_variables
Мmetrics
Ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Нlayer_metrics
Оlayers
m	variables
 Пlayer_regularization_losses
nregularization_losses
otrainable_variables
Рnon_trainable_variables
Сmetrics
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
/:-2block2_conv2/kernel
 :2block2_conv2/bias
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Тlayer_metrics
Уlayers
s	variables
 Фlayer_regularization_losses
tregularization_losses
utrainable_variables
Хnon_trainable_variables
Цmetrics
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Чlayer_metrics
Шlayers
w	variables
 Щlayer_regularization_losses
xregularization_losses
ytrainable_variables
Ъnon_trainable_variables
Ыmetrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv1/kernel
 :2block3_conv1/bias
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ьlayer_metrics
Эlayers
}	variables
 Юlayer_regularization_losses
~regularization_losses
trainable_variables
Яnon_trainable_variables
аmetrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv2/kernel
 :2block3_conv2/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
бlayer_metrics
вlayers
	variables
 гlayer_regularization_losses
regularization_losses
trainable_variables
дnon_trainable_variables
еmetrics
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv3/kernel
 :2block3_conv3/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
жlayer_metrics
зlayers
	variables
 иlayer_regularization_losses
regularization_losses
trainable_variables
йnon_trainable_variables
кmetrics
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv4/kernel
 :2block3_conv4/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
лlayer_metrics
мlayers
	variables
 нlayer_regularization_losses
regularization_losses
trainable_variables
оnon_trainable_variables
пmetrics
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
рlayer_metrics
сlayers
	variables
 тlayer_regularization_losses
regularization_losses
trainable_variables
уnon_trainable_variables
фmetrics
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv1/kernel
 :2block4_conv1/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
хlayer_metrics
цlayers
	variables
 чlayer_regularization_losses
regularization_losses
trainable_variables
шnon_trainable_variables
щmetrics
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv2/kernel
 :2block4_conv2/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъlayer_metrics
ыlayers
	variables
 ьlayer_regularization_losses
 regularization_losses
Ёtrainable_variables
эnon_trainable_variables
юmetrics
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv3/kernel
 :2block4_conv3/bias
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
яlayer_metrics
№layers
Ѕ	variables
 ёlayer_regularization_losses
Іregularization_losses
Їtrainable_variables
ђnon_trainable_variables
ѓmetrics
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv4/kernel
 :2block4_conv4/bias
0
Љ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
єlayer_metrics
ѕlayers
Ћ	variables
 іlayer_regularization_losses
Ќregularization_losses
­trainable_variables
їnon_trainable_variables
јmetrics
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љlayer_metrics
њlayers
Џ	variables
 ћlayer_regularization_losses
Аregularization_losses
Бtrainable_variables
ќnon_trainable_variables
§metrics
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ўlayer_metrics
џlayers
Г	variables
 layer_regularization_losses
Дregularization_losses
Еtrainable_variables
non_trainable_variables
metrics
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
З	variables
 layer_regularization_losses
Иregularization_losses
Йtrainable_variables
non_trainable_variables
metrics
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Л	variables
 layer_regularization_losses
Мregularization_losses
Нtrainable_variables
non_trainable_variables
metrics
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
П	variables
 layer_regularization_losses
Рregularization_losses
Сtrainable_variables
non_trainable_variables
metrics
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
У	variables
 layer_regularization_losses
Фregularization_losses
Хtrainable_variables
non_trainable_variables
metrics
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Ч	variables
 layer_regularization_losses
Шregularization_losses
Щtrainable_variables
non_trainable_variables
metrics
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Ы	variables
 layer_regularization_losses
Ьregularization_losses
Эtrainable_variables
non_trainable_variables
 metrics
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёlayer_metrics
Ђlayers
Я	variables
 Ѓlayer_regularization_losses
аregularization_losses
бtrainable_variables
Єnon_trainable_variables
Ѕmetrics
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Іlayer_metrics
Їlayers
г	variables
 Јlayer_regularization_losses
дregularization_losses
еtrainable_variables
Љnon_trainable_variables
Њmetrics
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћlayer_metrics
Ќlayers
з	variables
 ­layer_regularization_losses
иregularization_losses
йtrainable_variables
Ўnon_trainable_variables
Џmetrics
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аlayer_metrics
Бlayers
л	variables
 Вlayer_regularization_losses
мregularization_losses
нtrainable_variables
Гnon_trainable_variables
Дmetrics
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еlayer_metrics
Жlayers
п	variables
 Зlayer_regularization_losses
рregularization_losses
сtrainable_variables
Иnon_trainable_variables
Йmetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Кlayer_metrics
Лlayers
у	variables
 Мlayer_regularization_losses
фregularization_losses
хtrainable_variables
Нnon_trainable_variables
Оmetrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Пlayer_metrics
Рlayers
ч	variables
 Сlayer_regularization_losses
шregularization_losses
щtrainable_variables
Тnon_trainable_variables
Уmetrics
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фlayer_metrics
Хlayers
ы	variables
 Цlayer_regularization_losses
ьregularization_losses
эtrainable_variables
Чnon_trainable_variables
Шmetrics
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Щlayer_metrics
Ъlayers
я	variables
 Ыlayer_regularization_losses
№regularization_losses
ёtrainable_variables
Ьnon_trainable_variables
Эmetrics
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Юlayer_metrics
Яlayers
ѓ	variables
 аlayer_regularization_losses
єregularization_losses
ѕtrainable_variables
бnon_trainable_variables
вmetrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
гlayer_metrics
дlayers
ї	variables
 еlayer_regularization_losses
јregularization_losses
љtrainable_variables
жnon_trainable_variables
зmetrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иlayer_metrics
йlayers
ћ	variables
 кlayer_regularization_losses
ќregularization_losses
§trainable_variables
лnon_trainable_variables
мmetrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
нlayer_metrics
оlayers
џ	variables
 пlayer_regularization_losses
regularization_losses
trainable_variables
рnon_trainable_variables
сmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
тlayer_metrics
уlayers
	variables
 фlayer_regularization_losses
regularization_losses
trainable_variables
хnon_trainable_variables
цmetrics
№__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
чlayer_metrics
шlayers
	variables
 щlayer_regularization_losses
regularization_losses
trainable_variables
ъnon_trainable_variables
ыmetrics
ђ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ьlayer_metrics
эlayers
	variables
 юlayer_regularization_losses
regularization_losses
trainable_variables
яnon_trainable_variables
№metrics
є__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ёlayer_metrics
ђlayers
	variables
 ѓlayer_regularization_losses
regularization_losses
trainable_variables
єnon_trainable_variables
ѕmetrics
і__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
іlayer_metrics
їlayers
	variables
 јlayer_regularization_losses
regularization_losses
trainable_variables
љnon_trainable_variables
њmetrics
ј__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћlayer_metrics
ќlayers
	variables
 §layer_regularization_losses
regularization_losses
trainable_variables
ўnon_trainable_variables
џmetrics
њ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
 regularization_losses
Ёtrainable_variables
non_trainable_variables
metrics
ў__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Ѓ	variables
 layer_regularization_losses
Єregularization_losses
Ѕtrainable_variables
non_trainable_variables
metrics
__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Ї	variables
 layer_regularization_losses
Јregularization_losses
Љtrainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Ћ	variables
 layer_regularization_losses
Ќregularization_losses
­trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Џ	variables
 layer_regularization_losses
Аregularization_losses
Бtrainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
Г	variables
  layer_regularization_losses
Дregularization_losses
Еtrainable_variables
Ёnon_trainable_variables
Ђmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѓlayer_metrics
Єlayers
З	variables
 Ѕlayer_regularization_losses
Иregularization_losses
Йtrainable_variables
Іnon_trainable_variables
Їmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Јlayer_metrics
Љlayers
Л	variables
 Њlayer_regularization_losses
Мregularization_losses
Нtrainable_variables
Ћnon_trainable_variables
Ќmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
­layer_metrics
Ўlayers
П	variables
 Џlayer_regularization_losses
Рregularization_losses
Сtrainable_variables
Аnon_trainable_variables
Бmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вlayer_metrics
Гlayers
У	variables
 Дlayer_regularization_losses
Фregularization_losses
Хtrainable_variables
Еnon_trainable_variables
Жmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зlayer_metrics
Иlayers
Ч	variables
 Йlayer_regularization_losses
Шregularization_losses
Щtrainable_variables
Кnon_trainable_variables
Лmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мlayer_metrics
Нlayers
Ы	variables
 Оlayer_regularization_losses
Ьregularization_losses
Эtrainable_variables
Пnon_trainable_variables
Рmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Сlayer_metrics
Тlayers
Я	variables
 Уlayer_regularization_losses
аregularization_losses
бtrainable_variables
Фnon_trainable_variables
Хmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Цlayer_metrics
Чlayers
г	variables
 Шlayer_regularization_losses
дregularization_losses
еtrainable_variables
Щnon_trainable_variables
Ъmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыlayer_metrics
Ьlayers
з	variables
 Эlayer_regularization_losses
иregularization_losses
йtrainable_variables
Юnon_trainable_variables
Яmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
аlayer_metrics
бlayers
л	variables
 вlayer_regularization_losses
мregularization_losses
нtrainable_variables
гnon_trainable_variables
дmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
еlayer_metrics
жlayers
п	variables
 зlayer_regularization_losses
рregularization_losses
сtrainable_variables
иnon_trainable_variables
йmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
кlayer_metrics
лlayers
у	variables
 мlayer_regularization_losses
фregularization_losses
хtrainable_variables
нnon_trainable_variables
оmetrics
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
пlayer_metrics
рlayers
ч	variables
 сlayer_regularization_losses
шregularization_losses
щtrainable_variables
тnon_trainable_variables
уmetrics
Ђ__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
фlayer_metrics
хlayers
ы	variables
 цlayer_regularization_losses
ьregularization_losses
эtrainable_variables
чnon_trainable_variables
шmetrics
Є__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
щlayer_metrics
ъlayers
я	variables
 ыlayer_regularization_losses
№regularization_losses
ёtrainable_variables
ьnon_trainable_variables
эmetrics
І__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv1/kernel
 :2block5_conv1/bias
0
ѓ0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
юlayer_metrics
яlayers
ѕ	variables
 №layer_regularization_losses
іregularization_losses
їtrainable_variables
ёnon_trainable_variables
ђmetrics
Ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓlayer_metrics
єlayers
љ	variables
 ѕlayer_regularization_losses
њregularization_losses
ћtrainable_variables
іnon_trainable_variables
їmetrics
Њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
јlayer_metrics
љlayers
§	variables
 њlayer_regularization_losses
ўregularization_losses
џtrainable_variables
ћnon_trainable_variables
ќmetrics
Ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
§layer_metrics
ўlayers
	variables
 џlayer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv2/kernel
 :2block5_conv2/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
layers
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
non_trainable_variables
metrics
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
ў
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
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76"
trackable_list_wrapper
 "
trackable_list_wrapper

[0
\1
a2
b3
k4
l5
q6
r7
{8
|9
10
11
12
13
14
15
16
17
18
19
Ѓ20
Є21
Љ22
Њ23
ѓ24
є25
26
27"
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
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
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
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
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
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Љ0
Њ1"
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
0
ѓ0
є1"
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
0
0
1"
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
Ъ2Ч
?__inference_model_layer_call_and_return_conditional_losses_9396
?__inference_model_layer_call_and_return_conditional_losses_8290
?__inference_model_layer_call_and_return_conditional_losses_9164
?__inference_model_layer_call_and_return_conditional_losses_8434Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
$__inference_model_layer_call_fn_9465
$__inference_model_layer_call_fn_9534
$__inference_model_layer_call_fn_8861
$__inference_model_layer_call_fn_8648Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
__inference__wrapped_model_6977а
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
§2њ
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_9542Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
8__inference_tf_op_layer_strided_slice_layer_call_fn_9547Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_9553Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_BiasAdd_layer_call_fn_9558Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѕ2Ђ
F__inference_block1_conv1_layer_call_and_return_conditional_losses_6989з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block1_conv1_layer_call_fn_6999з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѕ2Ђ
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7011з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
+__inference_block1_conv2_layer_call_fn_7021з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2ў
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_9565Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ц2у
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_9572Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѕ2Ђ
F__inference_block2_conv1_layer_call_and_return_conditional_losses_7033з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
+__inference_block2_conv1_layer_call_fn_7043з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
І2Ѓ
F__inference_block2_conv2_layer_call_and_return_conditional_losses_7055и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block2_conv2_layer_call_fn_7065и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_9579Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ш2х
>__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_fn_9586Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
І2Ѓ
F__inference_block3_conv1_layer_call_and_return_conditional_losses_7077и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block3_conv1_layer_call_fn_7087и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block3_conv2_layer_call_and_return_conditional_losses_7099и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block3_conv2_layer_call_fn_7109и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block3_conv3_layer_call_and_return_conditional_losses_7121и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block3_conv3_layer_call_fn_7131и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block3_conv4_layer_call_and_return_conditional_losses_7143и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block3_conv4_layer_call_fn_7153и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_9593Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ш2х
>__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_fn_9600Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
І2Ѓ
F__inference_block4_conv1_layer_call_and_return_conditional_losses_7165и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block4_conv1_layer_call_fn_7175и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block4_conv2_layer_call_and_return_conditional_losses_7187и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block4_conv2_layer_call_fn_7197и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block4_conv3_layer_call_and_return_conditional_losses_7209и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block4_conv3_layer_call_fn_7219и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
F__inference_block4_conv4_layer_call_and_return_conditional_losses_7231и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block4_conv4_layer_call_fn_7241и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_9607Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ш2х
>__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_fn_9614Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_9619Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_1_layer_call_fn_9624Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_9629Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_3_layer_call_fn_9634Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_9639Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_5_layer_call_fn_9644Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_9649Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_7_layer_call_fn_9654Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_9662Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_9667Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_9675Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_4_layer_call_fn_9680Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_9688Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_7_layer_call_fn_9693Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2§
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_9701Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
х2т
;__inference_tf_op_layer_strided_slice_10_layer_call_fn_9706Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_9713Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_tf_op_layer_Range_layer_call_fn_9718Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_9723Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_tf_op_layer_Shape_layer_call_fn_9728Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_9735Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Range_1_layer_call_fn_9740Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_9745Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_2_layer_call_fn_9750Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_9757Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Range_2_layer_call_fn_9762Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_9767Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_4_layer_call_fn_9772Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_9779Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Range_3_layer_call_fn_9784Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_9789Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Shape_6_layer_call_fn_9794Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_9802Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_9807Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_9815Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_9820Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_9828Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_5_layer_call_fn_9833Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_9841Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_6_layer_call_fn_9846Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_9854Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_8_layer_call_fn_9859Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_9867Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_9_layer_call_fn_9872Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2§
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_9880Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
х2т
;__inference_tf_op_layer_strided_slice_11_layer_call_fn_9885Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2§
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_9893Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
х2т
;__inference_tf_op_layer_strided_slice_12_layer_call_fn_9898Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_9904Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_9910Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_9916Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_tf_op_layer_Prod_layer_call_fn_9921Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§2њ
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_9927Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
8__inference_tf_op_layer_BroadcastTo_1_layer_call_fn_9933Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_9939Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Prod_2_layer_call_fn_9944Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§2њ
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_9950Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
8__inference_tf_op_layer_BroadcastTo_2_layer_call_fn_9956Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_9962Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Prod_4_layer_call_fn_9967Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§2њ
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_9973Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
8__inference_tf_op_layer_BroadcastTo_3_layer_call_fn_9979Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_9985Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Prod_6_layer_call_fn_9990Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_9996Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_tf_op_layer_Mul_layer_call_fn_10002Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_10008Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Mul_1_layer_call_fn_10014Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_10020Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Mul_2_layer_call_fn_10026Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_10032Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Mul_3_layer_call_fn_10038Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_10044Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_AddV2_layer_call_fn_10050Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_10056Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Prod_1_layer_call_fn_10061Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_10067Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_tf_op_layer_AddV2_1_layer_call_fn_10073Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_10079Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Prod_3_layer_call_fn_10084Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_10090Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_tf_op_layer_AddV2_2_layer_call_fn_10096Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_10102Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Prod_5_layer_call_fn_10107Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_10113Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_tf_op_layer_AddV2_3_layer_call_fn_10119Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_10125Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_tf_op_layer_Prod_7_layer_call_fn_10130Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_10136Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_tf_op_layer_Reshape_layer_call_fn_10142Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_10148Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_tf_op_layer_Reshape_1_layer_call_fn_10154Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_10160Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_tf_op_layer_Reshape_2_layer_call_fn_10166Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_10172Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_tf_op_layer_Reshape_3_layer_call_fn_10178Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
І2Ѓ
F__inference_block5_conv1_layer_call_and_return_conditional_losses_7253и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block5_conv1_layer_call_fn_7263и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
§2њ
S__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_10184Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
8__inference_tf_op_layer_UnravelIndex_layer_call_fn_10190Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_10196Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_UnravelIndex_1_layer_call_fn_10202Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_10208Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_UnravelIndex_2_layer_call_fn_10214Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_10220Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_UnravelIndex_3_layer_call_fn_10226Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
І2Ѓ
F__inference_block5_conv2_layer_call_and_return_conditional_losses_7275и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block5_conv2_layer_call_fn_7285и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
њ2ї
P__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_10232Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_tf_op_layer_Transpose_layer_call_fn_10237Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
R__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_10243Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
7__inference_tf_op_layer_Transpose_1_layer_call_fn_10248Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
R__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_10254Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
7__inference_tf_op_layer_Transpose_2_layer_call_fn_10259Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
R__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_10265Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
7__inference_tf_op_layer_Transpose_3_layer_call_fn_10270Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
1B/
"__inference_signature_wrapper_8932input_2Ў
__inference__wrapped_model_6977.[\abklqr{|ЃЄЉЊѓєJЂG
@Ђ=
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Њ
Q
block5_conv2A>
block5_conv2,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
H
tf_op_layer_Transpose/,
tf_op_layer_Transposeџџџџџџџџџ	
L
tf_op_layer_Transpose_11.
tf_op_layer_Transpose_1џџџџџџџџџ	
L
tf_op_layer_Transpose_21.
tf_op_layer_Transpose_2џџџџџџџџџ	
L
tf_op_layer_Transpose_31.
tf_op_layer_Transpose_3џџџџџџџџџ	л
F__inference_block1_conv1_layer_call_and_return_conditional_losses_6989[\IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Г
+__inference_block1_conv1_layer_call_fn_6999[\IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@л
F__inference_block1_conv2_layer_call_and_return_conditional_losses_7011abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Г
+__inference_block1_conv2_layer_call_fn_7021abIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@м
F__inference_block2_conv1_layer_call_and_return_conditional_losses_7033klIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
+__inference_block2_conv1_layer_call_fn_7043klIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџн
F__inference_block2_conv2_layer_call_and_return_conditional_losses_7055qrJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
+__inference_block2_conv2_layer_call_fn_7065qrJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџн
F__inference_block3_conv1_layer_call_and_return_conditional_losses_7077{|JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
+__inference_block3_conv1_layer_call_fn_7087{|JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block3_conv2_layer_call_and_return_conditional_losses_7099JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block3_conv2_layer_call_fn_7109JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block3_conv3_layer_call_and_return_conditional_losses_7121JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block3_conv3_layer_call_fn_7131JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block3_conv4_layer_call_and_return_conditional_losses_7143JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block3_conv4_layer_call_fn_7153JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block4_conv1_layer_call_and_return_conditional_losses_7165JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block4_conv1_layer_call_fn_7175JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block4_conv2_layer_call_and_return_conditional_losses_7187JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block4_conv2_layer_call_fn_7197JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block4_conv3_layer_call_and_return_conditional_losses_7209ЃЄJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block4_conv3_layer_call_fn_7219ЃЄJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block4_conv4_layer_call_and_return_conditional_losses_7231ЉЊJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block4_conv4_layer_call_fn_7241ЉЊJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block5_conv1_layer_call_and_return_conditional_losses_7253ѓєJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block5_conv1_layer_call_fn_7263ѓєJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
F__inference_block5_conv2_layer_call_and_return_conditional_losses_7275JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
+__inference_block5_conv2_layer_call_fn_7285JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?__inference_model_layer_call_and_return_conditional_losses_8290Э.[\abklqr{|ЃЄЉЊѓєRЂO
HЂE
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "ЦЂТ
КЖ
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

0/1џџџџџџџџџ	

0/2џџџџџџџџџ	

0/3џџџџџџџџџ	

0/4џџџџџџџџџ	
 
?__inference_model_layer_call_and_return_conditional_losses_8434Э.[\abklqr{|ЃЄЉЊѓєRЂO
HЂE
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "ЦЂТ
КЖ
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

0/1џџџџџџџџџ	

0/2џџџџџџџџџ	

0/3џџџџџџџџџ	

0/4џџџџџџџџџ	
 
?__inference_model_layer_call_and_return_conditional_losses_9164Ь.[\abklqr{|ЃЄЉЊѓєQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "ЦЂТ
КЖ
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

0/1џџџџџџџџџ	

0/2џџџџџџџџџ	

0/3џџџџџџџџџ	

0/4џџџџџџџџџ	
 
?__inference_model_layer_call_and_return_conditional_losses_9396Ь.[\abklqr{|ЃЄЉЊѓєQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "ЦЂТ
КЖ
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

0/1џџџџџџџџџ	

0/2џџџџџџџџџ	

0/3џџџџџџџџџ	

0/4џџџџџџџџџ	
 р
$__inference_model_layer_call_fn_8648З.[\abklqr{|ЃЄЉЊѓєRЂO
HЂE
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "АЌ
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

1џџџџџџџџџ	

2џџџџџџџџџ	

3џџџџџџџџџ	

4џџџџџџџџџ	р
$__inference_model_layer_call_fn_8861З.[\abklqr{|ЃЄЉЊѓєRЂO
HЂE
;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "АЌ
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

1џџџџџџџџџ	

2џџџџџџџџџ	

3џџџџџџџџџ	

4џџџџџџџџџ	п
$__inference_model_layer_call_fn_9465Ж.[\abklqr{|ЃЄЉЊѓєQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "АЌ
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

1џџџџџџџџџ	

2џџџџџџџџџ	

3џџџџџџџџџ	

4џџџџџџџџџ	п
$__inference_model_layer_call_fn_9534Ж.[\abklqr{|ЃЄЉЊѓєQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "АЌ
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ

1џџџџџџџџџ	

2џџџџџџџџџ	

3џџџџџџџџџ	

4џџџџџџџџџ	М
"__inference_signature_wrapper_8932.[\abklqr{|ЃЄЉЊѓєUЂR
Ђ 
KЊH
F
input_2;8
input_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"Њ
Q
block5_conv2A>
block5_conv2,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
H
tf_op_layer_Transpose/,
tf_op_layer_Transposeџџџџџџџџџ	
L
tf_op_layer_Transpose_11.
tf_op_layer_Transpose_1џџџџџџџџџ	
L
tf_op_layer_Transpose_21.
tf_op_layer_Transpose_2џџџџџџџџџ	
L
tf_op_layer_Transpose_31.
tf_op_layer_Transpose_3џџџџџџџџџ	Е
N__inference_tf_op_layer_AddV2_1_layer_call_and_return_conditional_losses_10067тЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
3__inference_tf_op_layer_AddV2_1_layer_call_fn_10073еЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Е
N__inference_tf_op_layer_AddV2_2_layer_call_and_return_conditional_losses_10090тЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
3__inference_tf_op_layer_AddV2_2_layer_call_fn_10096еЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Е
N__inference_tf_op_layer_AddV2_3_layer_call_and_return_conditional_losses_10113тЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
3__inference_tf_op_layer_AddV2_3_layer_call_fn_10119еЂ
Ђ

=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Б
L__inference_tf_op_layer_AddV2_layer_call_and_return_conditional_losses_10044рЂ
Ђ

<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
 
1__inference_tf_op_layer_AddV2_layer_call_fn_10050гЂ
Ђ

<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
EB
inputs/14џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	о
M__inference_tf_op_layer_BiasAdd_layer_call_and_return_conditional_losses_9553IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
2__inference_tf_op_layer_BiasAdd_layer_call_fn_9558IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџљ
S__inference_tf_op_layer_BroadcastTo_1_layer_call_and_return_conditional_losses_9927ЁUЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 б
8__inference_tf_op_layer_BroadcastTo_1_layer_call_fn_9933UЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	љ
S__inference_tf_op_layer_BroadcastTo_2_layer_call_and_return_conditional_losses_9950ЁUЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 б
8__inference_tf_op_layer_BroadcastTo_2_layer_call_fn_9956UЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	љ
S__inference_tf_op_layer_BroadcastTo_3_layer_call_and_return_conditional_losses_9973ЁUЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 б
8__inference_tf_op_layer_BroadcastTo_3_layer_call_fn_9979UЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	ї
Q__inference_tf_op_layer_BroadcastTo_layer_call_and_return_conditional_losses_9904ЁUЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 Я
6__inference_tf_op_layer_BroadcastTo_layer_call_fn_9910UЂR
KЂH
FC
*'
inputs/0џџџџџџџџџ	

inputs/1	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	Ў
Y__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_and_return_conditional_losses_9579аJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ~
wt
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
85
0/1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
>__inference_tf_op_layer_MaxPoolWithArgmax_1_layer_call_fn_9586СJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "sp
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
63
1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Ў
Y__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_and_return_conditional_losses_9593аJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ~
wt
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
85
0/1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
>__inference_tf_op_layer_MaxPoolWithArgmax_2_layer_call_fn_9600СJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "sp
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
63
1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Ў
Y__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_and_return_conditional_losses_9607аJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ~
wt
85
0/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
85
0/1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 
>__inference_tf_op_layer_MaxPoolWithArgmax_3_layer_call_fn_9614СJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "sp
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
63
1,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	Ј
W__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_and_return_conditional_losses_9565ЬIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "Ђ|
ur
74
0/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
74
0/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
 џ
<__inference_tf_op_layer_MaxPoolWithArgmax_layer_call_fn_9572ОIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "qn
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
52
1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
L__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_10008ИlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 с
1__inference_tf_op_layer_Mul_1_layer_call_fn_10014ЋlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
L__inference_tf_op_layer_Mul_2_layer_call_and_return_conditional_losses_10020ИlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 с
1__inference_tf_op_layer_Mul_2_layer_call_fn_10026ЋlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
L__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_10032ИlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 с
1__inference_tf_op_layer_Mul_3_layer_call_fn_10038ЋlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_9996ИlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 п
/__inference_tf_op_layer_Mul_layer_call_fn_10002ЋlЂi
bЂ_
]Z
EB
inputs/04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1 	
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ	
M__inference_tf_op_layer_Prod_1_layer_call_and_return_conditional_losses_10056>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 g
2__inference_tf_op_layer_Prod_1_layer_call_fn_100611"Ђ
Ђ

inputs	
Њ "	
L__inference_tf_op_layer_Prod_2_layer_call_and_return_conditional_losses_9939:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 b
1__inference_tf_op_layer_Prod_2_layer_call_fn_9944-"Ђ
Ђ

inputs	
Њ " 	
M__inference_tf_op_layer_Prod_3_layer_call_and_return_conditional_losses_10079>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 g
2__inference_tf_op_layer_Prod_3_layer_call_fn_100841"Ђ
Ђ

inputs	
Њ "	
L__inference_tf_op_layer_Prod_4_layer_call_and_return_conditional_losses_9962:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 b
1__inference_tf_op_layer_Prod_4_layer_call_fn_9967-"Ђ
Ђ

inputs	
Њ " 	
M__inference_tf_op_layer_Prod_5_layer_call_and_return_conditional_losses_10102>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 g
2__inference_tf_op_layer_Prod_5_layer_call_fn_101071"Ђ
Ђ

inputs	
Њ "	
L__inference_tf_op_layer_Prod_6_layer_call_and_return_conditional_losses_9985:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 b
1__inference_tf_op_layer_Prod_6_layer_call_fn_9990-"Ђ
Ђ

inputs	
Њ " 	
M__inference_tf_op_layer_Prod_7_layer_call_and_return_conditional_losses_10125>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 g
2__inference_tf_op_layer_Prod_7_layer_call_fn_101301"Ђ
Ђ

inputs	
Њ "	
J__inference_tf_op_layer_Prod_layer_call_and_return_conditional_losses_9916:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 `
/__inference_tf_op_layer_Prod_layer_call_fn_9921-"Ђ
Ђ

inputs	
Њ " 	
M__inference_tf_op_layer_Range_1_layer_call_and_return_conditional_losses_9735CЂ
Ђ

inputs 	
Њ "!Ђ

0џџџџџџџџџ	
 l
2__inference_tf_op_layer_Range_1_layer_call_fn_97406Ђ
Ђ

inputs 	
Њ "џџџџџџџџџ	
M__inference_tf_op_layer_Range_2_layer_call_and_return_conditional_losses_9757CЂ
Ђ

inputs 	
Њ "!Ђ

0џџџџџџџџџ	
 l
2__inference_tf_op_layer_Range_2_layer_call_fn_97626Ђ
Ђ

inputs 	
Њ "џџџџџџџџџ	
M__inference_tf_op_layer_Range_3_layer_call_and_return_conditional_losses_9779CЂ
Ђ

inputs 	
Њ "!Ђ

0џџџџџџџџџ	
 l
2__inference_tf_op_layer_Range_3_layer_call_fn_97846Ђ
Ђ

inputs 	
Њ "џџџџџџџџџ	
K__inference_tf_op_layer_Range_layer_call_and_return_conditional_losses_9713CЂ
Ђ

inputs 	
Њ "!Ђ

0џџџџџџџџџ	
 j
0__inference_tf_op_layer_Range_layer_call_fn_97186Ђ
Ђ

inputs 	
Њ "џџџџџџџџџ	т
P__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_10148hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "!Ђ

0џџџџџџџџџ	
 К
5__inference_tf_op_layer_Reshape_1_layer_call_fn_10154hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	т
P__inference_tf_op_layer_Reshape_2_layer_call_and_return_conditional_losses_10160hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "!Ђ

0џџџџџџџџџ	
 К
5__inference_tf_op_layer_Reshape_2_layer_call_fn_10166hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	т
P__inference_tf_op_layer_Reshape_3_layer_call_and_return_conditional_losses_10172hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "!Ђ

0џџџџџџџџџ	
 К
5__inference_tf_op_layer_Reshape_3_layer_call_fn_10178hЂe
^Ђ[
YV
=:
inputs/0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	п
N__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_10136gЂd
]ЂZ
XU
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	

inputs/1	
Њ "!Ђ

0џџџџџџџџџ	
 Ж
3__inference_tf_op_layer_Reshape_layer_call_fn_10142gЂd
]ЂZ
XU
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	

inputs/1	
Њ "џџџџџџџџџ	Ж
M__inference_tf_op_layer_Shape_1_layer_call_and_return_conditional_losses_9619eIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_1_layer_call_fn_9624XIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@	
Њ "	З
M__inference_tf_op_layer_Shape_2_layer_call_and_return_conditional_losses_9745fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_2_layer_call_fn_9750YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "	З
M__inference_tf_op_layer_Shape_3_layer_call_and_return_conditional_losses_9629fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_3_layer_call_fn_9634YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "	З
M__inference_tf_op_layer_Shape_4_layer_call_and_return_conditional_losses_9767fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_4_layer_call_fn_9772YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "	З
M__inference_tf_op_layer_Shape_5_layer_call_and_return_conditional_losses_9639fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_5_layer_call_fn_9644YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "	З
M__inference_tf_op_layer_Shape_6_layer_call_and_return_conditional_losses_9789fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_6_layer_call_fn_9794YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "	З
M__inference_tf_op_layer_Shape_7_layer_call_and_return_conditional_losses_9649fJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "Ђ

0	
 
2__inference_tf_op_layer_Shape_7_layer_call_fn_9654YJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "	Д
K__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_9723eIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "Ђ

0	
 
0__inference_tf_op_layer_Shape_layer_call_fn_9728XIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "	Ў
R__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_10243X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ	
 
7__inference_tf_op_layer_Transpose_1_layer_call_fn_10248K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "џџџџџџџџџ	Ў
R__inference_tf_op_layer_Transpose_2_layer_call_and_return_conditional_losses_10254X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ	
 
7__inference_tf_op_layer_Transpose_2_layer_call_fn_10259K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "џџџџџџџџџ	Ў
R__inference_tf_op_layer_Transpose_3_layer_call_and_return_conditional_losses_10265X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ	
 
7__inference_tf_op_layer_Transpose_3_layer_call_fn_10270K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "џџџџџџџџџ	Ќ
P__inference_tf_op_layer_Transpose_layer_call_and_return_conditional_losses_10232X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ	
 
5__inference_tf_op_layer_Transpose_layer_call_fn_10237K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ	
Њ "џџџџџџџџџ	Ы
U__inference_tf_op_layer_UnravelIndex_1_layer_call_and_return_conditional_losses_10196rIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "%Ђ"

0џџџџџџџџџ	
 Ѓ
:__inference_tf_op_layer_UnravelIndex_1_layer_call_fn_10202eIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	Ы
U__inference_tf_op_layer_UnravelIndex_2_layer_call_and_return_conditional_losses_10208rIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "%Ђ"

0џџџџџџџџџ	
 Ѓ
:__inference_tf_op_layer_UnravelIndex_2_layer_call_fn_10214eIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	Ы
U__inference_tf_op_layer_UnravelIndex_3_layer_call_and_return_conditional_losses_10220rIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "%Ђ"

0џџџџџџџџџ	
 Ѓ
:__inference_tf_op_layer_UnravelIndex_3_layer_call_fn_10226eIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	Щ
S__inference_tf_op_layer_UnravelIndex_layer_call_and_return_conditional_losses_10184rIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "%Ђ"

0џџџџџџџџџ	
 Ё
8__inference_tf_op_layer_UnravelIndex_layer_call_fn_10190eIЂF
?Ђ<
:7

inputs/0џџџџџџџџџ	

inputs/1	
Њ "џџџџџџџџџ	
V__inference_tf_op_layer_strided_slice_10_layer_call_and_return_conditional_losses_9701:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 l
;__inference_tf_op_layer_strided_slice_10_layer_call_fn_9706-"Ђ
Ђ

inputs	
Њ " 	Ж
V__inference_tf_op_layer_strided_slice_11_layer_call_and_return_conditional_losses_9880\+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
;__inference_tf_op_layer_strided_slice_11_layer_call_fn_9885O+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	
V__inference_tf_op_layer_strided_slice_12_layer_call_and_return_conditional_losses_9893>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 p
;__inference_tf_op_layer_strided_slice_12_layer_call_fn_98981"Ђ
Ђ

inputs	
Њ "	
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_9662:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 k
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_9667-"Ђ
Ђ

inputs	
Њ " 	Е
U__inference_tf_op_layer_strided_slice_2_layer_call_and_return_conditional_losses_9802\+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
:__inference_tf_op_layer_strided_slice_2_layer_call_fn_9807O+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	
U__inference_tf_op_layer_strided_slice_3_layer_call_and_return_conditional_losses_9815>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 o
:__inference_tf_op_layer_strided_slice_3_layer_call_fn_98201"Ђ
Ђ

inputs	
Њ "	
U__inference_tf_op_layer_strided_slice_4_layer_call_and_return_conditional_losses_9675:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 k
:__inference_tf_op_layer_strided_slice_4_layer_call_fn_9680-"Ђ
Ђ

inputs	
Њ " 	Е
U__inference_tf_op_layer_strided_slice_5_layer_call_and_return_conditional_losses_9828\+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
:__inference_tf_op_layer_strided_slice_5_layer_call_fn_9833O+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	
U__inference_tf_op_layer_strided_slice_6_layer_call_and_return_conditional_losses_9841>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 o
:__inference_tf_op_layer_strided_slice_6_layer_call_fn_98461"Ђ
Ђ

inputs	
Њ "	
U__inference_tf_op_layer_strided_slice_7_layer_call_and_return_conditional_losses_9688:"Ђ
Ђ

inputs	
Њ "Ђ


0 	
 k
:__inference_tf_op_layer_strided_slice_7_layer_call_fn_9693-"Ђ
Ђ

inputs	
Њ " 	Е
U__inference_tf_op_layer_strided_slice_8_layer_call_and_return_conditional_losses_9854\+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
:__inference_tf_op_layer_strided_slice_8_layer_call_fn_9859O+Ђ(
!Ђ

inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	
U__inference_tf_op_layer_strided_slice_9_layer_call_and_return_conditional_losses_9867>"Ђ
Ђ

inputs	
Њ "Ђ

0	
 o
:__inference_tf_op_layer_strided_slice_9_layer_call_fn_98721"Ђ
Ђ

inputs	
Њ "	ф
S__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_9542IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Л
8__inference_tf_op_layer_strided_slice_layer_call_fn_9547IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ