��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.10.02unknown8ؒ
�
"Adam/lstm_122/lstm_cell_403/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/lstm_122/lstm_cell_403/bias/v
�
6Adam/lstm_122/lstm_cell_403/bias/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_122/lstm_cell_403/bias/v*
_output_shapes	
:�*
dtype0
�
.Adam/lstm_122/lstm_cell_403/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*?
shared_name0.Adam/lstm_122/lstm_cell_403/recurrent_kernel/v
�
BAdam/lstm_122/lstm_cell_403/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/lstm_122/lstm_cell_403/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
$Adam/lstm_122/lstm_cell_403/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*5
shared_name&$Adam/lstm_122/lstm_cell_403/kernel/v
�
8Adam/lstm_122/lstm_cell_403/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/lstm_122/lstm_cell_403/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/v
y
(Adam/dense_87/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_87/kernel/v
�
*Adam/dense_87/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/v*
_output_shapes

:d*
dtype0
�
"Adam/lstm_122/lstm_cell_403/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/lstm_122/lstm_cell_403/bias/m
�
6Adam/lstm_122/lstm_cell_403/bias/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_122/lstm_cell_403/bias/m*
_output_shapes	
:�*
dtype0
�
.Adam/lstm_122/lstm_cell_403/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*?
shared_name0.Adam/lstm_122/lstm_cell_403/recurrent_kernel/m
�
BAdam/lstm_122/lstm_cell_403/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/lstm_122/lstm_cell_403/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
$Adam/lstm_122/lstm_cell_403/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*5
shared_name&$Adam/lstm_122/lstm_cell_403/kernel/m
�
8Adam/lstm_122/lstm_cell_403/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/lstm_122/lstm_cell_403/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/m
y
(Adam/dense_87/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_87/kernel/m
�
*Adam/dense_87/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/m*
_output_shapes

:d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
lstm_122/lstm_cell_403/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namelstm_122/lstm_cell_403/bias
�
/lstm_122/lstm_cell_403/bias/Read/ReadVariableOpReadVariableOplstm_122/lstm_cell_403/bias*
_output_shapes	
:�*
dtype0
�
'lstm_122/lstm_cell_403/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*8
shared_name)'lstm_122/lstm_cell_403/recurrent_kernel
�
;lstm_122/lstm_cell_403/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_122/lstm_cell_403/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
lstm_122/lstm_cell_403/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_namelstm_122/lstm_cell_403/kernel
�
1lstm_122/lstm_cell_403/kernel/Read/ReadVariableOpReadVariableOplstm_122/lstm_cell_403/kernel*
_output_shapes
:	�*
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:*
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

:d*
dtype0
�
serving_default_lstm_122_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_122_inputlstm_122/lstm_cell_403/kernel'lstm_122/lstm_cell_403/recurrent_kernellstm_122/lstm_cell_403/biasdense_87/kerneldense_87/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_60323510

NoOpNoOp
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�/
value�/B�/ B�/
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
�
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
%trace_0
&trace_1
'trace_2
(trace_3* 
6
)trace_0
*trace_1
+trace_2
,trace_3* 
* 
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemimjmkmlmmvnvovpvqvr*

2serving_default* 

0
1
2*

0
1
2*
* 
�

3states
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
9trace_0
:trace_1
;trace_2
<trace_3* 
6
=trace_0
>trace_1
?trace_2
@trace_3* 
* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator
H
state_size

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
_Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_87/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_122/lstm_cell_403/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_122/lstm_cell_403/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_122/lstm_cell_403/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

P0
Q1
R2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

Xtrace_0
Ytrace_1* 

Ztrace_0
[trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
\	variables
]	keras_api
	^total
	_count*
8
`	variables
a	keras_api
	btotal
	ccount*
H
d	variables
e	keras_api
	ftotal
	gcount
h
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*

\	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

`	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

d	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�|
VARIABLE_VALUEAdam/dense_87/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_87/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/lstm_122/lstm_cell_403/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE.Adam/lstm_122/lstm_cell_403/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_122/lstm_cell_403/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_87/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_87/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/lstm_122/lstm_cell_403/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE.Adam/lstm_122/lstm_cell_403/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_122/lstm_cell_403/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOp1lstm_122/lstm_cell_403/kernel/Read/ReadVariableOp;lstm_122/lstm_cell_403/recurrent_kernel/Read/ReadVariableOp/lstm_122/lstm_cell_403/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_87/kernel/m/Read/ReadVariableOp(Adam/dense_87/bias/m/Read/ReadVariableOp8Adam/lstm_122/lstm_cell_403/kernel/m/Read/ReadVariableOpBAdam/lstm_122/lstm_cell_403/recurrent_kernel/m/Read/ReadVariableOp6Adam/lstm_122/lstm_cell_403/bias/m/Read/ReadVariableOp*Adam/dense_87/kernel/v/Read/ReadVariableOp(Adam/dense_87/bias/v/Read/ReadVariableOp8Adam/lstm_122/lstm_cell_403/kernel/v/Read/ReadVariableOpBAdam/lstm_122/lstm_cell_403/recurrent_kernel/v/Read/ReadVariableOp6Adam/lstm_122/lstm_cell_403/bias/v/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_60324684
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_87/kerneldense_87/biaslstm_122/lstm_cell_403/kernel'lstm_122/lstm_cell_403/recurrent_kernellstm_122/lstm_cell_403/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/dense_87/kernel/mAdam/dense_87/bias/m$Adam/lstm_122/lstm_cell_403/kernel/m.Adam/lstm_122/lstm_cell_403/recurrent_kernel/m"Adam/lstm_122/lstm_cell_403/bias/mAdam/dense_87/kernel/vAdam/dense_87/bias/v$Adam/lstm_122/lstm_cell_403/kernel/v.Adam/lstm_122/lstm_cell_403/recurrent_kernel/v"Adam/lstm_122/lstm_cell_403/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_60324772ז
�D
�

lstm_122_while_body_60323751.
*lstm_122_while_lstm_122_while_loop_counter4
0lstm_122_while_lstm_122_while_maximum_iterations
lstm_122_while_placeholder 
lstm_122_while_placeholder_1 
lstm_122_while_placeholder_2 
lstm_122_while_placeholder_3-
)lstm_122_while_lstm_122_strided_slice_1_0i
elstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0P
=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0:	�R
?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�M
>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
lstm_122_while_identity
lstm_122_while_identity_1
lstm_122_while_identity_2
lstm_122_while_identity_3
lstm_122_while_identity_4
lstm_122_while_identity_5+
'lstm_122_while_lstm_122_strided_slice_1g
clstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensorN
;lstm_122_while_lstm_cell_403_matmul_readvariableop_resource:	�P
=lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource:	d�K
<lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource:	���3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp�2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp�4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp�
@lstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_122/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0lstm_122_while_placeholderIlstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
#lstm_122/while/lstm_cell_403/MatMulMatMul9lstm_122/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
%lstm_122/while/lstm_cell_403/MatMul_1MatMullstm_122_while_placeholder_2<lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 lstm_122/while/lstm_cell_403/addAddV2-lstm_122/while/lstm_cell_403/MatMul:product:0/lstm_122/while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
$lstm_122/while/lstm_cell_403/BiasAddBiasAdd$lstm_122/while/lstm_cell_403/add:z:0;lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������n
,lstm_122/while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_122/while/lstm_cell_403/splitSplit5lstm_122/while/lstm_cell_403/split/split_dim:output:0-lstm_122/while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
$lstm_122/while/lstm_cell_403/SigmoidSigmoid+lstm_122/while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
&lstm_122/while/lstm_cell_403/Sigmoid_1Sigmoid+lstm_122/while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
 lstm_122/while/lstm_cell_403/mulMul*lstm_122/while/lstm_cell_403/Sigmoid_1:y:0lstm_122_while_placeholder_3*
T0*'
_output_shapes
:���������d�
!lstm_122/while/lstm_cell_403/ReluRelu+lstm_122/while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/mul_1Mul(lstm_122/while/lstm_cell_403/Sigmoid:y:0/lstm_122/while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/add_1AddV2$lstm_122/while/lstm_cell_403/mul:z:0&lstm_122/while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
&lstm_122/while/lstm_cell_403/Sigmoid_2Sigmoid+lstm_122/while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������d�
#lstm_122/while/lstm_cell_403/Relu_1Relu&lstm_122/while/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/mul_2Mul*lstm_122/while/lstm_cell_403/Sigmoid_2:y:01lstm_122/while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������d{
9lstm_122/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_122/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_122_while_placeholder_1Blstm_122/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_122/while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_122/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_122/while/addAddV2lstm_122_while_placeholderlstm_122/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_122/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/while/add_1AddV2*lstm_122_while_lstm_122_while_loop_counterlstm_122/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_122/while/IdentityIdentitylstm_122/while/add_1:z:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_1Identity0lstm_122_while_lstm_122_while_maximum_iterations^lstm_122/while/NoOp*
T0*
_output_shapes
: t
lstm_122/while/Identity_2Identitylstm_122/while/add:z:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_3IdentityClstm_122/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_4Identity&lstm_122/while/lstm_cell_403/mul_2:z:0^lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_122/while/Identity_5Identity&lstm_122/while/lstm_cell_403/add_1:z:0^lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_122/while/NoOpNoOp4^lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp3^lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp5^lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_122_while_identity lstm_122/while/Identity:output:0"?
lstm_122_while_identity_1"lstm_122/while/Identity_1:output:0"?
lstm_122_while_identity_2"lstm_122/while/Identity_2:output:0"?
lstm_122_while_identity_3"lstm_122/while/Identity_3:output:0"?
lstm_122_while_identity_4"lstm_122/while/Identity_4:output:0"?
lstm_122_while_identity_5"lstm_122/while/Identity_5:output:0"T
'lstm_122_while_lstm_122_strided_slice_1)lstm_122_while_lstm_122_strided_slice_1_0"~
<lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0"�
=lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0"|
;lstm_122_while_lstm_cell_403_matmul_readvariableop_resource=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0"�
clstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensorelstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2j
3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp2h
2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp2l
4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_78_layer_call_fn_60323217
lstm_122_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_122_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�^
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323691

inputsH
5lstm_122_lstm_cell_403_matmul_readvariableop_resource:	�J
7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource:	d�E
6lstm_122_lstm_cell_403_biasadd_readvariableop_resource:	�9
'dense_87_matmul_readvariableop_resource:d6
(dense_87_biasadd_readvariableop_resource:
identity��dense_87/BiasAdd/ReadVariableOp�dense_87/MatMul/ReadVariableOp�-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp�,lstm_122/lstm_cell_403/MatMul/ReadVariableOp�.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp�lstm_122/whileD
lstm_122/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_122/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_122/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_122/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_sliceStridedSlicelstm_122/Shape:output:0%lstm_122/strided_slice/stack:output:0'lstm_122/strided_slice/stack_1:output:0'lstm_122/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_122/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_122/zeros/packedPacklstm_122/strided_slice:output:0 lstm_122/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_122/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_122/zerosFilllstm_122/zeros/packed:output:0lstm_122/zeros/Const:output:0*
T0*'
_output_shapes
:���������d[
lstm_122/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_122/zeros_1/packedPacklstm_122/strided_slice:output:0"lstm_122/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_122/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_122/zeros_1Fill lstm_122/zeros_1/packed:output:0lstm_122/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dl
lstm_122/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_122/transpose	Transposeinputs lstm_122/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_122/Shape_1Shapelstm_122/transpose:y:0*
T0*
_output_shapes
:h
lstm_122/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_122/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_1StridedSlicelstm_122/Shape_1:output:0'lstm_122/strided_slice_1/stack:output:0)lstm_122/strided_slice_1/stack_1:output:0)lstm_122/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_122/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_122/TensorArrayV2TensorListReserve-lstm_122/TensorArrayV2/element_shape:output:0!lstm_122/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_122/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_122/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_122/transpose:y:0Glstm_122/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_122/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_122/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_2StridedSlicelstm_122/transpose:y:0'lstm_122/strided_slice_2/stack:output:0)lstm_122/strided_slice_2/stack_1:output:0)lstm_122/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_122/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp5lstm_122_lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_122/lstm_cell_403/MatMulMatMul!lstm_122/strided_slice_2:output:04lstm_122/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_122/lstm_cell_403/MatMul_1MatMullstm_122/zeros:output:06lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_122/lstm_cell_403/addAddV2'lstm_122/lstm_cell_403/MatMul:product:0)lstm_122/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp6lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_122/lstm_cell_403/BiasAddBiasAddlstm_122/lstm_cell_403/add:z:05lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
&lstm_122/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/lstm_cell_403/splitSplit/lstm_122/lstm_cell_403/split/split_dim:output:0'lstm_122/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
lstm_122/lstm_cell_403/SigmoidSigmoid%lstm_122/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
 lstm_122/lstm_cell_403/Sigmoid_1Sigmoid%lstm_122/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mulMul$lstm_122/lstm_cell_403/Sigmoid_1:y:0lstm_122/zeros_1:output:0*
T0*'
_output_shapes
:���������d|
lstm_122/lstm_cell_403/ReluRelu%lstm_122/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mul_1Mul"lstm_122/lstm_cell_403/Sigmoid:y:0)lstm_122/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/add_1AddV2lstm_122/lstm_cell_403/mul:z:0 lstm_122/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_122/lstm_cell_403/Sigmoid_2Sigmoid%lstm_122/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dy
lstm_122/lstm_cell_403/Relu_1Relu lstm_122/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mul_2Mul$lstm_122/lstm_cell_403/Sigmoid_2:y:0+lstm_122/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dw
&lstm_122/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   g
%lstm_122/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/TensorArrayV2_1TensorListReserve/lstm_122/TensorArrayV2_1/element_shape:output:0.lstm_122/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_122/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_122/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_122/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_122/whileWhile$lstm_122/while/loop_counter:output:0*lstm_122/while/maximum_iterations:output:0lstm_122/time:output:0!lstm_122/TensorArrayV2_1:handle:0lstm_122/zeros:output:0lstm_122/zeros_1:output:0!lstm_122/strided_slice_1:output:0@lstm_122/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_122_lstm_cell_403_matmul_readvariableop_resource7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource6lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_122_while_body_60323600*(
cond R
lstm_122_while_cond_60323599*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
9lstm_122/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
+lstm_122/TensorArrayV2Stack/TensorListStackTensorListStacklstm_122/while:output:3Blstm_122/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsq
lstm_122/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_122/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_3StridedSlice4lstm_122/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_122/strided_slice_3/stack:output:0)lstm_122/strided_slice_3/stack_1:output:0)lstm_122/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskn
lstm_122/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_122/transpose_1	Transpose4lstm_122/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_122/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd
lstm_122/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
dense_87/MatMulMatMul!lstm_122/strided_slice_3:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp.^lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp-^lstm_122/lstm_cell_403/MatMul/ReadVariableOp/^lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp^lstm_122/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2^
-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp2\
,lstm_122/lstm_cell_403/MatMul/ReadVariableOp,lstm_122/lstm_cell_403/MatMul/ReadVariableOp2`
.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp2 
lstm_122/whilelstm_122/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_122_layer_call_fn_60323886

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323385o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60322826

inputs)
lstm_cell_403_60322742:	�)
lstm_cell_403_60322744:	d�%
lstm_cell_403_60322746:	�
identity��%lstm_cell_403/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%lstm_cell_403/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_403_60322742lstm_cell_403_60322744lstm_cell_403_60322746*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322741n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_403_60322742lstm_cell_403_60322744lstm_cell_403_60322746*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60322756*
condR
while_cond_60322755*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dv
NoOpNoOp&^lstm_cell_403/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_403/StatefulPartitionedCall%lstm_cell_403/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323204

inputs$
lstm_122_60323180:	�$
lstm_122_60323182:	d� 
lstm_122_60323184:	�#
dense_87_60323198:d
dense_87_60323200:
identity�� dense_87/StatefulPartitionedCall� lstm_122/StatefulPartitionedCall�
 lstm_122/StatefulPartitionedCallStatefulPartitionedCallinputslstm_122_60323180lstm_122_60323182lstm_122_60323184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323179�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)lstm_122/StatefulPartitionedCall:output:0dense_87_60323198dense_87_60323200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_122/StatefulPartitionedCall lstm_122/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324466

inputs?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60324381*
condR
while_cond_60324380*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_122_layer_call_fn_60323853
inputs_0
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60322826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_60322948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60322948___redundant_placeholder06
2while_while_cond_60322948___redundant_placeholder16
2while_while_cond_60322948___redundant_placeholder26
2while_while_cond_60322948___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_60323300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�K
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324321

inputs?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60324236*
condR
while_cond_60324235*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_60323945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60323945___redundant_placeholder06
2while_while_cond_60323945___redundant_placeholder16
2while_while_cond_60323945___redundant_placeholder26
2while_while_cond_60323945___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322741

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�

�
lstm_122_while_cond_60323599.
*lstm_122_while_lstm_122_while_loop_counter4
0lstm_122_while_lstm_122_while_maximum_iterations
lstm_122_while_placeholder 
lstm_122_while_placeholder_1 
lstm_122_while_placeholder_2 
lstm_122_while_placeholder_30
,lstm_122_while_less_lstm_122_strided_slice_1H
Dlstm_122_while_lstm_122_while_cond_60323599___redundant_placeholder0H
Dlstm_122_while_lstm_122_while_cond_60323599___redundant_placeholder1H
Dlstm_122_while_lstm_122_while_cond_60323599___redundant_placeholder2H
Dlstm_122_while_lstm_122_while_cond_60323599___redundant_placeholder3
lstm_122_while_identity
�
lstm_122/while/LessLesslstm_122_while_placeholder,lstm_122_while_less_lstm_122_strided_slice_1*
T0*
_output_shapes
: ]
lstm_122/while/IdentityIdentitylstm_122/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_122_while_identity lstm_122/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
0__inference_lstm_cell_403_layer_call_fn_60324519

inputs
states_0
states_1
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322889o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323427

inputs$
lstm_122_60323414:	�$
lstm_122_60323416:	d� 
lstm_122_60323418:	�#
dense_87_60323421:d
dense_87_60323423:
identity�� dense_87/StatefulPartitionedCall� lstm_122/StatefulPartitionedCall�
 lstm_122/StatefulPartitionedCallStatefulPartitionedCallinputslstm_122_60323414lstm_122_60323416lstm_122_60323418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323385�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)lstm_122/StatefulPartitionedCall:output:0dense_87_60323421dense_87_60323423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_122/StatefulPartitionedCall lstm_122/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�D
�

lstm_122_while_body_60323600.
*lstm_122_while_lstm_122_while_loop_counter4
0lstm_122_while_lstm_122_while_maximum_iterations
lstm_122_while_placeholder 
lstm_122_while_placeholder_1 
lstm_122_while_placeholder_2 
lstm_122_while_placeholder_3-
)lstm_122_while_lstm_122_strided_slice_1_0i
elstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0P
=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0:	�R
?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�M
>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
lstm_122_while_identity
lstm_122_while_identity_1
lstm_122_while_identity_2
lstm_122_while_identity_3
lstm_122_while_identity_4
lstm_122_while_identity_5+
'lstm_122_while_lstm_122_strided_slice_1g
clstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensorN
;lstm_122_while_lstm_cell_403_matmul_readvariableop_resource:	�P
=lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource:	d�K
<lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource:	���3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp�2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp�4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp�
@lstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_122/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0lstm_122_while_placeholderIlstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
#lstm_122/while/lstm_cell_403/MatMulMatMul9lstm_122/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
%lstm_122/while/lstm_cell_403/MatMul_1MatMullstm_122_while_placeholder_2<lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 lstm_122/while/lstm_cell_403/addAddV2-lstm_122/while/lstm_cell_403/MatMul:product:0/lstm_122/while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
$lstm_122/while/lstm_cell_403/BiasAddBiasAdd$lstm_122/while/lstm_cell_403/add:z:0;lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������n
,lstm_122/while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_122/while/lstm_cell_403/splitSplit5lstm_122/while/lstm_cell_403/split/split_dim:output:0-lstm_122/while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
$lstm_122/while/lstm_cell_403/SigmoidSigmoid+lstm_122/while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
&lstm_122/while/lstm_cell_403/Sigmoid_1Sigmoid+lstm_122/while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
 lstm_122/while/lstm_cell_403/mulMul*lstm_122/while/lstm_cell_403/Sigmoid_1:y:0lstm_122_while_placeholder_3*
T0*'
_output_shapes
:���������d�
!lstm_122/while/lstm_cell_403/ReluRelu+lstm_122/while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/mul_1Mul(lstm_122/while/lstm_cell_403/Sigmoid:y:0/lstm_122/while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/add_1AddV2$lstm_122/while/lstm_cell_403/mul:z:0&lstm_122/while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
&lstm_122/while/lstm_cell_403/Sigmoid_2Sigmoid+lstm_122/while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������d�
#lstm_122/while/lstm_cell_403/Relu_1Relu&lstm_122/while/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
"lstm_122/while/lstm_cell_403/mul_2Mul*lstm_122/while/lstm_cell_403/Sigmoid_2:y:01lstm_122/while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������d{
9lstm_122/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_122/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_122_while_placeholder_1Blstm_122/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_122/while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_122/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_122/while/addAddV2lstm_122_while_placeholderlstm_122/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_122/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/while/add_1AddV2*lstm_122_while_lstm_122_while_loop_counterlstm_122/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_122/while/IdentityIdentitylstm_122/while/add_1:z:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_1Identity0lstm_122_while_lstm_122_while_maximum_iterations^lstm_122/while/NoOp*
T0*
_output_shapes
: t
lstm_122/while/Identity_2Identitylstm_122/while/add:z:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_3IdentityClstm_122/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_122/while/NoOp*
T0*
_output_shapes
: �
lstm_122/while/Identity_4Identity&lstm_122/while/lstm_cell_403/mul_2:z:0^lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_122/while/Identity_5Identity&lstm_122/while/lstm_cell_403/add_1:z:0^lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_122/while/NoOpNoOp4^lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp3^lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp5^lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_122_while_identity lstm_122/while/Identity:output:0"?
lstm_122_while_identity_1"lstm_122/while/Identity_1:output:0"?
lstm_122_while_identity_2"lstm_122/while/Identity_2:output:0"?
lstm_122_while_identity_3"lstm_122/while/Identity_3:output:0"?
lstm_122_while_identity_4"lstm_122/while/Identity_4:output:0"?
lstm_122_while_identity_5"lstm_122/while/Identity_5:output:0"T
'lstm_122_while_lstm_122_strided_slice_1)lstm_122_while_lstm_122_strided_slice_1_0"~
<lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource>lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0"�
=lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource?lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0"|
;lstm_122_while_lstm_cell_403_matmul_readvariableop_resource=lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0"�
clstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensorelstm_122_while_tensorarrayv2read_tensorlistgetitem_lstm_122_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2j
3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp3lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp2h
2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp2lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp2l
4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp4lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_60324235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60324235___redundant_placeholder06
2while_while_cond_60324235___redundant_placeholder16
2while_while_cond_60324235___redundant_placeholder26
2while_while_cond_60324235___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_60324380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60324380___redundant_placeholder06
2while_while_cond_60324380___redundant_placeholder16
2while_while_cond_60324380___redundant_placeholder26
2while_while_cond_60324380___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_122_layer_call_fn_60323864
inputs_0
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323019o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
0__inference_sequential_78_layer_call_fn_60323540

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323427o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_60323510
lstm_122_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_122_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_60322674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�9
�
while_body_60324091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_60322755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60322755___redundant_placeholder06
2while_while_cond_60322755___redundant_placeholder16
2while_while_cond_60322755___redundant_placeholder26
2while_while_cond_60322755___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�K
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323385

inputs?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60323300*
condR
while_cond_60323299*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_87_layer_call_and_return_conditional_losses_60324485

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�$
�
while_body_60322949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_403_60322973_0:	�1
while_lstm_cell_403_60322975_0:	d�-
while_lstm_cell_403_60322977_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_403_60322973:	�/
while_lstm_cell_403_60322975:	d�+
while_lstm_cell_403_60322977:	���+while/lstm_cell_403/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_403/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_403_60322973_0while_lstm_cell_403_60322975_0while_lstm_cell_403_60322977_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322889r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_403/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity4while/lstm_cell_403/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity4while/lstm_cell_403/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dz

while/NoOpNoOp,^while/lstm_cell_403/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_403_60322973while_lstm_cell_403_60322973_0">
while_lstm_cell_403_60322975while_lstm_cell_403_60322975_0">
while_lstm_cell_403_60322977while_lstm_cell_403_60322977_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2Z
+while/lstm_cell_403/StatefulPartitionedCall+while/lstm_cell_403/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324583

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�
�
while_cond_60323299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60323299___redundant_placeholder06
2while_while_cond_60323299___redundant_placeholder16
2while_while_cond_60323299___redundant_placeholder26
2while_while_cond_60323299___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
0__inference_sequential_78_layer_call_fn_60323455
lstm_122_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_122_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323427o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�
�
+__inference_lstm_122_layer_call_fn_60323875

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322889

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�9
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323019

inputs)
lstm_cell_403_60322935:	�)
lstm_cell_403_60322937:	d�%
lstm_cell_403_60322939:	�
identity��%lstm_cell_403/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%lstm_cell_403/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_403_60322935lstm_cell_403_60322937lstm_cell_403_60322939*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322889n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_403_60322935lstm_cell_403_60322937lstm_cell_403_60322939*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60322949*
condR
while_cond_60322948*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dv
NoOpNoOp&^lstm_cell_403/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_403/StatefulPartitionedCall%lstm_cell_403/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�:
�
!__inference__traced_save_60324684
file_prefix.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop<
8savev2_lstm_122_lstm_cell_403_kernel_read_readvariableopF
Bsavev2_lstm_122_lstm_cell_403_recurrent_kernel_read_readvariableop:
6savev2_lstm_122_lstm_cell_403_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_87_kernel_m_read_readvariableop3
/savev2_adam_dense_87_bias_m_read_readvariableopC
?savev2_adam_lstm_122_lstm_cell_403_kernel_m_read_readvariableopM
Isavev2_adam_lstm_122_lstm_cell_403_recurrent_kernel_m_read_readvariableopA
=savev2_adam_lstm_122_lstm_cell_403_bias_m_read_readvariableop5
1savev2_adam_dense_87_kernel_v_read_readvariableop3
/savev2_adam_dense_87_bias_v_read_readvariableopC
?savev2_adam_lstm_122_lstm_cell_403_kernel_v_read_readvariableopM
Isavev2_adam_lstm_122_lstm_cell_403_recurrent_kernel_v_read_readvariableopA
=savev2_adam_lstm_122_lstm_cell_403_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop8savev2_lstm_122_lstm_cell_403_kernel_read_readvariableopBsavev2_lstm_122_lstm_cell_403_recurrent_kernel_read_readvariableop6savev2_lstm_122_lstm_cell_403_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_87_kernel_m_read_readvariableop/savev2_adam_dense_87_bias_m_read_readvariableop?savev2_adam_lstm_122_lstm_cell_403_kernel_m_read_readvariableopIsavev2_adam_lstm_122_lstm_cell_403_recurrent_kernel_m_read_readvariableop=savev2_adam_lstm_122_lstm_cell_403_bias_m_read_readvariableop1savev2_adam_dense_87_kernel_v_read_readvariableop/savev2_adam_dense_87_bias_v_read_readvariableop?savev2_adam_lstm_122_lstm_cell_403_kernel_v_read_readvariableopIsavev2_adam_lstm_122_lstm_cell_403_recurrent_kernel_v_read_readvariableop=savev2_adam_lstm_122_lstm_cell_403_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :d::	�:	d�:�: : : : : : : : : : : :d::	�:	d�:�:d::	�:	d�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:!

_output_shapes	
:�:
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
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:!

_output_shapes	
:�:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:!

_output_shapes	
:�:

_output_shapes
: 
�
�
+__inference_dense_87_layer_call_fn_60324475

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�q
�
#__inference__wrapped_model_60322674
lstm_122_inputV
Csequential_78_lstm_122_lstm_cell_403_matmul_readvariableop_resource:	�X
Esequential_78_lstm_122_lstm_cell_403_matmul_1_readvariableop_resource:	d�S
Dsequential_78_lstm_122_lstm_cell_403_biasadd_readvariableop_resource:	�G
5sequential_78_dense_87_matmul_readvariableop_resource:dD
6sequential_78_dense_87_biasadd_readvariableop_resource:
identity��-sequential_78/dense_87/BiasAdd/ReadVariableOp�,sequential_78/dense_87/MatMul/ReadVariableOp�;sequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp�:sequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOp�<sequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp�sequential_78/lstm_122/whileZ
sequential_78/lstm_122/ShapeShapelstm_122_input*
T0*
_output_shapes
:t
*sequential_78/lstm_122/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_78/lstm_122/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_78/lstm_122/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_78/lstm_122/strided_sliceStridedSlice%sequential_78/lstm_122/Shape:output:03sequential_78/lstm_122/strided_slice/stack:output:05sequential_78/lstm_122/strided_slice/stack_1:output:05sequential_78/lstm_122/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_78/lstm_122/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
#sequential_78/lstm_122/zeros/packedPack-sequential_78/lstm_122/strided_slice:output:0.sequential_78/lstm_122/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_78/lstm_122/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_78/lstm_122/zerosFill,sequential_78/lstm_122/zeros/packed:output:0+sequential_78/lstm_122/zeros/Const:output:0*
T0*'
_output_shapes
:���������di
'sequential_78/lstm_122/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
%sequential_78/lstm_122/zeros_1/packedPack-sequential_78/lstm_122/strided_slice:output:00sequential_78/lstm_122/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_78/lstm_122/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_78/lstm_122/zeros_1Fill.sequential_78/lstm_122/zeros_1/packed:output:0-sequential_78/lstm_122/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dz
%sequential_78/lstm_122/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_78/lstm_122/transpose	Transposelstm_122_input.sequential_78/lstm_122/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_78/lstm_122/Shape_1Shape$sequential_78/lstm_122/transpose:y:0*
T0*
_output_shapes
:v
,sequential_78/lstm_122/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_78/lstm_122/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_78/lstm_122/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_78/lstm_122/strided_slice_1StridedSlice'sequential_78/lstm_122/Shape_1:output:05sequential_78/lstm_122/strided_slice_1/stack:output:07sequential_78/lstm_122/strided_slice_1/stack_1:output:07sequential_78/lstm_122/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_78/lstm_122/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_78/lstm_122/TensorArrayV2TensorListReserve;sequential_78/lstm_122/TensorArrayV2/element_shape:output:0/sequential_78/lstm_122/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_78/lstm_122/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_78/lstm_122/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_78/lstm_122/transpose:y:0Usequential_78/lstm_122/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_78/lstm_122/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_78/lstm_122/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_78/lstm_122/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_78/lstm_122/strided_slice_2StridedSlice$sequential_78/lstm_122/transpose:y:05sequential_78/lstm_122/strided_slice_2/stack:output:07sequential_78/lstm_122/strided_slice_2/stack_1:output:07sequential_78/lstm_122/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOpReadVariableOpCsequential_78_lstm_122_lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
+sequential_78/lstm_122/lstm_cell_403/MatMulMatMul/sequential_78/lstm_122/strided_slice_2:output:0Bsequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<sequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOpEsequential_78_lstm_122_lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
-sequential_78/lstm_122/lstm_cell_403/MatMul_1MatMul%sequential_78/lstm_122/zeros:output:0Dsequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential_78/lstm_122/lstm_cell_403/addAddV25sequential_78/lstm_122/lstm_cell_403/MatMul:product:07sequential_78/lstm_122/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
;sequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOpDsequential_78_lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,sequential_78/lstm_122/lstm_cell_403/BiasAddBiasAdd,sequential_78/lstm_122/lstm_cell_403/add:z:0Csequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
4sequential_78/lstm_122/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_78/lstm_122/lstm_cell_403/splitSplit=sequential_78/lstm_122/lstm_cell_403/split/split_dim:output:05sequential_78/lstm_122/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
,sequential_78/lstm_122/lstm_cell_403/SigmoidSigmoid3sequential_78/lstm_122/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
.sequential_78/lstm_122/lstm_cell_403/Sigmoid_1Sigmoid3sequential_78/lstm_122/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
(sequential_78/lstm_122/lstm_cell_403/mulMul2sequential_78/lstm_122/lstm_cell_403/Sigmoid_1:y:0'sequential_78/lstm_122/zeros_1:output:0*
T0*'
_output_shapes
:���������d�
)sequential_78/lstm_122/lstm_cell_403/ReluRelu3sequential_78/lstm_122/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
*sequential_78/lstm_122/lstm_cell_403/mul_1Mul0sequential_78/lstm_122/lstm_cell_403/Sigmoid:y:07sequential_78/lstm_122/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
*sequential_78/lstm_122/lstm_cell_403/add_1AddV2,sequential_78/lstm_122/lstm_cell_403/mul:z:0.sequential_78/lstm_122/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
.sequential_78/lstm_122/lstm_cell_403/Sigmoid_2Sigmoid3sequential_78/lstm_122/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������d�
+sequential_78/lstm_122/lstm_cell_403/Relu_1Relu.sequential_78/lstm_122/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
*sequential_78/lstm_122/lstm_cell_403/mul_2Mul2sequential_78/lstm_122/lstm_cell_403/Sigmoid_2:y:09sequential_78/lstm_122/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
4sequential_78/lstm_122/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   u
3sequential_78/lstm_122/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_78/lstm_122/TensorArrayV2_1TensorListReserve=sequential_78/lstm_122/TensorArrayV2_1/element_shape:output:0<sequential_78/lstm_122/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_78/lstm_122/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_78/lstm_122/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_78/lstm_122/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_78/lstm_122/whileWhile2sequential_78/lstm_122/while/loop_counter:output:08sequential_78/lstm_122/while/maximum_iterations:output:0$sequential_78/lstm_122/time:output:0/sequential_78/lstm_122/TensorArrayV2_1:handle:0%sequential_78/lstm_122/zeros:output:0'sequential_78/lstm_122/zeros_1:output:0/sequential_78/lstm_122/strided_slice_1:output:0Nsequential_78/lstm_122/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_78_lstm_122_lstm_cell_403_matmul_readvariableop_resourceEsequential_78_lstm_122_lstm_cell_403_matmul_1_readvariableop_resourceDsequential_78_lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_78_lstm_122_while_body_60322583*6
cond.R,
*sequential_78_lstm_122_while_cond_60322582*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
Gsequential_78/lstm_122/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
9sequential_78/lstm_122/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_78/lstm_122/while:output:3Psequential_78/lstm_122/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elements
,sequential_78/lstm_122/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_78/lstm_122/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_78/lstm_122/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_78/lstm_122/strided_slice_3StridedSliceBsequential_78/lstm_122/TensorArrayV2Stack/TensorListStack:tensor:05sequential_78/lstm_122/strided_slice_3/stack:output:07sequential_78/lstm_122/strided_slice_3/stack_1:output:07sequential_78/lstm_122/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask|
'sequential_78/lstm_122/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_78/lstm_122/transpose_1	TransposeBsequential_78/lstm_122/TensorArrayV2Stack/TensorListStack:tensor:00sequential_78/lstm_122/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dr
sequential_78/lstm_122/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
,sequential_78/dense_87/MatMul/ReadVariableOpReadVariableOp5sequential_78_dense_87_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
sequential_78/dense_87/MatMulMatMul/sequential_78/lstm_122/strided_slice_3:output:04sequential_78/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_78/dense_87/BiasAdd/ReadVariableOpReadVariableOp6sequential_78_dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_78/dense_87/BiasAddBiasAdd'sequential_78/dense_87/MatMul:product:05sequential_78/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_78/dense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_78/dense_87/BiasAdd/ReadVariableOp-^sequential_78/dense_87/MatMul/ReadVariableOp<^sequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp;^sequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOp=^sequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp^sequential_78/lstm_122/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2^
-sequential_78/dense_87/BiasAdd/ReadVariableOp-sequential_78/dense_87/BiasAdd/ReadVariableOp2\
,sequential_78/dense_87/MatMul/ReadVariableOp,sequential_78/dense_87/MatMul/ReadVariableOp2z
;sequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp;sequential_78/lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp2x
:sequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOp:sequential_78/lstm_122/lstm_cell_403/MatMul/ReadVariableOp2|
<sequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp<sequential_78/lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp2<
sequential_78/lstm_122/whilesequential_78/lstm_122/while:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�
�
while_cond_60323093
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60323093___redundant_placeholder06
2while_while_cond_60323093___redundant_placeholder16
2while_while_cond_60323093___redundant_placeholder26
2while_while_cond_60323093___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_60324236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_60323094
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_78_layer_call_fn_60323525

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*sequential_78_lstm_122_while_cond_60322582J
Fsequential_78_lstm_122_while_sequential_78_lstm_122_while_loop_counterP
Lsequential_78_lstm_122_while_sequential_78_lstm_122_while_maximum_iterations,
(sequential_78_lstm_122_while_placeholder.
*sequential_78_lstm_122_while_placeholder_1.
*sequential_78_lstm_122_while_placeholder_2.
*sequential_78_lstm_122_while_placeholder_3L
Hsequential_78_lstm_122_while_less_sequential_78_lstm_122_strided_slice_1d
`sequential_78_lstm_122_while_sequential_78_lstm_122_while_cond_60322582___redundant_placeholder0d
`sequential_78_lstm_122_while_sequential_78_lstm_122_while_cond_60322582___redundant_placeholder1d
`sequential_78_lstm_122_while_sequential_78_lstm_122_while_cond_60322582___redundant_placeholder2d
`sequential_78_lstm_122_while_sequential_78_lstm_122_while_cond_60322582___redundant_placeholder3)
%sequential_78_lstm_122_while_identity
�
!sequential_78/lstm_122/while/LessLess(sequential_78_lstm_122_while_placeholderHsequential_78_lstm_122_while_less_sequential_78_lstm_122_strided_slice_1*
T0*
_output_shapes
: y
%sequential_78/lstm_122/while/IdentityIdentity%sequential_78/lstm_122/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_78_lstm_122_while_identity.sequential_78/lstm_122/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�L
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324031
inputs_0?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60323946*
condR
while_cond_60323945*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_60324090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_60324090___redundant_placeholder06
2while_while_cond_60324090___redundant_placeholder16
2while_while_cond_60324090___redundant_placeholder26
2while_while_cond_60324090___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�	
�
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�K
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323179

inputs?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60323094*
condR
while_cond_60323093*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323487
lstm_122_input$
lstm_122_60323474:	�$
lstm_122_60323476:	d� 
lstm_122_60323478:	�#
dense_87_60323481:d
dense_87_60323483:
identity�� dense_87/StatefulPartitionedCall� lstm_122/StatefulPartitionedCall�
 lstm_122/StatefulPartitionedCallStatefulPartitionedCalllstm_122_inputlstm_122_60323474lstm_122_60323476lstm_122_60323478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323385�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)lstm_122/StatefulPartitionedCall:output:0dense_87_60323481dense_87_60323483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_122/StatefulPartitionedCall lstm_122/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�
�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324551

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�^
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323842

inputsH
5lstm_122_lstm_cell_403_matmul_readvariableop_resource:	�J
7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource:	d�E
6lstm_122_lstm_cell_403_biasadd_readvariableop_resource:	�9
'dense_87_matmul_readvariableop_resource:d6
(dense_87_biasadd_readvariableop_resource:
identity��dense_87/BiasAdd/ReadVariableOp�dense_87/MatMul/ReadVariableOp�-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp�,lstm_122/lstm_cell_403/MatMul/ReadVariableOp�.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp�lstm_122/whileD
lstm_122/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_122/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_122/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_122/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_sliceStridedSlicelstm_122/Shape:output:0%lstm_122/strided_slice/stack:output:0'lstm_122/strided_slice/stack_1:output:0'lstm_122/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_122/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_122/zeros/packedPacklstm_122/strided_slice:output:0 lstm_122/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_122/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_122/zerosFilllstm_122/zeros/packed:output:0lstm_122/zeros/Const:output:0*
T0*'
_output_shapes
:���������d[
lstm_122/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_122/zeros_1/packedPacklstm_122/strided_slice:output:0"lstm_122/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_122/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_122/zeros_1Fill lstm_122/zeros_1/packed:output:0lstm_122/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dl
lstm_122/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_122/transpose	Transposeinputs lstm_122/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_122/Shape_1Shapelstm_122/transpose:y:0*
T0*
_output_shapes
:h
lstm_122/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_122/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_1StridedSlicelstm_122/Shape_1:output:0'lstm_122/strided_slice_1/stack:output:0)lstm_122/strided_slice_1/stack_1:output:0)lstm_122/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_122/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_122/TensorArrayV2TensorListReserve-lstm_122/TensorArrayV2/element_shape:output:0!lstm_122/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_122/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_122/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_122/transpose:y:0Glstm_122/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_122/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_122/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_2StridedSlicelstm_122/transpose:y:0'lstm_122/strided_slice_2/stack:output:0)lstm_122/strided_slice_2/stack_1:output:0)lstm_122/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_122/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp5lstm_122_lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_122/lstm_cell_403/MatMulMatMul!lstm_122/strided_slice_2:output:04lstm_122/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_122/lstm_cell_403/MatMul_1MatMullstm_122/zeros:output:06lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_122/lstm_cell_403/addAddV2'lstm_122/lstm_cell_403/MatMul:product:0)lstm_122/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp6lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_122/lstm_cell_403/BiasAddBiasAddlstm_122/lstm_cell_403/add:z:05lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
&lstm_122/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/lstm_cell_403/splitSplit/lstm_122/lstm_cell_403/split/split_dim:output:0'lstm_122/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
lstm_122/lstm_cell_403/SigmoidSigmoid%lstm_122/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
 lstm_122/lstm_cell_403/Sigmoid_1Sigmoid%lstm_122/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mulMul$lstm_122/lstm_cell_403/Sigmoid_1:y:0lstm_122/zeros_1:output:0*
T0*'
_output_shapes
:���������d|
lstm_122/lstm_cell_403/ReluRelu%lstm_122/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mul_1Mul"lstm_122/lstm_cell_403/Sigmoid:y:0)lstm_122/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/add_1AddV2lstm_122/lstm_cell_403/mul:z:0 lstm_122/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_122/lstm_cell_403/Sigmoid_2Sigmoid%lstm_122/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dy
lstm_122/lstm_cell_403/Relu_1Relu lstm_122/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_122/lstm_cell_403/mul_2Mul$lstm_122/lstm_cell_403/Sigmoid_2:y:0+lstm_122/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dw
&lstm_122/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   g
%lstm_122/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_122/TensorArrayV2_1TensorListReserve/lstm_122/TensorArrayV2_1/element_shape:output:0.lstm_122/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_122/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_122/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_122/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_122/whileWhile$lstm_122/while/loop_counter:output:0*lstm_122/while/maximum_iterations:output:0lstm_122/time:output:0!lstm_122/TensorArrayV2_1:handle:0lstm_122/zeros:output:0lstm_122/zeros_1:output:0!lstm_122/strided_slice_1:output:0@lstm_122/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_122_lstm_cell_403_matmul_readvariableop_resource7lstm_122_lstm_cell_403_matmul_1_readvariableop_resource6lstm_122_lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_122_while_body_60323751*(
cond R
lstm_122_while_cond_60323750*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
9lstm_122/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
+lstm_122/TensorArrayV2Stack/TensorListStackTensorListStacklstm_122/while:output:3Blstm_122/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsq
lstm_122/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_122/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_122/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_122/strided_slice_3StridedSlice4lstm_122/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_122/strided_slice_3/stack:output:0)lstm_122/strided_slice_3/stack_1:output:0)lstm_122/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskn
lstm_122/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_122/transpose_1	Transpose4lstm_122/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_122/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd
lstm_122/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
dense_87/MatMulMatMul!lstm_122/strided_slice_3:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp.^lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp-^lstm_122/lstm_cell_403/MatMul/ReadVariableOp/^lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp^lstm_122/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2^
-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp-lstm_122/lstm_cell_403/BiasAdd/ReadVariableOp2\
,lstm_122/lstm_cell_403/MatMul/ReadVariableOp,lstm_122/lstm_cell_403/MatMul/ReadVariableOp2`
.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp.lstm_122/lstm_cell_403/MatMul_1/ReadVariableOp2 
lstm_122/whilelstm_122/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
*sequential_78_lstm_122_while_body_60322583J
Fsequential_78_lstm_122_while_sequential_78_lstm_122_while_loop_counterP
Lsequential_78_lstm_122_while_sequential_78_lstm_122_while_maximum_iterations,
(sequential_78_lstm_122_while_placeholder.
*sequential_78_lstm_122_while_placeholder_1.
*sequential_78_lstm_122_while_placeholder_2.
*sequential_78_lstm_122_while_placeholder_3I
Esequential_78_lstm_122_while_sequential_78_lstm_122_strided_slice_1_0�
�sequential_78_lstm_122_while_tensorarrayv2read_tensorlistgetitem_sequential_78_lstm_122_tensorarrayunstack_tensorlistfromtensor_0^
Ksequential_78_lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0:	�`
Msequential_78_lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�[
Lsequential_78_lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0:	�)
%sequential_78_lstm_122_while_identity+
'sequential_78_lstm_122_while_identity_1+
'sequential_78_lstm_122_while_identity_2+
'sequential_78_lstm_122_while_identity_3+
'sequential_78_lstm_122_while_identity_4+
'sequential_78_lstm_122_while_identity_5G
Csequential_78_lstm_122_while_sequential_78_lstm_122_strided_slice_1�
sequential_78_lstm_122_while_tensorarrayv2read_tensorlistgetitem_sequential_78_lstm_122_tensorarrayunstack_tensorlistfromtensor\
Isequential_78_lstm_122_while_lstm_cell_403_matmul_readvariableop_resource:	�^
Ksequential_78_lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource:	d�Y
Jsequential_78_lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource:	���Asequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp�@sequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp�Bsequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp�
Nsequential_78/lstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_78/lstm_122/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_78_lstm_122_while_tensorarrayv2read_tensorlistgetitem_sequential_78_lstm_122_tensorarrayunstack_tensorlistfromtensor_0(sequential_78_lstm_122_while_placeholderWsequential_78/lstm_122/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOpKsequential_78_lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
1sequential_78/lstm_122/while/lstm_cell_403/MatMulMatMulGsequential_78/lstm_122/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bsequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOpMsequential_78_lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
3sequential_78/lstm_122/while/lstm_cell_403/MatMul_1MatMul*sequential_78_lstm_122_while_placeholder_2Jsequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_78/lstm_122/while/lstm_cell_403/addAddV2;sequential_78/lstm_122/while/lstm_cell_403/MatMul:product:0=sequential_78/lstm_122/while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Asequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOpLsequential_78_lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
2sequential_78/lstm_122/while/lstm_cell_403/BiasAddBiasAdd2sequential_78/lstm_122/while/lstm_cell_403/add:z:0Isequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������|
:sequential_78/lstm_122/while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_78/lstm_122/while/lstm_cell_403/splitSplitCsequential_78/lstm_122/while/lstm_cell_403/split/split_dim:output:0;sequential_78/lstm_122/while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
2sequential_78/lstm_122/while/lstm_cell_403/SigmoidSigmoid9sequential_78/lstm_122/while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d�
4sequential_78/lstm_122/while/lstm_cell_403/Sigmoid_1Sigmoid9sequential_78/lstm_122/while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
.sequential_78/lstm_122/while/lstm_cell_403/mulMul8sequential_78/lstm_122/while/lstm_cell_403/Sigmoid_1:y:0*sequential_78_lstm_122_while_placeholder_3*
T0*'
_output_shapes
:���������d�
/sequential_78/lstm_122/while/lstm_cell_403/ReluRelu9sequential_78/lstm_122/while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
0sequential_78/lstm_122/while/lstm_cell_403/mul_1Mul6sequential_78/lstm_122/while/lstm_cell_403/Sigmoid:y:0=sequential_78/lstm_122/while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
0sequential_78/lstm_122/while/lstm_cell_403/add_1AddV22sequential_78/lstm_122/while/lstm_cell_403/mul:z:04sequential_78/lstm_122/while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d�
4sequential_78/lstm_122/while/lstm_cell_403/Sigmoid_2Sigmoid9sequential_78/lstm_122/while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������d�
1sequential_78/lstm_122/while/lstm_cell_403/Relu_1Relu4sequential_78/lstm_122/while/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
0sequential_78/lstm_122/while/lstm_cell_403/mul_2Mul8sequential_78/lstm_122/while/lstm_cell_403/Sigmoid_2:y:0?sequential_78/lstm_122/while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
Gsequential_78/lstm_122/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Asequential_78/lstm_122/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_78_lstm_122_while_placeholder_1Psequential_78/lstm_122/while/TensorArrayV2Write/TensorListSetItem/index:output:04sequential_78/lstm_122/while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_78/lstm_122/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_78/lstm_122/while/addAddV2(sequential_78_lstm_122_while_placeholder+sequential_78/lstm_122/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_78/lstm_122/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_78/lstm_122/while/add_1AddV2Fsequential_78_lstm_122_while_sequential_78_lstm_122_while_loop_counter-sequential_78/lstm_122/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_78/lstm_122/while/IdentityIdentity&sequential_78/lstm_122/while/add_1:z:0"^sequential_78/lstm_122/while/NoOp*
T0*
_output_shapes
: �
'sequential_78/lstm_122/while/Identity_1IdentityLsequential_78_lstm_122_while_sequential_78_lstm_122_while_maximum_iterations"^sequential_78/lstm_122/while/NoOp*
T0*
_output_shapes
: �
'sequential_78/lstm_122/while/Identity_2Identity$sequential_78/lstm_122/while/add:z:0"^sequential_78/lstm_122/while/NoOp*
T0*
_output_shapes
: �
'sequential_78/lstm_122/while/Identity_3IdentityQsequential_78/lstm_122/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_78/lstm_122/while/NoOp*
T0*
_output_shapes
: �
'sequential_78/lstm_122/while/Identity_4Identity4sequential_78/lstm_122/while/lstm_cell_403/mul_2:z:0"^sequential_78/lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
'sequential_78/lstm_122/while/Identity_5Identity4sequential_78/lstm_122/while/lstm_cell_403/add_1:z:0"^sequential_78/lstm_122/while/NoOp*
T0*'
_output_shapes
:���������d�
!sequential_78/lstm_122/while/NoOpNoOpB^sequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOpA^sequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOpC^sequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_78_lstm_122_while_identity.sequential_78/lstm_122/while/Identity:output:0"[
'sequential_78_lstm_122_while_identity_10sequential_78/lstm_122/while/Identity_1:output:0"[
'sequential_78_lstm_122_while_identity_20sequential_78/lstm_122/while/Identity_2:output:0"[
'sequential_78_lstm_122_while_identity_30sequential_78/lstm_122/while/Identity_3:output:0"[
'sequential_78_lstm_122_while_identity_40sequential_78/lstm_122/while/Identity_4:output:0"[
'sequential_78_lstm_122_while_identity_50sequential_78/lstm_122/while/Identity_5:output:0"�
Jsequential_78_lstm_122_while_lstm_cell_403_biasadd_readvariableop_resourceLsequential_78_lstm_122_while_lstm_cell_403_biasadd_readvariableop_resource_0"�
Ksequential_78_lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resourceMsequential_78_lstm_122_while_lstm_cell_403_matmul_1_readvariableop_resource_0"�
Isequential_78_lstm_122_while_lstm_cell_403_matmul_readvariableop_resourceKsequential_78_lstm_122_while_lstm_cell_403_matmul_readvariableop_resource_0"�
Csequential_78_lstm_122_while_sequential_78_lstm_122_strided_slice_1Esequential_78_lstm_122_while_sequential_78_lstm_122_strided_slice_1_0"�
sequential_78_lstm_122_while_tensorarrayv2read_tensorlistgetitem_sequential_78_lstm_122_tensorarrayunstack_tensorlistfromtensor�sequential_78_lstm_122_while_tensorarrayv2read_tensorlistgetitem_sequential_78_lstm_122_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2�
Asequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOpAsequential_78/lstm_122/while/lstm_cell_403/BiasAdd/ReadVariableOp2�
@sequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp@sequential_78/lstm_122/while/lstm_cell_403/MatMul/ReadVariableOp2�
Bsequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOpBsequential_78/lstm_122/while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�i
�
$__inference__traced_restore_60324772
file_prefix2
 assignvariableop_dense_87_kernel:d.
 assignvariableop_1_dense_87_bias:C
0assignvariableop_2_lstm_122_lstm_cell_403_kernel:	�M
:assignvariableop_3_lstm_122_lstm_cell_403_recurrent_kernel:	d�=
.assignvariableop_4_lstm_122_lstm_cell_403_bias:	�&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_2: %
assignvariableop_11_count_2: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: <
*assignvariableop_16_adam_dense_87_kernel_m:d6
(assignvariableop_17_adam_dense_87_bias_m:K
8assignvariableop_18_adam_lstm_122_lstm_cell_403_kernel_m:	�U
Bassignvariableop_19_adam_lstm_122_lstm_cell_403_recurrent_kernel_m:	d�E
6assignvariableop_20_adam_lstm_122_lstm_cell_403_bias_m:	�<
*assignvariableop_21_adam_dense_87_kernel_v:d6
(assignvariableop_22_adam_dense_87_bias_v:K
8assignvariableop_23_adam_lstm_122_lstm_cell_403_kernel_v:	�U
Bassignvariableop_24_adam_lstm_122_lstm_cell_403_recurrent_kernel_v:	d�E
6assignvariableop_25_adam_lstm_122_lstm_cell_403_bias_v:	�
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_87_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_87_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_lstm_122_lstm_cell_403_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_lstm_122_lstm_cell_403_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_122_lstm_cell_403_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_87_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_87_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_lstm_122_lstm_cell_403_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_lstm_122_lstm_cell_403_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_122_lstm_cell_403_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_87_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_87_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_lstm_122_lstm_cell_403_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpBassignvariableop_24_adam_lstm_122_lstm_cell_403_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_122_lstm_cell_403_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
0__inference_lstm_cell_403_layer_call_fn_60324502

inputs
states_0
states_1
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322741o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�L
�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324176
inputs_0?
,lstm_cell_403_matmul_readvariableop_resource:	�A
.lstm_cell_403_matmul_1_readvariableop_resource:	d�<
-lstm_cell_403_biasadd_readvariableop_resource:	�
identity��$lstm_cell_403/BiasAdd/ReadVariableOp�#lstm_cell_403/MatMul/ReadVariableOp�%lstm_cell_403/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
#lstm_cell_403/MatMul/ReadVariableOpReadVariableOp,lstm_cell_403_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_403/MatMulMatMulstrided_slice_2:output:0+lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_403_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_403/MatMul_1MatMulzeros:output:0-lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_403/addAddV2lstm_cell_403/MatMul:product:0 lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
$lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_403_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_403/BiasAddBiasAddlstm_cell_403/add:z:0,lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_403/splitSplit&lstm_cell_403/split/split_dim:output:0lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitp
lstm_cell_403/SigmoidSigmoidlstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_1Sigmoidlstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������dy
lstm_cell_403/mulMullstm_cell_403/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dj
lstm_cell_403/ReluRelulstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_1Mullstm_cell_403/Sigmoid:y:0 lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d~
lstm_cell_403/add_1AddV2lstm_cell_403/mul:z:0lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������dr
lstm_cell_403/Sigmoid_2Sigmoidlstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������dg
lstm_cell_403/Relu_1Relulstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_403/mul_2Mullstm_cell_403/Sigmoid_2:y:0"lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_403_matmul_readvariableop_resource.lstm_cell_403_matmul_1_readvariableop_resource-lstm_cell_403_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_60324091*
condR
while_cond_60324090*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp%^lstm_cell_403/BiasAdd/ReadVariableOp$^lstm_cell_403/MatMul/ReadVariableOp&^lstm_cell_403/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_403/BiasAdd/ReadVariableOp$lstm_cell_403/BiasAdd/ReadVariableOp2J
#lstm_cell_403/MatMul/ReadVariableOp#lstm_cell_403/MatMul/ReadVariableOp2N
%lstm_cell_403/MatMul_1/ReadVariableOp%lstm_cell_403/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�$
�
while_body_60322756
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_403_60322780_0:	�1
while_lstm_cell_403_60322782_0:	d�-
while_lstm_cell_403_60322784_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_403_60322780:	�/
while_lstm_cell_403_60322782:	d�+
while_lstm_cell_403_60322784:	���+while/lstm_cell_403/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_403/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_403_60322780_0while_lstm_cell_403_60322782_0while_lstm_cell_403_60322784_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60322741r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_403/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity4while/lstm_cell_403/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity4while/lstm_cell_403/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dz

while/NoOpNoOp,^while/lstm_cell_403/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_403_60322780while_lstm_cell_403_60322780_0">
while_lstm_cell_403_60322782while_lstm_cell_403_60322782_0">
while_lstm_cell_403_60322784while_lstm_cell_403_60322784_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2Z
+while/lstm_cell_403/StatefulPartitionedCall+while/lstm_cell_403/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323471
lstm_122_input$
lstm_122_60323458:	�$
lstm_122_60323460:	d� 
lstm_122_60323462:	�#
dense_87_60323465:d
dense_87_60323467:
identity�� dense_87/StatefulPartitionedCall� lstm_122/StatefulPartitionedCall�
 lstm_122/StatefulPartitionedCallStatefulPartitionedCalllstm_122_inputlstm_122_60323458lstm_122_60323460lstm_122_60323462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_122_layer_call_and_return_conditional_losses_60323179�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)lstm_122/StatefulPartitionedCall:output:0dense_87_60323465dense_87_60323467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_60323197x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_122/StatefulPartitionedCall lstm_122/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_122_input
�

�
lstm_122_while_cond_60323750.
*lstm_122_while_lstm_122_while_loop_counter4
0lstm_122_while_lstm_122_while_maximum_iterations
lstm_122_while_placeholder 
lstm_122_while_placeholder_1 
lstm_122_while_placeholder_2 
lstm_122_while_placeholder_30
,lstm_122_while_less_lstm_122_strided_slice_1H
Dlstm_122_while_lstm_122_while_cond_60323750___redundant_placeholder0H
Dlstm_122_while_lstm_122_while_cond_60323750___redundant_placeholder1H
Dlstm_122_while_lstm_122_while_cond_60323750___redundant_placeholder2H
Dlstm_122_while_lstm_122_while_cond_60323750___redundant_placeholder3
lstm_122_while_identity
�
lstm_122/while/LessLesslstm_122_while_placeholder,lstm_122_while_less_lstm_122_strided_slice_1*
T0*
_output_shapes
: ]
lstm_122/while/IdentityIdentitylstm_122/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_122_while_identity lstm_122/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_60323946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_60324381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_403_matmul_readvariableop_resource_0:	�I
6while_lstm_cell_403_matmul_1_readvariableop_resource_0:	d�D
5while_lstm_cell_403_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_403_matmul_readvariableop_resource:	�G
4while_lstm_cell_403_matmul_1_readvariableop_resource:	d�B
3while_lstm_cell_403_biasadd_readvariableop_resource:	���*while/lstm_cell_403/BiasAdd/ReadVariableOp�)while/lstm_cell_403/MatMul/ReadVariableOp�+while/lstm_cell_403/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_403/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_403_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_403/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+while/lstm_cell_403/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_403_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_403/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_403/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_403/addAddV2$while/lstm_cell_403/MatMul:product:0&while/lstm_cell_403/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_403/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_403_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_403/BiasAddBiasAddwhile/lstm_cell_403/add:z:02while/lstm_cell_403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#while/lstm_cell_403/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_403/splitSplit,while/lstm_cell_403/split/split_dim:output:0$while/lstm_cell_403/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split|
while/lstm_cell_403/SigmoidSigmoid"while/lstm_cell_403/split:output:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_1Sigmoid"while/lstm_cell_403/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mulMul!while/lstm_cell_403/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dv
while/lstm_cell_403/ReluRelu"while/lstm_cell_403/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_1Mulwhile/lstm_cell_403/Sigmoid:y:0&while/lstm_cell_403/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/add_1AddV2while/lstm_cell_403/mul:z:0while/lstm_cell_403/mul_1:z:0*
T0*'
_output_shapes
:���������d~
while/lstm_cell_403/Sigmoid_2Sigmoid"while/lstm_cell_403/split:output:3*
T0*'
_output_shapes
:���������ds
while/lstm_cell_403/Relu_1Reluwhile/lstm_cell_403/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_403/mul_2Mul!while/lstm_cell_403/Sigmoid_2:y:0(while/lstm_cell_403/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_403/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_403/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dz
while/Identity_5Identitywhile/lstm_cell_403/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp+^while/lstm_cell_403/BiasAdd/ReadVariableOp*^while/lstm_cell_403/MatMul/ReadVariableOp,^while/lstm_cell_403/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_403_biasadd_readvariableop_resource5while_lstm_cell_403_biasadd_readvariableop_resource_0"n
4while_lstm_cell_403_matmul_1_readvariableop_resource6while_lstm_cell_403_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_403_matmul_readvariableop_resource4while_lstm_cell_403_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_403/BiasAdd/ReadVariableOp*while/lstm_cell_403/BiasAdd/ReadVariableOp2V
)while/lstm_cell_403/MatMul/ReadVariableOp)while/lstm_cell_403/MatMul/ReadVariableOp2Z
+while/lstm_cell_403/MatMul_1/ReadVariableOp+while/lstm_cell_403/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: "�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
lstm_122_input;
 serving_default_lstm_122_input:0���������<
dense_870
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
%trace_0
&trace_1
'trace_2
(trace_32�
0__inference_sequential_78_layer_call_fn_60323217
0__inference_sequential_78_layer_call_fn_60323525
0__inference_sequential_78_layer_call_fn_60323540
0__inference_sequential_78_layer_call_fn_60323455�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z%trace_0z&trace_1z'trace_2z(trace_3
�
)trace_0
*trace_1
+trace_2
,trace_32�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323691
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323842
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323471
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323487�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z)trace_0z*trace_1z+trace_2z,trace_3
�B�
#__inference__wrapped_model_60322674lstm_122_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemimjmkmlmmvnvovpvqvr"
	optimizer
,
2serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

3states
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
9trace_0
:trace_1
;trace_2
<trace_32�
+__inference_lstm_122_layer_call_fn_60323853
+__inference_lstm_122_layer_call_fn_60323864
+__inference_lstm_122_layer_call_fn_60323875
+__inference_lstm_122_layer_call_fn_60323886�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z9trace_0z:trace_1z;trace_2z<trace_3
�
=trace_0
>trace_1
?trace_2
@trace_32�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324031
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324176
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324321
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324466�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z=trace_0z>trace_1z?trace_2z@trace_3
"
_generic_user_object
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator
H
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
+__inference_dense_87_layer_call_fn_60324475�
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
 zNtrace_0
�
Otrace_02�
F__inference_dense_87_layer_call_and_return_conditional_losses_60324485�
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
 zOtrace_0
!:d2dense_87/kernel
:2dense_87/bias
0:.	�2lstm_122/lstm_cell_403/kernel
::8	d�2'lstm_122/lstm_cell_403/recurrent_kernel
*:(�2lstm_122/lstm_cell_403/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
P0
Q1
R2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_sequential_78_layer_call_fn_60323217lstm_122_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_78_layer_call_fn_60323525inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_78_layer_call_fn_60323540inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_78_layer_call_fn_60323455lstm_122_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323691inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323842inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323471lstm_122_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323487lstm_122_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
&__inference_signature_wrapper_60323510lstm_122_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_lstm_122_layer_call_fn_60323853inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_122_layer_call_fn_60323864inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_122_layer_call_fn_60323875inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_122_layer_call_fn_60323886inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324031inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324176inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324321inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324466inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_0
Ytrace_12�
0__inference_lstm_cell_403_layer_call_fn_60324502
0__inference_lstm_cell_403_layer_call_fn_60324519�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0zYtrace_1
�
Ztrace_0
[trace_12�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324551
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324583�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0z[trace_1
"
_generic_user_object
 "
trackable_list_wrapper
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
�B�
+__inference_dense_87_layer_call_fn_60324475inputs"�
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
�B�
F__inference_dense_87_layer_call_and_return_conditional_losses_60324485inputs"�
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
N
\	variables
]	keras_api
	^total
	_count"
_tf_keras_metric
N
`	variables
a	keras_api
	btotal
	ccount"
_tf_keras_metric
^
d	variables
e	keras_api
	ftotal
	gcount
h
_fn_kwargs"
_tf_keras_metric
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
�B�
0__inference_lstm_cell_403_layer_call_fn_60324502inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_lstm_cell_403_layer_call_fn_60324519inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324551inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324583inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
^0
_1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$d2Adam/dense_87/kernel/m
 :2Adam/dense_87/bias/m
5:3	�2$Adam/lstm_122/lstm_cell_403/kernel/m
?:=	d�2.Adam/lstm_122/lstm_cell_403/recurrent_kernel/m
/:-�2"Adam/lstm_122/lstm_cell_403/bias/m
&:$d2Adam/dense_87/kernel/v
 :2Adam/dense_87/bias/v
5:3	�2$Adam/lstm_122/lstm_cell_403/kernel/v
?:=	d�2.Adam/lstm_122/lstm_cell_403/recurrent_kernel/v
/:-�2"Adam/lstm_122/lstm_cell_403/bias/v�
#__inference__wrapped_model_60322674y;�8
1�.
,�)
lstm_122_input���������
� "3�0
.
dense_87"�
dense_87����������
F__inference_dense_87_layer_call_and_return_conditional_losses_60324485\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� ~
+__inference_dense_87_layer_call_fn_60324475O/�,
%�"
 �
inputs���������d
� "�����������
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324031}O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������d
� �
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324176}O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������d
� �
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324321m?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������d
� �
F__inference_lstm_122_layer_call_and_return_conditional_losses_60324466m?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������d
� �
+__inference_lstm_122_layer_call_fn_60323853pO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������d�
+__inference_lstm_122_layer_call_fn_60323864pO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������d�
+__inference_lstm_122_layer_call_fn_60323875`?�<
5�2
$�!
inputs���������

 
p 

 
� "����������d�
+__inference_lstm_122_layer_call_fn_60323886`?�<
5�2
$�!
inputs���������

 
p

 
� "����������d�
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324551���}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p 
� "s�p
i�f
�
0/0���������d
E�B
�
0/1/0���������d
�
0/1/1���������d
� �
K__inference_lstm_cell_403_layer_call_and_return_conditional_losses_60324583���}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p
� "s�p
i�f
�
0/0���������d
E�B
�
0/1/0���������d
�
0/1/1���������d
� �
0__inference_lstm_cell_403_layer_call_fn_60324502���}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p 
� "c�`
�
0���������d
A�>
�
1/0���������d
�
1/1���������d�
0__inference_lstm_cell_403_layer_call_fn_60324519���}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p
� "c�`
�
0���������d
A�>
�
1/0���������d
�
1/1���������d�
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323471sC�@
9�6
,�)
lstm_122_input���������
p 

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323487sC�@
9�6
,�)
lstm_122_input���������
p

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323691k;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_60323842k;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
0__inference_sequential_78_layer_call_fn_60323217fC�@
9�6
,�)
lstm_122_input���������
p 

 
� "�����������
0__inference_sequential_78_layer_call_fn_60323455fC�@
9�6
,�)
lstm_122_input���������
p

 
� "�����������
0__inference_sequential_78_layer_call_fn_60323525^;�8
1�.
$�!
inputs���������
p 

 
� "�����������
0__inference_sequential_78_layer_call_fn_60323540^;�8
1�.
$�!
inputs���������
p

 
� "�����������
&__inference_signature_wrapper_60323510�M�J
� 
C�@
>
lstm_122_input,�)
lstm_122_input���������"3�0
.
dense_87"�
dense_87���������