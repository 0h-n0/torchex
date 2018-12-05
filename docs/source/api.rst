torchex API
===========

Initialization
**************
.. automodule:: torchex.nn.init
                
.. autofunction:: chrono_init

.. autofunction:: feedforward_init

.. autofunction:: rnn_init                                    


Utility
********

.. automodule:: torchex.nn
               
.. autoclass:: PeriodicPad2d
      
.. autoclass:: PeriodicPad3d

.. autoclass:: Flatten               



Convlution
************

.. autoclass:: MLPConv2d

.. autoclass:: UpsampleConvLayer


Pooling
**************

.. autoclass:: GlobalAvgPool1d
               
.. autoclass:: GlobalAvgPool2d
               
.. autoclass:: GlobalMaxPool1d
               
.. autoclass:: GlobalMaxPool2d                              

.. autoclass:: MaxAvgPool2d
               
Cropping
********

.. autoclass:: Crop2d

.. autoclass:: Crop3d

Local Convolution
******************

.. autoclass:: Conv2dLocal               


Highway
**********

.. autoclass:: Highway



Inception
**********

.. autoclass:: Inception

.. autoclass:: InceptionBN

               
Graph
*******
   
.. autoclass:: GraphLinear               

.. autoclass:: GraphConv               

.. autoclass:: SparseMM

.. autoclass:: GraphBatchNorm

.. autoclass:: SequenceLinear


indrnn
*******

.. autoclass:: IndRNN

.. autoclass:: IndRNNCell

.. autoclass:: IndRNNTanhCell

.. autoclass:: IndRNNReLuCell


Lazy Modules
************

.. autoclass:: Linear

.. autoclass:: Conv1d

.. autoclass:: Conv2d                              

.. autoclass:: Conv3d

.. autoclass:: LazyRNNBase                                             


