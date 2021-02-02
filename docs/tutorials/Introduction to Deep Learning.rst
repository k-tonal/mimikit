Introduction to Deep Learning
=============================

In this guide we will briefly cover the most essential & practical aspects of working with neural networks.
If you are interested in the inner-working of neural networks, we recommend you to take a course or follow one of the dedicated tutorials we listed at the bottom of this guide.

How Does Deep-Learning Work?
----------------------------

Inputs, Parameters & Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the most general standpoint, all neural networks are computer programs that perform a pre-defined mathematical function on some input data.

For the network to perform this mathematical function, it needs parameters : arrays of numbers that are multiplied and/or summed with the input data
and that, accordingly, determine the outcome of the function for a given input.

As a quick example, you could define a neural network to take one number as input, to have one number as parameter and to output one number, if you were to define the "network" with the following function :

.. code-block::

    y = a * x

where ``y`` is the output, ``x`` is the input and ``a`` is the parameter.

Once you work with "real-life" neural networks, you'll see that the function can take many forms and that there is almost always many more parameters.
Nevertheless, ``y = a * x`` is a valid definition for a neural network.

Since the input is the data you select and the outputs are the result of the mathematical function, one could say : **neural networks are defined by their parameters**.
This is the parameters (and how they are used by in the mathematical function) that makes up the behaviour of the net,
i.e. whether it succeeds or fails to recognize cats, generate music or images and so on.


Training & Inference : Lifecycle of a Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

