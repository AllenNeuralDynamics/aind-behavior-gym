.. Doc Template documentation master file, created by
   sphinx-quickstart on Wed Aug 17 15:36:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to aind-behavior-gym documentation!
===========================================

Implementing AIND behavior *tasks* using `gymnasium <https://gymnasium.farama.org/index.html>`_ and setting up a base class for *agents* that perform the tasks.

What is this library for?
--------------------------

It aims to provide a common language among behavior-related animal training, artificial agent training, model fitting, and model simulation in AIND.

See related repositories:

- `aind-dynamic-foraging-models <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models>`_

.. image:: https://github.com/user-attachments/assets/8a5bfc68-e592-4dee-8fc9-b00b2e0a39ab
   :width: 1668px
   :alt: Example Image

Structure
---------

For now, three dynamic foraging tasks have been implemented.

- :ref:`Coupled block task <coupled_block_task>`

- :ref:`Uncoupled block task <uncoupled_block_task>`

- :ref:`Random walk task <random_walk_task>`

|classes_aind-behavior-gym|

To develop more dynamic foraging tasks, please subclass :ref:`DynamicForagingTaskBase <foraging_task_base>`.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   aind_behavior_gym.dynamic_foraging.task


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
