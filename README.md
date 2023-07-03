# iia-sf3-machine-learning

In this project, an inverted pendulum system receiving a software simulator of a cart with a pendulum attached is written in Python.
 
The goal will be to learn a controller that balances the pendulum in a data-driven way. Initially, I learn how to operate the simulator and explore the different types of behaviour that the system can exhibit. Next, I collect training data from the simulator and use this to train non-linear regression models, including linear regression with non-linear basis functions. The trained models will be assessed on test data from the simulator. Once accurate models are learned these will be used to learn controllers that can balance the pendulum in the upright position and keep it there. Finally, the controllers and the models will will be stress tested in various ways to test their robustness. 
