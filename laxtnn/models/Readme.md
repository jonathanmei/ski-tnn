This folder has several models. In the top level,
1. language_model.py has several causal models. In that file, when the decorator @register_model_architecture shows up, its second argument can be used in laxtnn/scripts/train_alm.sh and train_blm.sh to use the corresponding model. Simply change arch.
2. bidirectional_lm.py has bidirectional models. One can use the 2nd argument seen in @register_model_architecture the same way.
