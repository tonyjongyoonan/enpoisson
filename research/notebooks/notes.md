# Short Term Experiments
 - set up GPUs and test DenseNet vs. Maia (Deep-Residual-CNN) vs. Multimodal (CNN + RNN)
 - how much of past sequence matters? (currently doing 16 half-moves)
-  12 channels vs. 6 channels

# Architectural Experiments
- ELO as a parameter 
- Transformers with much more data 
- fine-tuning model on someone's 10 games (effective?) -> to somehow provide context to our model when doing a prediction for a user? -> SAM?

# Things to Do
- After deciding on architecture: train on full dataset on GPUs, LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy), 