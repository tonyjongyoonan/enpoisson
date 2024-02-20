# Short Term Experiments
-  12 channels vs. 6 channels [DONE it's much better]
- Channels representing queen side and king side castling + many other stuff
- add [CLS] and [SEP] to sequence of moves 

# Long-Term/Architectural Experiments
- Transfer + Fine-tuning vs. ELO as a parameter 
- MoE
- ViT (it kinda works but found research paper that tells us it's not needed. ResNet is enough)
- Transformers with much more data 
- fine-tuning model on someone's 10 games (effective?) -> to somehow provide context to our model when doing a prediction for a user? -> SAM?

# Things to Do
- After deciding on architecture: train on full dataset on GPUs, LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy), 

# Note: 2/5/24 11:30PM
Things to do before the next presentation (15 hours)
- Test Extra Input Channels (5 hours)
- GradCAM (1 hour)
- Test attention mechanism (3 hour)
- Scale everything with GPUs (1 hour)
- Error Analysis (5 hours)