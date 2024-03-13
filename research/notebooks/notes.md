# TODO
- test cnn + transformers
- test UCI vs. SAN again (on local machine)
- make another instance with more data -> create memmaps with special tokens -> create memmaps of other months
    - train new multimodal model (on 6 months + special tokens + potentially transformers)
- experiment with cross attention
    - [0] working on right now -> cross attention in which seq are queries, and cnn is key + value
    - full transformer cross-attention
    - also experiment with replacing RNN with transformer (but give it more history moves)
    - [1] let's go here right away -> self attention transformer on cnn (including past 4 moves), do positional encoding on each move
- scaling 2.0 
    - find best CNN model as large as I can go (then train and transfer)
    - find best RNN/transformer model (then train and transfer)
    - transfer and fine-tune multimodal
- do transfer + finetuning on RNN (token prediction)
- combine files
    - https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
- create dataset of 50 positions (tactical plays: check, taking, forks | positional plays: taking the center) -- moves that are very obvious and good, (maybe not perfect), to a decent human player
First Item on Later
- do pretraining (on all ELOs) + finetuning (on ELO)


## Long-Term/Architectural Experiments
- Transfer + Fine-tuning vs. ELO as a parameter 
- fine-tuning model on someone's 10 games (effective?) -> to somehow provide context to our model when doing a prediction for a user? -> SAM?
- RLHF (using RL paradigms to diversify dataset... selfplay???)
- MoE
- ViT (it kinda works but found research paper that tells us it's not needed. ResNet is enough)
- Transformers with much more data 

## Things to Do At The End
- finalize multimodal architecture
- train on full dataset on GPUs, do hyperparameter tuning (batch size, learning rate, weight decay), LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy) 

# Notes 

## Note: 2/5/24 11:30PM
Things to do before the next presentation (15 hours)
- Test Extra Input Channels (5 hours)
- GradCAM (1 hour)
- Test attention mechanism (3 hour)
- Scale everything with GPUs (1 hour)
- Error Analysis (5 hours)

## Note: 3/12
Discarded Experiments
- experiment: [SEP] tokens
- add more channels (@nate)
- experiment with tony's games
- play around with RL paradigms to explore unseen board positions
- test global averaging + more channels vs. flattening (for the input into the fc)