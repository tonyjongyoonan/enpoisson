# TODO
- add more channels (@nate)
- do transfer + finetuning on RNN (token prediction) + CNN (token shape?)
- combine files
    - https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
- experiment: add [CLS] and [SEP] tokens
- create dataset of 50 positions (tactical plays: check, taking, forks | positional plays: taking the center) -- moves that are very obvious and good, (maybe not perfect), to a decent human player
- experiment with tony's games
- experiment with cross attention
First Item on Later
- do transfer + finetuning (on ELO)


## Long-Term/Architectural Experiments
- Transfer + Fine-tuning vs. ELO as a parameter 
- fine-tuning model on someone's 10 games (effective?) -> to somehow provide context to our model when doing a prediction for a user? -> SAM?
- RLHF
- MoE
- ViT (it kinda works but found research paper that tells us it's not needed. ResNet is enough)
- Transformers with much more data 

## Things to Do At The End
- finalize multimodal architecture
- train on full dataset on GPUs, LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy) 

# Notes 

## Note: 2/5/24 11:30PM
Things to do before the next presentation (15 hours)
- Test Extra Input Channels (5 hours)
- GradCAM (1 hour)
- Test attention mechanism (3 hour)
- Scale everything with GPUs (1 hour)
- Error Analysis (5 hours)