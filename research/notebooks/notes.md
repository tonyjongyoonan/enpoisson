# TODO
- create pipeline from PGN with next move -> model input (12,8,8) tensor -> adjusted PGN with predicted move
- create dataset of 50 positions (tactical plays: check, taking, forks | positional plays: taking the center) -- moves that are very obvious and good, (maybe not perfect), to a decent human player
- error analysis and gradCAM analysis of those moves
AFTER DEMO:
- add more channels (@nate)
- experiment: add [CLS] and [SEP] tokens
- experiment with tony's games
- experiment with cross attention
- finalize Multimodal architecture
LATER
- do transfer + finetuning on RNN (token prediction) + CNN (token shape)
- do transfer + finetuning (on ELO)
- MoE???


## Long-Term/Architectural Experiments
- Transfer + Fine-tuning vs. ELO as a parameter 
- MoE
- ViT (it kinda works but found research paper that tells us it's not needed. ResNet is enough)
- Transformers with much more data 
- fine-tuning model on someone's 10 games (effective?) -> to somehow provide context to our model when doing a prediction for a user? -> SAM?

## Things to Do At The End
- After finalizing on architecture: train on full dataset on GPUs, LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy), 

# Notes 

## Note: 2/5/24 11:30PM
Things to do before the next presentation (15 hours)
- Test Extra Input Channels (5 hours)
- GradCAM (1 hour)
- Test attention mechanism (3 hour)
- Scale everything with GPUs (1 hour)
- Error Analysis (5 hours)