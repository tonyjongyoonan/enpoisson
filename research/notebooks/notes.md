# TODO
- test UCI vs. SAN again (on local machine, using haha_longer.csv and doing df_data twice) @ 3/13
- create memmaps with special tokens (need to repeat everything breh... also add channels + include FEN to allow cross attention jawns) @ 3/16
    - repeat for other months
    - train new multimodal model (on 6 months + special tokens + potentially transformers)
    - train with batch size of 8
    - combine files
        -   https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
- experiment with cross attention @ 3/17
    - [0] start with CNN encodings and just one big MLP
    - [1] self attention transformer on cnn (including past 8 moves), do positional encoding on each move
- scaling 2.0 
    - find best CNN model as large as I can go (then train and transfer)
    - find best RNN/transformer model (then train and transfer)
    - transfer and fine-tune multimodal
- are we harmed? by looking at only white board positions? (well we're always only looking at white board positions... so uh this might be a problem)
- do transfer + finetuning on RNN (token prediction)
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
- including future positions into the data??? when humans play... there is a clear sense of the past... but more importantly there is a greater sense of the future... and they all fall into culmination into the present
    - whenever I play there a few pruned trees that I keep in my brain and that I select from.... should the fundamental decision driver of a chess player be a tree? instead of a neural network?
- alphazero teaches us that neural networks are capable of thinking like a human that deepl understands the mechanics of a chess and its developed its own personal style that shows depth, character, and intelligence 
    - based on our goal that teaching requires simulating this kind of depth and play of chess that's human but for a lower caliber (this isn't possible with straight up alpha zero... altho perhaps you can just cut it short until it reaches a particular elo -> as we do alphazero does it do better on elo1100 prediction or worse? what does the graph look like)... if this doesn't work... then there are two fundamental questions/goals here: are neural networks capable of playing like an exact person of a particular elo? to pickup on their particularities and playing styles? 2) if we take this neural network that plays exactly like a person... what does it mean to teach them better chess? will it develop like the player?
    - people play games with two threads... an attempt to brute-force calculations but also to have a long term strategy... is chess more than pattern recognition? yes.. CNN captures patterns.. RNN captures long term strategy... and CNN transformer perhaps captures both of those working together... but how does it capture raw calculations? precision... how do we, one, give it the future and ,two, give it an ability to work precisely. 
    - I want to teach it a way of progressing trees... a nueral model that's mapped from one board state to another... well it's doing that already.... ooo maybe contrastive learning? it needs to know what smaller pool it's choosing from and what it's saying no too... whenever I play a move I'm choosing from some candidate set..
    - use our model to do MCTS w/ a depth of 4 -- explore if there's any rewards with stockfish at the 3 most likely options (time control is associated with amount of threads in our head) -> we can experiment and see how well these various methods to against different time controls
    - plan: after we get multimodal 2.0 model -> let's implement MCTS w/ a depth of 4 (60% accuracy)
    - the goal: to replicate human gameplay
- RL!!!!!!!!!!!!!! ahhhhhh there's so much potential here
- hm how diverse is our data? how can we measure that?
- OHHHHHHHH transformer with FEN???? why don't they include past FENs? 
    - gg lol (https://lczero.org/blog/2024/02/transformer-progress/)
        - https://arxiv.org/abs/1701.06538
    - then grid search over MCTS possibilities
- how do systems like maia evaluate on GMs? 

## Things to Do At The End
- finalize multimodal architecture
- train on full dataset on GPUs, do hyperparameter tuning (batch size, learning rate, weight decay), LR warm up, SWA after 25+ epochs (garuntees +1 test accuracy) 
- replicate everything for each ELO range

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
- cross attention in which seq are queries, and cnn is key + value