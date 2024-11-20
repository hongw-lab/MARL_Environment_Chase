# MARL_Environment_Chase

Code to train and evaluate the MARL environment

0. Create the environment and install dependencies:

- create a conda environment with: `conda create --name marl`
- activate conda environment with: `conda activate marl`
- install all dependencies with: `pip install -r requirements.txt`

1. Train chaser and explorer agents in the environment:  
   To start training:  
    `python MultiAgent_Train_Final.py --dir-out Path\To\Output\ --arena-file ArenaOfSelection(see Arena below)`

   Essential files for training:

   - Arena:
     - Social: MultiAgentArena_v1d_5.py
     - Mutual Interaction: MultiAgentArena_v1d_27.py
     - Non-social : MultiAgentArena_v1d_11.py
   - Callback: callback_v1j.py to track various behavioral metrics
   - Neural network: simple_rnn_v2_3_2.py as a simple implementation of vanilla RNN

2. Evaluate agents:  
   To evaluate agents:  
   `python MultiAgent_Evaluate_Final.py --dir-in Path\To\Checkpoint\Output --dir-out Path\To\Output\ --arena-file ArenaOfSelection(see Arena below) --file-suffix test`  
   <ins>Select Arena from below for desired agent evaluation.</ins>
   - Arena ():
     - Evaluating in playoffs:
       - Social: MultiAgentArena_v1d_5.py
       - Mutual Interaction: MultiAgentArena_v1d_27.py
       - Non-social:
         - MultiAgentArena_v1d_11.py
         - MultiAgentArena_v1d_11_2.py: Partial vision of Partner
         - MultiAgentArena_v1d_11_3.py: Full vision of Partner
     - Evaluating against standard agent (that randomly samples action from a uniform distribution)
       - Social: MultiAgentArena_v1d_5_RandA1.py/MultiAgentArena_v1d_5_RandA2.py
       - Mutual Interaction: MultiAgentArena_v1d_27_RandA1.py/MultiAgentArena_v1d_27_RandA2.py
       - Non-social: MultiAgentArena_v1d_11_RandA1.py/MultiAgentArena_v1d_11_RandA2.py
