# MARL_Environment_Chase

Code to train and evaluate the MARL environment

0. Verified Operational System: 
- Windows 10
- Linux: Ubuntu 20/22
- Note: Agents were trained using ray==2.2, which has compatibility issues on macOS, particularly on Apple Silicon (M1/M2 chips). For more details and potential workarounds, please visit [Ray documentation](https://docs.ray.io/en/releases-2.2.0/ray-overview/installation.html#m1-mac-apple-silicon-support)

1. Create the environment and install dependencies:
- install Anaconda/Miniconda
- create a conda environment with: `conda create --name marl python==3.8` 
- activate conda environment with: `conda activate marl`
- install the GPU-supported version of PyTorch by visiting the official PyTorch website: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/). Make sure to select the appropriate version based on your system configuration (CUDA version, Python version, and package manager).
- install all dependencies with: `pip install -r requirements.txt`

2. Train chaser and explorer agents in the environment:  

   To start training:  
      `python MultiAgent_Train_Final.py --dir-out Path\To\Output\ --arena-file ArenaOfSelection(see Arena below)`

      Example: `python MultiAgent_Train_Final.py --dir-out .\ --arena-file MultiAgentArena_v1d_5 --train-iter 200`

   - Specify additional arguments as needed.
   - For a full list of modifiable parameters, refer to MultiAgent_Train_Final.py (arguments are handled via argparse).

   - Arena:
     - Social: MultiAgentArena_v1d_5
     - Mutual Interaction: MultiAgentArena_v1d_27
     - Non-social : MultiAgentArena_v1d_11
   - Callback: callback_v1j.py to track various behavioral metrics
   - Neural network: simple_rnn_v2_3_2.py as a simple implementation of vanilla RNN

3. Evaluate agents:
   
   To evaluate agents:  
      `python MultiAgent_Evaluate_Final.py --dir-in Path\To\Checkpoint\Output --dir-out Path\To\Output\ --arena-file ArenaOfSelection(see Arena below) --file-suffix test`

      Example: `python MultiAgent_Evaluate_Final.py --dir-in .\PPO_2025-02-18_16-43-52\PPO_MultiAgentArena_v1d_5_97666_00000_0_2025-02-18_16-43-52 --dir-out .\PPO_2025-02-18_16-43-52\out --arena-file MultiAgentArena_v1d_5 --file-suffix test --num-checkpoints 1 --num-episode 10`

   - Specify the paths to checkpoints and output and correct number of checkpoints available.
   - Specify additional arguments as needed.
   - For a full list of modifiable parameters, refer to MultiAgent_Evaluate_Final.py (arguments are handled via argparse). 
     
   - Arena for evaluation ():
     - Evaluating in playoffs:
       - Social: MultiAgentArena_v1d_5
       - Mutual Interaction: MultiAgentArena_v1d_27
       - Non-social:
         - MultiAgentArena_v1d_11
         - MultiAgentArena_v1d_11_2: Partial vision of Partner
         - MultiAgentArena_v1d_11_3: Full vision of Partner
     - Evaluating against standard agent (that randomly samples action from a uniform distribution)
       - Social: MultiAgentArena_v1d_5_RandA1 /MultiAgentArena_v1d_5_RandA2
       - Mutual Interaction: MultiAgentArena_v1d_27_RandA1 /MultiAgentArena_v1d_27_RandA2
       - Non-social: MultiAgentArena_v1d_11_RandA1 /MultiAgentArena_v1d_11_RandA2
