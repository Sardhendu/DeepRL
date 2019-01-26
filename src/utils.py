
def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter

    Idea
    ======
        Instead of performing a hard update on Fixed-Q-targets after say 10000 timesteps, perform soft-update
        (move slightly) every time-step
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def hard_update(local_model, target_model):
    """Hard update model parameters.
    θ_target = θ_local

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to

    Idea
    ======
        After t time-step copy the weights of local network to target network
    """
    # print('[Hard Update] Performuing hard update .... ')
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)