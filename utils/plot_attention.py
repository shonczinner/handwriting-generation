import matplotlib.pyplot as plt
import numpy as np

def plot_attention(text, phi, save_path=None):
    """
    Plots an attention map with the input text along the y-axis.

    Args:
        text (str): The input string of length U.
        phi (Tensor or ndarray): Attention tensor of shape [T, U], 
                                 where T is the number of output time steps and 
                                 U is the number of input characters.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    # Convert to numpy if it's a torch tensor
    if hasattr(phi, 'detach'):
        phi = phi.detach().cpu().numpy()

    # Transpose to make Y-axis = characters (U), X-axis = time steps (T)
    attention = phi.T

    # Plot
    fig, ax = plt.subplots(figsize=(10, len(text) * 0.4))
    im = ax.imshow(attention, aspect='auto', origin='lower', interpolation='none', cmap='viridis')

    # Set the y-axis with character labels
    ax.set_yticks(np.arange(len(text)))
    ax.set_yticklabels(list(text))

    ax.set_xlabel('Time step (t)')
    #ax.set_ylabel('Input character (u)')
    ax.set_title('Attention Map')

    # Add colorbar
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()

    plt.close()

if __name__ == "__main__":
    # Example usage
    example_text = "attention"
    U = len(example_text)
    T = 20  # Number of time steps

    # Create dummy attention weights (normalized along U axis)
    np.random.seed(0)
    dummy_phi = np.random.rand(1, T, U)
    dummy_phi /= dummy_phi.sum(axis=2, keepdims=True)

    # Call the function with and without saving
    plot_attention(example_text, dummy_phi)
