import torch
from ResNet import ResNet
def load_model_and_predict(input_data):
    # Initialize the model
    model = ResNet()  # Replace with your actual model class and its required arguments if any

    # Load the state dict
    state_dict = torch.load("model1.pth")

    # Update the model parameters
    model.load_state_dict(state_dict)

    model.eval()

    # Ensure the input data is a tensor
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.from_numpy(input_data)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_data)

    # Get the predicted class
    prediction = torch.argmax(outputs, dim=1)

    return prediction.item()