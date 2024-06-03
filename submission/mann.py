import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
        B, K_plus_one, N, _ = input_images.size()  # Extract size for reshaping

        # Prepare a tensor of zeros for the query set labels
        zeros_for_query = torch.zeros_like(input_labels[:, -1, :, :])

        # Concatenate the input labels with zeros for the last set
        labels_with_zeros = torch.cat([input_labels[:, :-1, :, :], zeros_for_query.unsqueeze(1)], dim=1)

        # Flatten the labels to match the input_images shape for concatenation
        labels_flattened = labels_with_zeros.reshape(B, K_plus_one * N, self.num_classes)

        # Concatenate images and labels along the feature dimension
        combined_input = torch.cat([input_images.reshape(B, K_plus_one * N, -1), labels_flattened], dim=-1)

        # Pass the combined input through the LSTM layers
        lstm1_output, _ = self.layer1(combined_input)
        lstm2_output, _ = self.layer2(lstm1_output)

        # Reshape the output from the last LSTM layer to match the expected output shape
        output = lstm2_output.view(B, K_plus_one, N, N)

        return output
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        # Isolate the query set predictions and labels
        B, _, N, _ = preds.size()
        # Isolate the query set predictions and labels
        query_preds = preds[:, -1, :, :]  # Shape: [B, N, N], focusing on the query set
        query_labels = labels[:, -1, :, :]  # Shape: [B, N, N], focusing on the query set

        # Convert the labels from one-hot to indices for CrossEntropyLoss
        _, query_labels_indices = query_labels.max(dim=2)  # Returns indices, shape: [B, N]

        # Ensure the predictions are reshaped correctly for CrossEntropyLoss
        # The correct shape should be [B*N, N] to match the target's [B*N]
        query_preds_reshaped = query_preds.reshape(B * N, N)  # Reshape to [B*N, N]
        query_labels_indices_reshaped = query_labels_indices.reshape(B*N)  # Flatten to [B*N]

        # Calculate the loss
        loss = F.cross_entropy(query_preds_reshaped, query_labels_indices_reshaped)
        ### END CODE HERE ###

        return loss
